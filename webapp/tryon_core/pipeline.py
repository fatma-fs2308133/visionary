"""One frame of the classical try-on (port of tryOnPipeline.m).

1. torso warp      homography from the 4 torso points (from scratch)
2. sleeves         each sleeve rotates with its upper arm
3. edge fit        TPS pulls the garment edges onto the person's real
                   clothing outline (from scratch, needs parsing)
4. body fit        occlusion (face, hair and forearms stay in front) and no
                   spill onto the background (needs parsing); without
                   parsing: clip to the MediaPipe person mask
5. lighting match  garment brightness/contrast -> scene
6. shading         copy the folds/roundness of the real clothes
7. feathered blend (from scratch)
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .blend import feather_blend
from .garment import Garment
from .homography import DegenerateError, estimate_homography
from .parsing import dilate, face_above_chin, forearms
from .pose import PoseResult
from .tps import apply_tps, fit_tps, remap_bilinear
from .warp import inverse_warp_bilinear


@dataclass
class TryOnOptions:
    width: float = 1.15          # garment width relative to the shoulder/hip points
    length: float = 1.0          # garment length relative to shoulder->hip
    sleeves: bool = True
    fit: bool = True
    proportional: bool = True    # garment length from the product's own proportions (not the hips)
    tps: bool = True             # TPS edge fit (only when parsing is given)
    curvature: float = 0.0       # elliptical-cylinder torso wrap (tested: marginal gain, adds seams; off)
    occlusion: bool = True       # face/hair/forearms in front (only with parsing)
    lighting: float = 0.3        # 0 = off
    shading: float = 0.45        # 0 = off
    feather: float = 2.0         # Gaussian sigma at 640 px width; 0 = hard edge
    min_visibility: float = 0.5
    debug: bool = False          # also return an image of every stage (info["stages"])

    @classmethod
    def from_dict(cls, d: dict | None) -> "TryOnOptions":
        o = cls()
        for k, v in (d or {}).items():
            if hasattr(o, k):
                setattr(o, k, type(getattr(o, k))(v))
        return o


TORSO = ("leftShoulder", "rightShoulder", "rightHip", "leftHip")


def try_on(frame_rgb: np.ndarray, pose: PoseResult, garment: Garment, opts: TryOnOptions,
           parsing: dict | None = None):
    """frame_rgb: H x W x 3 uint8. parsing: optional dict from tryon_core.parsing.
    Returns (out uint8, info dict)."""
    t0 = time.perf_counter()
    info = {"ok": False, "reason": "", "confidence": pose.confidence, "timing": {}}
    kp = pose.keypoints
    if not kp:
        info["reason"] = "No person found. Face the camera with shoulders and hips visible."
        return frame_rgb, info
    if any(kp[k][2] < opts.min_visibility for k in TORSO):
        info["reason"] = "Shoulders or hips not visible enough. Step back so your hips are in view."
        return frame_rgb, info

    h, w = frame_rgb.shape[:2]
    px = w / 640                                   # pixel parameters are tuned at 640 px
    frame = frame_rgb.astype(np.float32) / 255
    a = garment.anchors_px()
    timing = {}

    # ---- 1. torso warp ---------------------------------------------------
    t = time.perf_counter()
    if opts.proportional:
        dst = _proportional_torso(kp, a, opts.width, opts.length, pose.torso_yaw())
    else:
        dst = _scaled_torso(kp, opts.width, opts.length)
    src = np.array([a[k] for k in TORSO])
    try:
        hmat = estimate_homography(src, dst)
    except DegenerateError:
        info["reason"] = "Body points look crossed or squashed. Turn to face the camera."
        return frame_rgb, info

    # Sleeves are cut off the torso layer when they get their own transform
    has_elbow = {s: kp[f"{s}Elbow"][2] >= opts.min_visibility for s in ("left", "right")}
    sleeve_alpha = {}
    torso_alpha = garment.alpha
    if opts.sleeves:
        parts = _split_sleeves(garment)
        for s in ("left", "right"):
            if has_elbow[s]:
                sleeve_alpha[s] = parts[s]
                torso_alpha = torso_alpha * parts[f"{s}Keep"]
    stack = np.dstack([garment.rgb, torso_alpha]).astype(np.float32)
    warped = inverse_warp_bilinear(stack, hmat, (h, w))
    layer_rgb, layer_a = warped[:, :, :3], warped[:, :, 3]
    timing["warp"] = time.perf_counter() - t

    # ---- 2. sleeves follow the upper arms -------------------------------
    t = time.perf_counter()
    scale = np.linalg.norm(dst[0] - dst[1]) / max(np.linalg.norm(a["leftShoulder"] - a["rightShoulder"]), 1e-6)
    for s, s_alpha in sleeve_alpha.items():
        res = _warp_sleeve(garment.rgb, s_alpha, a[f"{s}Shoulder"], a[f"{s}SleeveEnd"],
                           np.array(kp[f"{s}Shoulder"][:2]), np.array(kp[f"{s}Elbow"][:2]), scale, (h, w))
        if res is None:
            continue
        s_rgb, s_a, (y0, y1, x0, x1) = res
        box_rgb, box_a = layer_rgb[y0:y1, x0:x1], layer_a[y0:y1, x0:x1]
        box_rgb[:] = s_rgb * s_a[..., None] + box_rgb * (1 - s_a[..., None])
        box_a[:] = s_a + box_a * (1 - s_a)
    timing["sleeves"] = time.perf_counter() - t
    if opts.debug:                               # for evaluation / hybrid prototypes
        info["_layer"] = (layer_rgb.copy(), layer_a.copy(), dst.copy())

    # ---- 3. TPS edge fit onto the person's clothing outline ---------------
    t = time.perf_counter()
    if parsing is not None and opts.tps:
        layer_rgb, layer_a = tps_edge_fit(layer_rgb, layer_a, parsing["upper"] & ~parsing["arms"], kp)
    timing["tps"] = time.perf_counter() - t

    # ---- 3b. torso curvature: the flat garment wraps around the body -------
    t = time.perf_counter()
    if opts.curvature > 0:
        layer_rgb, layer_a = cylinder_wrap(layer_rgb, layer_a, kp, pose.torso_yaw(), opts.curvature)
    timing["curvature"] = time.perf_counter() - t

    # The remaining stages only need the area around the garment
    rows = np.flatnonzero(layer_a.max(axis=1) > 0.01)
    cols = np.flatnonzero(layer_a.max(axis=0) > 0.01)
    if not len(rows):
        info["reason"] = "The garment landed outside the picture."
        return frame_rgb, info
    pad = int(12 * px)
    r1, r2 = max(0, rows[0] - pad), min(h, rows[-1] + pad + 1)
    c1, c2 = max(0, cols[0] - pad), min(w, cols[-1] + pad + 1)
    full_frame = frame
    frame = frame[r1:r2, c1:c2]
    layer_rgb, layer_a = layer_rgb[r1:r2, c1:c2], layer_a[r1:r2, c1:c2]
    dst_crop = dst - [c1, r1]
    stages = {}
    if opts.debug:
        stages["warp"] = _on_grey(layer_rgb, layer_a)

    # ---- 3. body fit: the garment never covers the background ------------
    t = time.perf_counter()
    if parsing is not None and opts.occlusion:
        crop = (slice(r1, r2), slice(c1, c2))
        front = face_above_chin(parsing, kp) | parsing["hair"] | forearms(parsing, kp)
        keep = ~dilate(front, max(1, round(px)))[crop]
        keep &= dilate(~parsing["background"], max(1, round(2 * px)))[crop]
        layer_a = layer_a * cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 0.8 * px)
        if opts.debug:
            stages["mask"] = (np.dstack([keep] * 3) * 255).astype(np.uint8)
    elif opts.fit and pose.mask is not None:
        body = (pose.mask[r1:r2, c1:c2] > 0.5).astype(np.uint8)
        shoulder_w = np.linalg.norm(dst[0] - dst[1])
        r = max(1, int(0.05 * shoulder_w))
        body = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        body = cv2.GaussianBlur(body.astype(np.float32), (0, 0), 1.5 * px)
        layer_a = layer_a * body
        if opts.debug:
            stages["mask"] = (np.dstack([body] * 3) * 255).astype(np.uint8)
    timing["fit"] = time.perf_counter() - t

    # ---- 4. lighting match -----------------------------------------------
    t = time.perf_counter()
    if opts.lighting > 0:
        layer_rgb = _match_lighting(layer_rgb, layer_a, frame, dst_crop, opts.lighting)
    timing["lighting"] = time.perf_counter() - t

    # ---- 5. shading transfer ---------------------------------------------
    t = time.perf_counter()
    if opts.shading > 0:
        layer_rgb = _transfer_shading(layer_rgb, layer_a, frame, opts.shading, px)
    timing["shading"] = time.perf_counter() - t
    if opts.debug:
        stages["light"] = _on_grey(layer_rgb, layer_a)

    # ---- 6. feathered blend ----------------------------------------------
    t = time.perf_counter()
    out_crop, final_a = feather_blend(frame, layer_rgb, layer_a, erode_radius=max(1, round(3 * px)),
                                sigma=opts.feather * px)
    out = full_frame.copy()
    out[r1:r2, c1:c2] = out_crop
    timing["blend"] = time.perf_counter() - t
    if opts.debug:
        stages["alpha"] = (np.dstack([final_a] * 3) * 255).astype(np.uint8)
        stages["result"] = (np.clip(out_crop, 0, 1) * 255).astype(np.uint8)
        info["stages"] = stages
        alpha_full = np.zeros((h, w), np.float32)
        alpha_full[r1:r2, c1:c2] = final_a
        info["_alpha"] = alpha_full              # for evaluation, not sent to the browser

    timing["total"] = time.perf_counter() - t0
    info.update(ok=True, timing={k: round(v * 1000, 1) for k, v in timing.items()},
                torso=dst.round(1).tolist())
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), info


# ---------------------------------------------------------------------------
def _run(row, xc):
    """The run of True pixels in a row that contains (or is nearest to) column xc."""
    xs = np.flatnonzero(row)
    if not len(xs):
        return None
    splits = np.flatnonzero(np.diff(xs) > 1)
    starts, ends = np.r_[xs[0], xs[splits + 1]], np.r_[xs[splits], xs[-1]]
    dist = np.where((starts <= xc) & (ends >= xc), 0, np.minimum(abs(starts - xc), abs(ends - xc)) + 1)
    k = int(np.argmin(dist))
    return starts[k], ends[k]


def tps_edge_fit(layer_rgb, layer_a, target, kp, lam=0.5):
    """Pull the warped garment's left/right edges onto the target outline.

    On 9 rows between chest and hips the garment edge (from the homography
    warp) and the clothing edge (from parsing) give point pairs; the torso
    centre line, the shoulders and the frame corners are fixed points. A
    smoothing TPS (tps.py) is fitted as a BACKWARD map (target -> garment)
    and the layer is resampled with our bilinear interpolation.
    """
    h, w = layer_a.shape
    ls, rs = np.array(kp["leftShoulder"][:2]), np.array(kp["rightShoulder"][:2])
    lh, rh = np.array(kp["leftHip"][:2]), np.array(kp["rightHip"][:2])
    y_sh, y_hip = (ls[1] + rs[1]) / 2, (lh[1] + rh[1]) / 2
    xc = (ls[0] + rs[0] + lh[0] + rh[0]) / 4
    g = layer_a > 0.5
    src, dst = [], []
    for y in np.linspace(y_sh + 0.35 * (y_hip - y_sh), y_hip, 9).astype(int):
        if not 0 <= y < h:
            continue
        a, b = _run(g[y], xc), _run(target[y], xc)
        if a is None or b is None:
            continue
        span = b[1] - b[0]
        # ignore rows where parsing glitches or an arm at the side would pull too far
        if abs(a[0] - b[0]) < 0.25 * span and abs(a[1] - b[1]) < 0.25 * span:
            dst += [(b[0], y), (b[1], y)]
            src += [(a[0], y), (a[1], y)]
    if len(dst) < 6:
        return layer_rgb, layer_a
    fixed = [(xc, y) for y in np.linspace(y_sh, y_hip, 5)] + [tuple(ls), tuple(rs)]
    fixed += [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    tps = fit_tps(np.array(dst + fixed, float), np.array(src + fixed, float), lam)
    # Region to resample: bounding box of garment + target, with a margin
    rows = np.flatnonzero((g | target).any(axis=1))
    cols = np.flatnonzero((g | target).any(axis=0))
    pad = max(4, w // 40)
    y0, y1 = max(0, rows[0] - pad), min(h, rows[-1] + pad + 1)
    x0, x1 = max(0, cols[0] - pad), min(w, cols[-1] + pad + 1)
    # The TPS is smooth, so evaluate it on a coarse grid (every 8 px) and
    # interpolate the map in between: same result to a fraction of a pixel,
    # ~50x fewer kernel evaluations (needed for live video)
    step = 8
    gy = np.arange(y0, y1 + step, step, dtype=float)
    gx = np.arange(x0, x1 + step, step, dtype=float)
    gxx, gyy = np.meshgrid(gx, gy)
    coarse = apply_tps(tps, np.c_[gxx.ravel(), gyy.ravel()]).reshape(len(gy), len(gx), 2).astype(np.float32)
    # Bilinear upsampling of a regular grid is separable: fine = Wy . coarse . Wx^T
    # with linear-interpolation weight matrices (two small matrix products)
    wy, wx = _interp_matrix(y1 - y0, len(gy), step), _interp_matrix(x1 - x0, len(gx), step)
    fine = np.stack([wy @ coarse[..., c] @ wx.T for c in (0, 1)], axis=-1)
    stack = np.dstack([layer_rgb, layer_a]).astype(np.float32)
    out = np.zeros_like(stack)
    out[y0:y1, x0:x1] = remap_bilinear(stack, fine[..., 0], fine[..., 1])
    return out[..., :3], out[..., 3]


def cylinder_wrap(layer_rgb, layer_a, kp, yaw, strength=0.5, shade=0.25):
    """Model the torso as an elliptical cylinder seen from the front.

    A point at angle theta on the cylinder appears at x = c + r sin(theta),
    so equal steps along the cloth get closer together toward the sides.
    For every output pixel of the torso band, theta = arcsin((x - c) / r)
    and the cloth position is c + r * theta / (pi/2); `strength` blends this
    with the flat (identity) map. The visible centre of the cylinder shifts
    by r sin(yaw) when the person is turned (yaw from the 3D landmarks).
    Lambert shading: brightness falls off as cos(theta) toward the sides.
    Sleeves outside the torso band are left unchanged.
    """
    h, w = layer_a.shape
    ls, rs = np.array(kp["leftShoulder"][:2]), np.array(kp["rightShoulder"][:2])
    lh, rh = np.array(kp["leftHip"][:2]), np.array(kp["rightHip"][:2])
    rows = np.flatnonzero((layer_a > 0.5).any(axis=1))
    if not len(rows):
        return layer_rgb, layer_a
    y0, y1 = int(min(ls[1], rs[1])), int(rows[-1]) + 1
    if y1 <= y0:
        return layer_rgb, layer_a
    ys = np.arange(y0, y1)
    # torso centre line and half-width per row (shoulders -> hips, extended below)
    f = np.clip((ys - (ls[1] + rs[1]) / 2) / max((lh[1] + rh[1]) / 2 - (ls[1] + rs[1]) / 2, 1), 0, None)[:, None]
    left = ls + (lh - ls) * np.minimum(f, 1.5)
    right = rs + (rh - rs) * np.minimum(f, 1.5)
    xl, xr = np.minimum(left[:, 0], right[:, 0]), np.maximum(left[:, 0], right[:, 0])
    r = (xr - xl) / 2 * 1.15                       # body half-width (shoulder/hip joints are inside the outline)
    c = (xl + xr) / 2 + r * np.sin(np.clip(yaw, -0.8, 0.8))
    xs = np.arange(w, dtype=np.float32)[None]
    t = np.clip((xs - c[:, None]) / r[:, None], -0.999, 0.999)
    theta = np.arcsin(t)
    src_x = (c[:, None] + r[:, None] * ((1 - strength) * t + strength * theta / (np.pi / 2))).astype(np.float32)
    inside = np.abs((xs - c[:, None]) / r[:, None]) < 1
    src_x = np.where(inside, src_x, xs)
    src_y = np.broadcast_to(ys[:, None].astype(np.float32), src_x.shape)
    stack = np.dstack([layer_rgb, layer_a]).astype(np.float32)
    band = remap_bilinear(stack, src_x, src_y)
    light = np.where(inside, 1 - shade * strength * (1 - np.cos(theta)), 1).astype(np.float32)
    out_rgb, out_a = layer_rgb.copy(), layer_a.copy()
    out_rgb[y0:y1] = band[..., :3] * light[..., None]
    out_a[y0:y1] = band[..., 3]
    return out_rgb, out_a


def _interp_matrix(n_fine, n_coarse, step):
    """n_fine x n_coarse matrix of linear-interpolation weights: fine sample i
    lies at i/step in coarse units, between coarse samples k and k+1."""
    pos = np.arange(n_fine, dtype=np.float32) / step
    k = np.minimum(pos.astype(int), n_coarse - 2)
    t = pos - k
    w = np.zeros((n_fine, n_coarse), np.float32)
    w[np.arange(n_fine), k] = 1 - t
    w[np.arange(n_fine), k + 1] = t
    return w


def _on_grey(rgb, alpha):
    a = alpha[..., None]
    return (np.clip(rgb * a + 0.9 * (1 - a), 0, 1) * 255).astype(np.uint8)


CHEST_PER_SHOULDER = 0.77


def _proportional_torso(kp, a, width, length, yaw):
    """Destination quad that keeps the PRODUCT's proportions.

    The old mapping (_scaled_torso) sends the garment's hem anchors to the
    person's hip joints, which stretches a cropped top and squashes a long
    one. Here one scale factor, body shoulder width / garment shoulder width,
    sets both width and length: the hem lands at shoulder line + (garment
    shoulder-to-hem distance x scale) along the torso axis. The scale comes
    from the garment's chest width (see below).

    The 2D shoulder width shrinks by cos(yaw) when the person is turned, so
    it is corrected with the torso yaw from MediaPipe's 3D world landmarks
    (clamped at 60 degrees).
    """
    ls, rs, rh, lh = (np.array(kp[k][:2], dtype=float) for k in TORSO)
    sm, hm = (ls + rs) / 2, (lh + rh) / 2
    axis = hm - sm
    d = axis / max(np.linalg.norm(axis), 1e-6)
    g_sm = (a["leftShoulder"] + a["rightShoulder"]) / 2
    g_hm = (a["leftHip"] + a["rightHip"]) / 2
    # Scale reference: the garment's chest width (between the armpit anchors).
    # The automatic "shoulder" anchors sit near the neckline (~0.73 of the seam
    # width), which made every garment ~35% too long. The chest width is
    # mapped to CHEST_PER_SHOULDER x body shoulder width, fitted on 30 VITON-HD
    # test models wearing the paired product (eval/calibrate_length.py).
    # Implausibly narrow armpit anchors fall back to the shoulder anchors
    # (median armpit/shoulder-anchor ratio 1.09).
    g_chest = abs(a["leftArmpit"][0] - a["rightArmpit"][0])
    g_shoulders = np.linalg.norm(a["leftShoulder"] - a["rightShoulder"])
    if g_chest < 0.8 * g_shoulders:
        g_chest = 1.09 * g_shoulders
    g_width = g_chest / CHEST_PER_SHOULDER
    g_len = np.linalg.norm(g_hm - g_sm)
    b_width = np.linalg.norm(ls - rs) / max(np.cos(yaw), 0.5)
    scale = b_width / max(g_width, 1e-6) * width
    hem = sm + d * g_len * scale * length
    ls, rs = sm + (ls - sm) * width, sm + (rs - sm) * width
    lh, rh = hem + (lh - hm) * width, hem + (rh - hm) * width
    return np.array([ls, rs, rh, lh])


def _scaled_torso(kp, width, length):
    ls, rs, rh, lh = (np.array(kp[k][:2], dtype=float) for k in TORSO)
    sm, hm = (ls + rs) / 2, (lh + rh) / 2
    ls, rs = sm + (ls - sm) * width, sm + (rs - sm) * width
    lh, rh = hm + (lh - hm) * width, hm + (rh - hm) * width
    down = (hm - sm) * (length - 1)
    return np.array([ls, rs, rh + down, lh + down])


_SLEEVE_CACHE: dict = {}


def _split_sleeves(garment: Garment) -> dict:
    """Sleeve alphas + torso keep-masks, computed once per garment."""
    key = (garment.id, id(garment.alpha))
    if key not in _SLEEVE_CACHE:
        a = garment.anchors_px()
        parts = {}
        for s in ("left", "right"):
            d = _outer_distance(garment.alpha.shape, a[f"{s}Shoulder"], a[f"{s}Armpit"], a[f"{s}SleeveEnd"])
            parts[s] = (garment.alpha * np.clip((d + 2) / 4, 0, 1)).astype(np.float32)
            parts[f"{s}Keep"] = (d <= 3).astype(np.float32)
        _SLEEVE_CACHE[key] = parts
    return _SLEEVE_CACHE[key]


def _outer_distance(shape, a, b, outer):
    """Signed distance to the shoulder->armpit line, positive on the sleeve side."""
    hh, ww = shape
    xs, ys = np.meshgrid(np.arange(ww), np.arange(hh))
    d_ab = (b - a) / max(np.linalg.norm(b - a), 1e-6)
    normal = np.array([-d_ab[1], d_ab[0]])
    d = (xs - a[0]) * normal[0] + (ys - a[1]) * normal[1]
    if np.dot(outer - a, normal) < 0:
        d = -d
    # The line continues past the shoulder into the collar: only the part
    # beyond the shoulder (horizontally, toward the sleeve) is sleeve.
    side = np.sign(outer[0] - a[0]) or 1.0
    return np.minimum(d, side * (xs - a[0]))


def _warp_sleeve(rgb, alpha, g_sh, g_end, f_sh, f_elbow, scale, out_hw):
    g_axis, arm = g_end - g_sh, f_elbow - f_sh
    if np.linalg.norm(g_axis) < 1 or np.linalg.norm(arm) < 1:
        return None
    f_end = f_sh + np.linalg.norm(g_axis) * scale * arm / np.linalg.norm(arm)

    def perp(v):
        return np.array([-v[1], v[0]])

    src = np.float32([g_sh, g_end, g_sh + perp(g_axis)])
    dst = np.float32([f_sh, f_end, f_sh + perp(f_end - f_sh)])
    m = cv2.getAffineTransform(src, dst)

    # Only render the box the sleeve lands in
    ys, xs = np.nonzero(alpha > 0.01)
    if not len(xs):
        return None
    corners = np.c_[[xs.min(), xs.max(), xs.min(), xs.max()], [ys.min(), ys.min(), ys.max(), ys.max()], np.ones(4)] @ m.T
    x0, y0 = np.maximum(np.floor(corners.min(axis=0)).astype(int) - 1, 0)
    x1 = min(out_hw[1], int(np.ceil(corners[:, 0].max())) + 2)
    y1 = min(out_hw[0], int(np.ceil(corners[:, 1].max())) + 2)
    if x1 <= x0 or y1 <= y0:
        return None
    m[:, 2] -= [x0, y0]
    size = (x1 - x0, y1 - y0)
    s_rgb = cv2.warpAffine(rgb, m, size, flags=cv2.INTER_LINEAR, borderValue=0)
    s_a = cv2.warpAffine(alpha, m, size, flags=cv2.INTER_LINEAR, borderValue=0)
    return s_rgb, np.clip(s_a, 0, 1), (y0, y1, x0, x1)


def _match_lighting(layer, alpha, frame, torso, strength, shrink=0.15):
    """Move the garment's brightness (V) mean/std part of the way to the torso's."""
    h, w = alpha.shape
    x0, y0 = torso.min(axis=0)
    x1, y1 = torso.max(axis=0)
    dx, dy = (x1 - x0) * shrink, (y1 - y0) * shrink
    c1, c2 = int(np.clip(x0 + dx, 0, w - 1)), int(np.clip(x1 - dx, 0, w - 1))
    r1, r2 = int(np.clip(y0 + dy, 0, h - 1)), int(np.clip(y1 - dy, 0, h - 1))
    scene_v = frame[r1:r2 + 1, c1:c2 + 1].max(axis=2).ravel()
    inside = alpha >= 0.5
    if scene_v.size < 10 or inside.sum() < 10:
        return layer
    garment_v = layer.max(axis=2)[inside]
    mu_g, sd_g = garment_v.mean(), garment_v.std()
    mu_s, sd_s = scene_v.mean(), scene_v.std()
    gain = 1 + strength * (np.clip(sd_s / max(sd_g, 1e-6), 0.7, 1.4) - 1)
    target = mu_g + strength * (mu_s - mu_g)
    return np.clip(gain * layer + (target - gain * mu_g), 0, 1)


def _transfer_shading(layer, alpha, frame, strength, px, base_sigma=15, fine_sigma=3.5):
    """shade = L / local mean of L inside the garment, so only the folds and
    the body's roundness are copied, not the colour of the real shirt.
    fine_sigma blurs L first so logos and prints on the real shirt do not
    show through (only broad folds survive)."""
    lum = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    lum = cv2.GaussianBlur(lum, (0, 0), fine_sigma * px)
    m = (alpha > 0.05).astype(np.float32)
    if not m.any():
        return layer
    num = cv2.GaussianBlur(lum * m, (0, 0), base_sigma * px)
    den = cv2.GaussianBlur(m, (0, 0), base_sigma * px)
    base = num / np.maximum(den, 1e-3)
    shade = 1 + strength * (lum / np.maximum(base, 0.02) - 1)
    shade = np.clip(shade, 0.5, 1.4)
    shade[m == 0] = 1
    return np.clip(layer * shade[..., None], 0, 1)
