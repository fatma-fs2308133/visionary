"""Studio render: our classical geometry decides WHERE, a pretrained diffusion
model (CatVTON, run locally) decides HOW it looks.

  1. pose + our DLT homography + inverse bilinear warp + TPS edge fit
     -> the new garment's target coverage T, with its length taken from the
        product's proportions (calibrated on VITON-HD ground truth)      (ours)
  2. coverage_masks (ours) separates what the garment covers from what has
     to change:
       garment region M = T grown + neck below the chin + arm corridors
       exposed region E = old clothing outside M (a cropped garment over a
                          long top, long sleeves under short ones, ...)
     - face above the chin, hair; visible trousers only where T covers them
  3. crop a 3:4 box around M + E and resize to the model resolution
  4. CatVTON repaints M + E, conditioned on the flat product photo. E is
     pinned to the person's own trousers (mirrored upward) below the
     waistband for most of the denoising (blended latent diffusion), so the
     model cannot extend the garment past it; skin above it is left to the
     model (coverage="mask" = previous behaviour: repaint old top + footprint)
  5. optional colour correction of the new garment toward the product
     photo (Lab chroma statistics)                                       (ours)
  6. paste back at full resolution with our feathered alpha blend, so
     pixels outside M + E are exactly the input photo                   (ours)
"""

import cv2
import numpy as np

from tryon_core.blend import feather_blend
from tryon_core.parsing import dilate, face_above_chin
from tryon_core.pipeline import TryOnOptions, try_on

SIZES = {"vitonhd": (384, 512),      # CatVTON VITON-HD checkpoint, native size
         "mix": (576, 768)}          # 1024-trained mix checkpoint; 1024x768 needs an 18 GiB attention buffer


def is_long_sleeved(garment) -> bool:
    """Long sleeves hang beside the lower torso, so the silhouette there is
    much wider than the torso itself. Compare the widest row at 55-80% of
    the garment height with the torso width between the hip anchors."""
    if garment.sleeves:                     # known from the catalogue
        return garment.sleeves == "long"
    m = garment.alpha > 0.5
    rows = np.flatnonzero(m.any(axis=1))
    top, h = rows[0], rows[-1] - rows[0] + 1
    band = m[top + int(0.55 * h): top + int(0.80 * h)]
    widths = [np.flatnonzero(r)[-1] - np.flatnonzero(r)[0] + 1 for r in band if r.any()]
    a = garment.anchors_px()
    torso = abs(a["leftHip"][0] - a["rightHip"][0]) / 0.68      # hips sit 16% inside each edge
    if not widths:
        return False
    full = np.flatnonzero(m.any(axis=0))
    full = full[-1] - full[0] + 1
    # long sleeves merged with the body: auto_anchors then sets the torso to
    # the central 64% of the width, and the lower rows span the whole garment
    merged = torso < 0.7 * full and max(widths) > 0.9 * full
    return max(widths) > 1.35 * torso or merged


def tryon_mask(person, pose, parsing, footprint, long_sleeves):
    """Region the model may repaint (H x W bool)."""
    h, w = person.shape[:2]
    kp = pose.keypoints
    ls, rs = np.array(kp["leftShoulder"][:2]), np.array(kp["rightShoulder"][:2])
    shoulder_w = float(np.linalg.norm(ls - rs))
    grow = max(3, int(0.04 * shoulder_w))

    m = dilate(parsing["upper"], grow) | (footprint > 0.3)

    # arm corridors: the new sleeves may be longer than the old ones, and a
    # sleeve is wider than the bare arm, so the corridor may include
    # background (otherwise the model can only paint skin there and turns a
    # long-sleeved garment into a short-sleeved one)
    arms = np.zeros((h, w), np.uint8)
    thick = max(6, int((0.42 if long_sleeves else 0.30) * shoulder_w))
    joints = ("Shoulder", "Elbow", "Wrist") if long_sleeves else ("Shoulder", "Elbow")
    lowest = []
    for side in ("left", "right"):
        pts = [kp.get(f"{side}{j}") for j in joints]
        pts = [p for p in pts if p and p[2] > 0.4]
        for a, b in zip(pts, pts[1:]):
            cv2.line(arms, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 1, thick)
        lowest += [p[1] for p in pts]
    m |= arms > 0

    # The garment may reach down to the lower of: old hem, sleeve ends, and
    # the NEW garment's predicted hem (its proportional footprint). Without
    # the footprint term the model can only fill the old top, so every new
    # garment would copy the old garment's length.
    rows_upper = np.flatnonzero(parsing["upper"].any(axis=1))
    rows_fp = np.flatnonzero((footprint > 0.3).any(axis=1))
    hem_new = rows_fp[-1] if len(rows_fp) else 0
    limit = max([rows_upper[-1] if len(rows_upper) else 0, hem_new] + lowest) + grow
    m[int(limit):] = False

    # neck: SegFormer's "face" label includes the neck. Everything of it below
    # the chin goes into the mask so the model can redraw neck + new neckline.
    face = face_above_chin(parsing, kp)
    neck = parsing["face"] & ~face
    neck &= np.arange(h)[:, None] < (ls[1] + rs[1]) / 2 + 0.1 * shoulder_w
    m |= neck

    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)) > 0
    protect = dilate(face, 2) | parsing["hair"]
    lower = cv2.erode(parsing["lower"].astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
    lower[: int(hem_new)] = False          # a longer garment may cover the top of the trousers
    return m & ~protect & ~lower


def crop_box(mask, h, w, aspect=3 / 4, margin=1.15):
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    bh = max((ys.max() - ys.min()) * margin, (xs.max() - xs.min()) * margin / aspect)
    bh = min(bh, h, w / aspect)
    bw = bh * aspect
    y0 = int(np.clip(cy - bh / 2, 0, h - bh))
    x0 = int(np.clip(cx - bw / 2, 0, w - bw))
    return y0, x0, int(bh), int(bw)


def garment_canvas(source_rgb, size):
    """Product photo resized and padded on white (CatVTON resize_and_padding)."""
    w, h = size
    s = min(w / source_rgb.shape[1], h / source_rgb.shape[0])
    r = cv2.resize(source_rgb, (int(source_rgb.shape[1] * s), int(source_rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), 255, np.uint8)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def match_garment_colour(out, region, garment_rgba, strength=0.6):
    """Move the a*/b* (chroma) mean and spread of the rendered garment toward
    the product photo's, leaving L* (folds and shading) untouched."""
    if region.sum() < 500:
        return out
    g_lab = cv2.cvtColor(garment_rgba[:, :, :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    g = g_lab[garment_rgba[:, :, 3] > 127]
    o_lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    o = o_lab[region]
    for c in (1, 2):
        gm, gs = g[:, c].mean(), g[:, c].std() + 1e-3
        om, os_ = o[:, c].mean(), o[:, c].std() + 1e-3
        scale = 1 + strength * (np.clip(gs / os_, 0.7, 1.4) - 1)
        o_lab[..., c][region] = (o[:, c] - om) * scale + om + strength * (gm - om)
    fixed = cv2.cvtColor(np.clip(o_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    soft = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 2)[..., None]
    return (fixed * soft + out * (1 - soft)).astype(np.uint8)


def coverage_masks(person, pose, parsing, target, long_sleeves):
    """Separate WHERE THE NEW GARMENT GOES from WHAT HAS TO CHANGE.

    target   T: the product silhouette placed by our proportional homography
                warp (+ sleeves), i.e. the garment's intended coverage
    garment  M: the region CatVTON may repaint = T grown by a small margin
                + neck below the chin + arm corridors (old sleeves become skin
                or new sleeves there; CatVTON was trained with arms in its mask)
    exposed  E: old clothing on the torso OUTSIDE M. It must become body,
                trousers or background, never garment. It is repainted too,
                but pinned to a skin/surroundings layout early in denoising.

    CatVTON fills its whole mask with garment, so giving it M instead of
    "old top + footprint" is what makes the hem, the length and the sleeve
    ends follow the product rather than the clothes the person had on.
    """
    h, w = person.shape[:2]
    kp = pose.keypoints
    ls, rs = np.array(kp["leftShoulder"][:2]), np.array(kp["rightShoulder"][:2])
    shoulder_w = float(np.linalg.norm(ls - rs))
    grow = max(3, int(0.04 * shoulder_w))
    t = target > 0.3

    def corridor(joints, thick):
        out = np.zeros((h, w), np.uint8)
        for side in ("left", "right"):
            pts = [kp.get(f"{side}{j}") for j in joints]
            pts = [q for q in pts if q and q[2] > 0.4]
            for a, b in zip(pts, pts[1:]):
                cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 1, thick)
        return out > 0

    joints = ("Shoulder", "Elbow", "Wrist") if long_sleeves else ("Shoulder", "Elbow")
    arms = corridor(joints, max(6, int((0.42 if long_sleeves else 0.30) * shoulder_w)))
    old = dilate(parsing["upper"], grow)
    old_sleeves = old & corridor(("Shoulder", "Elbow", "Wrist"), max(6, int(0.42 * shoulder_w)))

    face = face_above_chin(parsing, kp)
    neck = parsing["face"] & ~face
    neck &= np.arange(h)[:, None] < (ls[1] + rs[1]) / 2 + 0.1 * shoulder_w

    m = dilate(t, grow) | arms | old_sleeves | neck
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)) > 0
    protect = dilate(face, 2) | parsing["hair"]
    # Visible trousers stay untouched except where the new garment covers
    # them clearly (> 0.12 shoulder widths in that column): a sliver of
    # target over the waistband otherwise becomes a garment-coloured band
    lower = cv2.erode(parsing["lower"].astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
    covered = lower & dilate(t, grow)
    covered &= ((t & parsing["lower"]).sum(axis=0) > 0.12 * shoulder_w)[None, :]
    # ...and at most TROUSER_OVERLAP below the trousers' top edge: our length
    # estimate is uncertain by ~0.25 shoulder widths (calibrate_length.py), and
    # a too-long target makes CatVTON paint a skirt over the trousers
    top_edge = np.where(parsing["lower"].any(axis=0), np.argmax(parsing["lower"], axis=0), h)
    covered &= np.arange(h)[:, None] < top_edge[None, :] + TROUSER_OVERLAP * shoulder_w
    m &= ~protect & ~(lower & ~covered)
    exposed = old & ~m & ~protect
    exposed = cv2.morphologyEx(exposed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    return t, m, exposed


def exposed_anchor(person, exposed, target, parsing, kp):
    """Layout constraint for the exposed region (old clothing outside the new
    garment).

    Below the estimated waistband it must be the trousers: the person's own
    trouser pixels are mirrored upward about the old hem (real texture, not a
    flat colour) and pinned for most of the denoising, so the garment cannot
    grow past the waistband. Between the new hem and the waistband nothing is
    pinned: CatVTON renders skin there itself when the garment is cropped
    (tested: pinning a skin-coloured fill there turns into a beige "garment").
    Returns (anchor image, until map) for CatVTON(anchor=..., anchor_until=...)."""
    h, w = exposed.shape
    sw = abs(kp["leftShoulder"][0] - kp["rightShoulder"][0])
    hip_y = (kp["leftHip"][1] + kp["rightHip"][1]) / 2
    rows_t = np.flatnonzero(target.any(axis=1))
    waist = max(hip_y - WAIST_ABOVE_HIP * sw, (rows_t[-1] if len(rows_t) else 0) + 0.05 * sw)
    low = exposed & (np.arange(h)[:, None] >= waist)
    anchor = person.copy()
    until = np.zeros((h, w), np.float32)
    if low.any():
        cols = np.flatnonzero(low.any(axis=0))
        bottom = h - 1 - np.argmax(exposed[::-1, cols], axis=0)
        ys, xs = np.nonzero(low[:, cols])
        src = np.clip(2 * bottom[xs] - ys + 1, 0, h - 1)
        anchor[ys, cols[xs]] = person[src, cols[xs]]
        until[low] = TROUSER_PIN
    return anchor, until


TROUSER_OVERLAP = 0.25     # max garment overlap onto visible trousers, shoulder widths
WAIST_ABOVE_HIP = 0.25      # waistband height above the hip joints, in shoulder widths
TROUSER_PIN = 0.7           # share of denoising steps the trouser layout is pinned


def studio_render(person, pose, parsing, garment, source_rgb, model, steps=25, seed=42,
                  colour_fix=True, parser=None, progress=None, coverage="garment"):
    """Returns (output uint8, info dict with mask, timing, classical preview).

    coverage="garment": CatVTON repaints the garment's own target region; old
    clothing outside it is rebuilt first (coverage_masks).
    coverage="mask": previous behaviour, CatVTON repaints old top + footprint."""
    import time
    t0 = time.perf_counter()
    classical, info = try_on(person, pose, garment, TryOnOptions(debug=True), parsing=parsing)
    footprint = info.get("_alpha", np.zeros(person.shape[:2], np.float32))
    long_sleeves = is_long_sleeved(garment)
    exposed = target = None
    if coverage == "garment":
        target, mask, exposed = coverage_masks(person, pose, parsing, footprint, long_sleeves)
        region = mask | exposed
    else:
        mask = region = tryon_mask(person, pose, parsing, footprint, long_sleeves)
    h, w = person.shape[:2]
    y0, x0, bh, bw = crop_box(region, h, w)
    size = SIZES[model.checkpoint]

    def crop(a, interp=cv2.INTER_AREA):
        return cv2.resize(a[y0:y0 + bh, x0:x0 + bw], size, interpolation=interp)

    anchor = until = None
    cat_mask = mask
    if exposed is not None and exposed.any():
        # the exposed body is repainted too, but pinned to a trousers/skin
        # layout (exposed_anchor) so CatVTON cannot extend the garment into it
        cat_mask = mask | exposed
        anchor, until = exposed_anchor(person, exposed, target, parsing, pose.keypoints)
        anchor, until = crop(anchor), crop(until, cv2.INTER_NEAREST)
    # CatVTON renders the garment inside its own target region
    small, secs = model(crop(person), garment_canvas(source_rgb, size), crop(cat_mask.astype(np.uint8), cv2.INTER_NEAREST) > 0,
                        steps=steps, seed=seed, progress=progress, anchor=anchor, anchor_until=until)
    gen = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_CUBIC if bw > size[0] else cv2.INTER_AREA)
    layer = person.copy()
    layer[y0:y0 + bh, x0:x0 + bw] = gen

    uncorrected = layer.copy()
    if colour_fix and parser is not None:
        new_top = parser(layer)["upper"] & mask
        rgba = np.dstack([(garment.rgb * 255).astype(np.uint8), (garment.alpha * 255).astype(np.uint8)])
        layer = match_garment_colour(layer, new_top, rgba)

    # paste back: the soft edge lies OUTSIDE the repainted region, so every
    # repainted pixel is fully generated (no old fabric blended back in)
    r = max(2, int(0.006 * max(h, w)))
    alpha = dilate(region, r).astype(np.float32)
    out, _ = feather_blend(person.astype(np.float32) / 255, layer.astype(np.float32) / 255, alpha,
                           erode_radius=0, sigma=r / 2)
    out = (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)
    raw, _ = feather_blend(person.astype(np.float32) / 255, uncorrected.astype(np.float32) / 255, alpha,
                           erode_radius=0, sigma=r / 2)
    res = {"uncorrected": (np.clip(raw, 0, 1) * 255 + 0.5).astype(np.uint8), "mask": mask,
           "model_seconds": round(secs, 1), "seconds": round(time.perf_counter() - t0, 1),
           "classical": classical, "box": (y0, x0, bh, bw)}
    if target is not None:
        res.update(target=target, exposed=exposed)
    return out, res
