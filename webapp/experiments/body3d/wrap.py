"""Garment placement: our current 2D mapping vs wrapping around the measured 3D torso.

2D (current app): try_on() maps the product photo onto the torso quad between
the shoulder and hip keypoints (DLT homography + inverse bilinear warp). The
garment is scaled to the body, so its real size never matters.

3D (this experiment), built from the same from-scratch pieces:
  1. our DLT homography + inverse bilinear warp rectify the product photo into
     a METRIC flat panel (cm), using the garment's flat chest width from a size
     chart: the photo gives the shape, the size chart gives the scale;
  2. per height, the garment is a closed loop of circumference 2 x flat width
     around the reconstructed cross-section:
       larger than the body -> it stands off the body by the ease (uniform
                               offset: an approximation, real cloth would hang)
       smaller than the body -> it stretches (strain reported)
     the front panel covers the front half of the loop, from centre front
     to the side seams;
  3. for every output pixel we find the visible point of that loop (after the
     body's rotation), its arc length from centre front = panel x, and sample
     the panel with our bilinear remap;
  4. Lambert shading from the loop's surface normal, then our feathered blend.

This is GEOMETRY ONLY: no folds, no gravity, no collisions. That would need a
physically simulated drape of a 3D garment mesh (not available from a photo).
"""

import numpy as np

from tryon_core.blend import feather_blend
from tryon_core.homography import estimate_homography
from tryon_core.pipeline import TryOnOptions, try_on
from tryon_core.pose import PoseResult
from tryon_core.tps import remap_bilinear
from tryon_core.warp import inverse_warp_bilinear

from .body import rotate_y, superellipse
from .reconstruct import perimeter

PANEL_PX_PER_CM = 8.0


def metric_panel(garment, flat_chest_cm):
    """Rectify the product photo to a metric front panel. Returns (rgba,
    origin) with origin = panel pixel of (centre front, shoulder line)."""
    a = garment.anchors_px()
    chest_px = abs(a["leftArmpit"][0] - a["rightArmpit"][0])
    cm = flat_chest_cm / chest_px                         # cm per product pixel
    names = ("leftShoulder", "rightShoulder", "rightHip", "leftHip")
    src = np.array([a[k] for k in names])
    mid_x = (a["leftShoulder"][0] + a["rightShoulder"][0]) / 2
    sh_y = (a["leftShoulder"][1] + a["rightShoulder"][1]) / 2
    h, w = garment.alpha.shape
    size = (int(h * cm * PANEL_PX_PER_CM) + 40, int(w * cm * PANEL_PX_PER_CM) + 40)
    origin = np.array([size[1] / 2, 20 + sh_y * cm * PANEL_PX_PER_CM])
    # metric positions of the same anchors (flat-lay: a similarity; a photo
    # shot at an angle would make this a real perspective correction)
    dst = origin + (src - [mid_x, sh_y]) * cm * PANEL_PX_PER_CM
    hmat = estimate_homography(src, dst)
    stack = np.dstack([garment.rgb, garment.alpha]).astype(np.float32)
    return inverse_warp_bilinear(stack, hmat, size), origin


def place_2d(base_rgb, cam, body_joints, garment, yaw):
    """Current app: keypoints -> proportional DLT homography warp (+ sleeves).
    body_joints: unturned world joints (cm); the camera applies the yaw. The
    3D landmarks (as MediaPipe's world landmarks would) carry the yaw."""
    kp = {k: (*cam.project(np.array([v]))[0][:2], 1.0) for k, v in body_joints.items()}
    world = {k: tuple(rotate_y(np.array([v]), yaw)[0] / 100) for k, v in body_joints.items()}
    opts = TryOnOptions(tps=False, occlusion=False, debug=True)
    out, info = try_on((base_rgb * 255).astype(np.uint8), PoseResult(kp, None, world), garment, opts)
    return out.astype(np.float32) / 255, info


def place_3d(base_rgb, base_depth, cam, torso, panel, origin, flat_chest_cm, shoulder_y, yaw,
             light=(0.3, 0.5, 1.0)):
    """Wrap the metric panel around the torso cross-sections, render from cam."""
    h, w = base_depth.shape
    mx = np.full((h, w), -1.0, np.float32)
    my = np.full((h, w), -1.0, np.float32)
    shade = np.ones((h, w), np.float32)
    back = np.zeros((h, w), bool)
    strain = {}
    l = np.array(light) / np.linalg.norm(light)
    garment_girth = 2 * flat_chest_cm
    for row in range(h):
        y = cam.top_cm - (row + 0.5) / cam.px_per_cm
        v_cm = shoulder_y - y                               # panel cm below the shoulder line
        pv = origin[1] + v_cm * PANEL_PX_PER_CM
        if pv < 0 or pv >= panel.shape[0] - 1 or y < torso.ys[0]:
            continue
        a, b, c, n = torso.section(min(y, shoulder_y))      # above the shoulders it hangs from them
        body_p = perimeter(a, b, n)
        if garment_girth >= body_p:                         # ease: stands off the body
            lo, hi = 0.0, 20.0
            for _ in range(30):
                d = (lo + hi) / 2
                lo, hi = (d, hi) if perimeter(a + d, b + d, n) < garment_girth else (lo, d)
            d, stretch = (lo + hi) / 2, 1.0
        else:                                               # too small: stretched over the body
            d, stretch = 0.0, body_p / garment_girth
        strain[round(y)] = stretch - 1
        x, z = superellipse(a + d, b + d, n, 1440)
        s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(z)))]
        s = np.where(s > s[-1] / 2, s - s[-1] - np.hypot(x[0] - x[-1], z[0] - z[-1]), s)   # signed from centre front
        p = rotate_y(np.c_[x, np.full_like(x, y), z + c], yaw)
        tx, tz = np.gradient(p[:, 0]), np.gradient(p[:, 2])
        nx, nz = tz, -tx                                    # outward normal (loop runs counter-clockwise seen from above)
        if nz[0] < 0:
            nx, nz = -nx, -nz
        vis = nz > 1e-3
        if not vis.any():
            continue
        xi = cam.width / 2 + p[vis, 0] * cam.px_per_cm
        order = np.argsort(xi)
        cols = np.arange(int(np.ceil(xi.min())), int(xi.max()) + 1)
        cols = cols[(cols >= 0) & (cols < w)]
        s_at = np.interp(cols, xi[order], s[vis][order]) / stretch     # panel cm from centre front
        z_at = np.interp(cols, xi[order], p[vis, 2][order])
        nrm = np.interp(cols, xi[order], (nx * l[0] + nz * l[2])[vis][order] / np.hypot(nx, nz)[vis][order])
        front = z_at >= base_depth[row, cols] - 0.5        # not hidden by an arm
        cols, s_at, nrm = cols[front], s_at[front], nrm[front]
        is_back = np.abs(s_at) > flat_chest_cm / 2         # past the side seams: back panel
        u = origin[0] + np.clip(s_at, -flat_chest_cm / 2, flat_chest_cm / 2) * PANEL_PX_PER_CM
        mx[row, cols], my[row, cols] = u, pv
        shade[row, cols] = 0.45 + 0.55 * np.clip(nrm, 0, 1)
        back[row, cols] = is_back
    layer = remap_bilinear(panel, mx, my)
    rgb, alpha = layer[..., :3], layer[..., 3]
    # the product photo shows only the front: the back panel gets its mean colour
    mean = panel[..., :3][panel[..., 3] > 0.5].mean(axis=0)
    rgb[back] = mean * 0.9
    out, final_a = feather_blend(base_rgb.astype(np.float32), rgb * shade[..., None], alpha, erode_radius=1, sigma=0.8)
    return out, final_a, {"strain": strain, "mx": mx, "my": my}
