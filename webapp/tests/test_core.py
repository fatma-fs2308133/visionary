"""From-scratch parts vs OpenCV, plus pipeline smoke tests.

    .venv/bin/python -m tests.test_core
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tryon_core.blend import convolve_separable, erode_disk, feather_blend, gaussian_kernel_1d  # noqa: E402
from tryon_core.garment import Garment, auto_anchors  # noqa: E402
from tryon_core.homography import DegenerateError, apply_homography, estimate_homography  # noqa: E402
from tryon_core.pipeline import TryOnOptions, try_on  # noqa: E402
from tryon_core.pose import PoseResult  # noqa: E402
from tryon_core.tps import apply_tps, fit_tps, remap_bilinear  # noqa: E402
from tryon_core.warp import inverse_warp_bilinear  # noqa: E402

failures = 0


def check(ok, name):
    global failures
    print(("PASS  " if ok else "FAIL  ") + name)
    failures += not ok


rng = np.random.default_rng(7)

# ---- homography ----------------------------------------------------------
src = np.array([[100, 60], [300, 60], [270, 390], [130, 390]], float)
h_true = np.array([[1.1, 0.1, 40], [-0.05, 0.95, 20], [2e-4, -1e-4, 1]])
dst = apply_homography(h_true, src)
h = estimate_homography(src, dst)
h_cv, _ = cv2.findHomography(src, dst, 0)
check(np.abs(apply_homography(h, src) - dst).max() < 1e-9, "homography: exact on 4 points")
grid = np.c_[rng.uniform(0, 400, 50), rng.uniform(0, 400, 50)]
d_cv = np.abs(apply_homography(h, grid) - apply_homography(h_cv, grid)).max()
check(d_cv < 1e-3, f"homography: same mapping as cv2.findHomography (max {d_cv:.1e} px)")

src8 = np.r_[src, (src + np.roll(src, 1, axis=0)) / 2]
dst8 = apply_homography(h_true, src8) + rng.normal(0, 1, (8, 2))
h8, h8_cv = estimate_homography(src8, dst8), cv2.findHomography(src8, dst8, 0)[0]
test = np.c_[np.linspace(110, 290, 20), np.linspace(70, 380, 20)]
err_own = np.sqrt(np.mean(np.sum((apply_homography(h8, test) - apply_homography(h_true, test)) ** 2, 1)))
err_cv = np.sqrt(np.mean(np.sum((apply_homography(h8_cv, test) - apply_homography(h_true, test)) ** 2, 1)))
check(abs(err_own - err_cv) < 0.05, f"homography: noisy least squares RMSE {err_own:.3f} px vs OpenCV {err_cv:.3f} px")

try:
    estimate_homography(np.array([[0, 0], [1, 1], [2, 2], [0, 5]], float) * 50, dst)
    check(False, "homography: collinear points rejected")
except DegenerateError:
    check(True, "homography: collinear points rejected")

# ---- inverse warp ----------------------------------------------------------
img = rng.random((200, 160, 4)).astype(np.float32)
img = cv2.GaussianBlur(img, (0, 0), 2)
corners = np.array([[0, 0], [159, 0], [159, 199], [0, 199]], float)
hw = estimate_homography(corners, np.array([[180, 120], [430, 140], [410, 420], [200, 400]], float))
own = inverse_warp_bilinear(img, hw, (480, 640))
ref = cv2.warpPerspective(img, hw, (640, 480), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
inner = cv2.erode(((own[:, :, 3] > 0) & (ref[:, :, 3] > 0)).astype(np.uint8), np.ones((3, 3))) > 0
diff = np.abs(own - ref)[inner]
check(diff.mean() < 2e-3, f"warp: matches cv2.warpPerspective (mean diff {diff.mean():.1e})")

# ---- blend -----------------------------------------------------------------
mask = cv2.GaussianBlur(rng.random((120, 160)), (0, 0), 4) > 0.5
mask[:, :5] = True
ok = all(np.array_equal(erode_disk(mask, r),
                        cv2.erode(mask.astype(np.uint8),
                                  (np.add.outer(np.arange(-r, r + 1) ** 2, np.arange(-r, r + 1) ** 2) <= r * r).astype(np.uint8),
                                  borderType=cv2.BORDER_CONSTANT, borderValue=1) > 0)
         for r in range(1, 6))
check(ok, "blend: erode_disk == cv2.erode with a disk (radius 1..5)")

x = rng.random((90, 110)).astype(np.float32)
k = gaussian_kernel_1d(2.0)
ref = cv2.sepFilter2D(x, -1, k, k, borderType=cv2.BORDER_REPLICATE)
check(np.abs(convolve_separable(x, k) - ref).max() < 1e-5, "blend: convolution == cv2.sepFilter2D (replicate)")

frame = np.full((300, 400, 3), 0.5, np.float32)
garment = np.zeros_like(frame)
garment[..., 1] = 1
alpha = np.zeros((300, 400), np.float32)
alpha[100:200, 100:300] = 1
out, a = feather_blend(frame, garment, alpha)
check(np.allclose(out[150, 200], [0, 1, 0], atol=1e-3) and np.allclose(out[10, 10], 0.5),
      "blend: opaque inside, untouched outside")
profile = a[150, 90:115]
check(np.all(np.diff(profile) >= -1e-6) and len(np.unique(np.round(profile, 2))) > 4 and (a <= alpha + 1e-6).all(),
      "blend: soft, monotonic, non-bleeding edge")

# ---- garment anchors + full pipeline on a synthetic t-shirt --------------
shirt = np.zeros((400, 400), np.uint8)
cv2.fillPoly(shirt, [np.array([[120, 20], [170, 40], [230, 40], [280, 20], [390, 120], [340, 170],
                               [300, 130], [300, 390], [100, 390], [100, 130], [60, 170], [10, 120]])], 1)
anchors = auto_anchors(shirt > 0)
check(anchors["rightShoulder"][0] < anchors["leftShoulder"][0] and anchors["leftHip"][1] > 0.85,
      "garment: automatic anchors in plausible places")

g = Garment("synthetic", "synthetic", np.dstack([shirt * 0.8] * 3).astype(np.float32),
            shirt.astype(np.float32), anchors)
kp = {"leftShoulder": (400, 150, 1), "rightShoulder": (240, 150, 1), "leftHip": (380, 380, 1),
      "rightHip": (260, 380, 1), "leftElbow": (440, 260, 1), "rightElbow": (200, 260, 1)}
person = np.full((480, 640, 3), 90, np.uint8)
out, info = try_on(person, PoseResult(kp, None), g, TryOnOptions())
check(info["ok"] and out.shape == person.shape and out[260, 320].mean() > 120, "pipeline: garment drawn on torso")
out, info = try_on(person, PoseResult({}, None), g, TryOnOptions())
check(not info["ok"] and np.array_equal(out, person), "pipeline: no person -> frame unchanged")

# ---- thin-plate spline (from scratch) -----------------------------------
ctrl = rng.uniform(0, 400, (12, 2))
moved = ctrl + rng.normal(0, 8, (12, 2))
tps = fit_tps(ctrl, moved, lam=0)
check(np.abs(apply_tps(tps, ctrl) - moved).max() < 1e-6, "tps: interpolates the control points exactly (lambda = 0)")
ident = fit_tps(ctrl, ctrl, lam=0.5)
grid = rng.uniform(0, 400, (500, 2))
check(np.abs(apply_tps(ident, grid) - grid).max() < 1e-6, "tps: identity when no point moves")
aff = np.array([[1.1, 0.2], [-0.1, 0.9]])
taff = fit_tps(ctrl, ctrl @ aff.T + [5, -3], lam=0)
check(np.abs(apply_tps(taff, grid) - (grid @ aff.T + [5, -3])).max() < 1e-6, "tps: reproduces an affine map exactly")

img = cv2.GaussianBlur(rng.random((120, 160, 3)).astype(np.float32), (0, 0), 2)
mx, my = np.meshgrid(np.arange(160, dtype=np.float32) * 0.7 + 10.3, np.arange(120, dtype=np.float32) * 0.8 + 5.6)
own = remap_bilinear(img, mx, my)
ref = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
ok = (mx <= 158) & (my <= 118)
check(np.abs(own - ref)[ok].max() < 1e-3, "tps: our bilinear remap == cv2.remap")

# ---- occlusion + studio mask with a synthetic parsing ----------------------
h, w = 480, 640
yy, xx = np.mgrid[0:h, 0:w]
parsing = {k: np.zeros((h, w), bool) for k in ("background", "upper", "arms", "face", "hair", "lower")}
parsing["upper"][140:380, 250:390] = True
parsing["face"][40:160, 280:360] = True                       # SegFormer-style: face label incl. neck
parsing["arms"][260:420, 200:250] = True                      # right forearm hanging down
parsing["background"] = ~(parsing["upper"] | parsing["face"] | parsing["arms"])
kp2 = dict(kp, leftEye=(335, 70, 1), rightEye=(305, 70, 1), mouthLeft=(330, 105, 1), mouthRight=(310, 105, 1),
           rightElbow=(225, 250, 1), rightWrist=(225, 410, 1))
out_p, info_p = try_on(person, PoseResult(kp2, None), g, TryOnOptions(debug=True), parsing=parsing)
a = info_p["_alpha"]
check(info_p["ok"] and a[300:400, 205:245].max() < 0.05 and a[60:130, 290:350].max() < 0.05,
      "occlusion: forearm and face stay in front of the garment")
check(a[parsing["background"]].max() < 0.5, "occlusion: no garment on the background")

from hybrid.pipeline import tryon_mask  # noqa: E402
m = tryon_mask(person, PoseResult(kp2, None), parsing, a, long_sleeves=False)
check(m[150, 320] and m[300, 320] and not m[80, 320],
      "studio mask: neck below the chin and the torso are repainted, the face is not")
check(m[parsing["upper"]].all(), "studio mask: covers the whole old top (no old fabric left as context)")

# ---- garment coverage: a cropped garment over a long old top ---------------
from hybrid.pipeline import coverage_masks, exposed_anchor  # noqa: E402
parsing_c = {k: v.copy() for k, v in parsing.items()}
parsing_c["lower"][380:, 250:390] = True                      # trousers below the old hem
person_c = person.copy()
person_c[380:, 250:390] = (30, 40, 120)
crop_t = np.zeros((h, w), np.float32)
crop_t[140:260, 250:390] = 1                                  # new garment ends at y = 260
t, m, e = coverage_masks(person_c, PoseResult(kp2, None), parsing_c, crop_t, long_sleeves=False)
check(e[320, 320] and not m[320, 320], "coverage: old top below a cropped hem is exposed, not garment region")
check(m[200, 320] and not e[200, 320], "coverage: the garment's own target is garment region")
anchor, until = exposed_anchor(person_c, e, t, parsing_c, kp2)
check(until[370, 320] > 0 and abs(anchor[370, 320].astype(int) - (30, 40, 120)).max() < 5,
      "coverage: exposed region below the waistband is pinned to the mirrored trousers")
check(until[300, 320] == 0, "coverage: between the new hem and the waistband the model is left free (skin)")
sliver = np.zeros((h, w), np.float32)
sliver[140:386, 250:390] = 1                                  # target overlaps the trousers by 6 px
_, m2, _ = coverage_masks(person_c, PoseResult(kp2, None), parsing_c, sliver, long_sleeves=False)
check(not m2[395:, 300:340].any(), "coverage: a sliver of target over the trousers does not repaint them")

print(f"\n{'All checks passed' if not failures else f'{failures} check(s) FAILED'}")
sys.exit(1 if failures else 0)
