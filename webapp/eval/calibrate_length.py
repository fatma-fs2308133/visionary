"""Ground truth for garment length: VITON-HD test models wearing the very
garment shown in the paired product photo (eval/vitonhd_gt/{image,cloth}).

    .venv/bin/python -m eval.calibrate_length

For each pair: the hem our proportional mapping predicts from the product
photo, and the hem actually worn (lowest row the parser calls top/dress in the
central torso columns, so hanging sleeves do not count). Both in shoulder
widths below the shoulder line. How a top is worn (tucked, over high-waisted
bottoms) moves the worn hem a lot, so medians are used.
"""

from pathlib import Path

import cv2
import numpy as np

from eval.metrics import parse
from tryon_core.garment import prepare_garment
from tryon_core.pipeline import CHEST_PER_SHOULDER, TORSO
from tryon_core.pose import PoseDetector

ROOT = Path(__file__).resolve().parent / "vitonhd_gt"


def predicted_hem(rgba, anchors, chest_per_shoulder=CHEST_PER_SHOULDER):
    """Hem below the shoulder line in body shoulder widths, as _proportional_torso places it."""
    h, w = rgba.shape[:2]
    a = {k: np.array([v[0] * (w - 1), v[1] * (h - 1)]) for k, v in anchors.items()}
    g_len = np.linalg.norm((a["leftHip"] + a["rightHip"]) / 2 - (a["leftShoulder"] + a["rightShoulder"]) / 2)
    chest = abs(a["leftArmpit"][0] - a["rightArmpit"][0])
    shoulders = np.linalg.norm(a["leftShoulder"] - a["rightShoulder"])
    if chest < 0.8 * shoulders:                       # same fallback as the pipeline
        chest = 1.09 * shoulders
    return g_len * chest_per_shoulder / chest


def measure():
    det, rows = PoseDetector(video=False), []
    for f in sorted((ROOT / "image").glob("*.jpg")):
        img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        pose = det.detect(img)
        kp = pose.keypoints
        if not kp or any(kp[k][2] < 0.5 for k in TORSO):
            continue
        sw = np.linalg.norm(np.subtract(kp["leftShoulder"][:2], kp["rightShoulder"][:2])) / max(np.cos(pose.torso_yaw()), 0.5)
        sh_y = (kp["leftShoulder"][1] + kp["rightShoulder"][1]) / 2
        hip_y = (kp["leftHip"][1] + kp["rightHip"][1]) / 2
        xc = (kp["leftHip"][0] + kp["rightHip"][0]) / 2
        p = parse(img)
        band = slice(int(xc - 0.25 * sw), int(xc + 0.25 * sw))
        rows_t = np.flatnonzero(p["upper"][:, band].any(axis=1))
        rows_l = np.flatnonzero(p["lower"][:, band].any(axis=1))
        if not len(rows_t):
            continue
        try:
            rgba, anchors = prepare_garment(cv2.imread(str(ROOT / "cloth" / f.name), cv2.IMREAD_UNCHANGED))
        except ValueError:
            continue
        rows.append({"id": f.stem, "worn": (rows_t[-1] - sh_y) / sw, "hip": (hip_y - sh_y) / sw,
                     "trousers_top": (rows_l[0] - sh_y) / sw if len(rows_l) else np.nan,
                     "pred": predicted_hem(rgba, anchors)})
    return rows


def main():
    rows = measure()
    # a failed cut-out (white tee on white: rembg keeps only the print, 00654)
    # gives a sliver of a "garment" and an absurd length
    bad = [r["id"] for r in rows if r["pred"] > 3.5]
    rows = [r for r in rows if r["pred"] <= 3.5]
    print("excluded (failed cut-out):", bad)
    for r in rows:
        print(f"{r['id']}  worn={r['worn']:.2f}  predicted={r['pred']:.2f}  hip={r['hip']:.2f}  trousers_top={r['trousers_top']:.2f}")
    worn, pred = np.array([r["worn"] for r in rows]), np.array([r["pred"] for r in rows])
    print(f"\nn={len(rows)}  corr(worn, predicted) = {np.corrcoef(worn, pred)[0, 1]:+.2f}  "
          f"median worn/predicted = {np.median(worn / pred):.2f}")
    for k in (1.0, 0.9, 0.85, 0.8, 0.75):
        err = worn - k * pred
        print(f"  scale x{k:.2f}: median |error| {np.median(np.abs(err)):.2f}, mean {err.mean():+.2f} shoulder widths")
    print(f"worn hem - hip joints: median {np.median(worn - [r['hip'] for r in rows]):+.2f}; "
          f"worn hem - trousers top: median {np.nanmedian(worn - [r['trousers_top'] for r in rows]):+.2f}")


if __name__ == "__main__":
    main()
