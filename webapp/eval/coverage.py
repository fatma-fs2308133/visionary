"""Does the render show the SELECTED garment's coverage, and only change what it should?

    .venv/bin/python -m eval.coverage METHOD [METHOD ...] [--garment-dir eval/garments_short]

For every <person>__<garment>.jpg of each method, measured against references
computed from the INPUTS only (the same for every method):

  target T   product silhouette placed by our proportional warp (coverage_masks)
  exposed E  old top on the torso outside the new garment's region

  len_err      (rendered hem - target hem) / shoulder width, centre columns.
               Rendered hem = lowest row the parser calls top/dress.  + = too long
  E_garment    share of E the parser still calls top/dress (old or new fabric)
  E_unchanged  share of E whose colour did not change (old top left as it was)
  E_garmentlike share of E whose colour is closer to the product's palette than
               to the person's skin and trousers (parser-free: a garment-
               coloured "skirt" the parser calls skirt still counts)
  trousers_repainted share of ALL visible trousers whose colour changed (dE > 25),
               including legitimate cover by a longer garment
  trousers     SSIM on the trousers outside the new garment's region (1 = untouched)
  trousers_chg share of those trouser pixels whose colour changed (dE > 20)
  skin         SSIM on arm skin outside the repaint region
  colour       product vs rendered garment colour histogram (in T)
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from eval.metrics import _hist_similarity, _masked_ssim, parse
from eval.run_classical import rgba_of
from hybrid.pipeline import coverage_masks, is_long_sleeved
from tryon_core.garment import load_garment
from tryon_core.parsing import dilate
from tryon_core.pipeline import TryOnOptions, try_on
from tryon_core.pose import PoseDetector

ROOT = Path(__file__).resolve().parent
KEYS = ["len_err", "abs_len_err", "E_px", "E_garment", "E_garmentlike", "E_unchanged", "trousers_repainted",
        "trousers", "trousers_chg", "skin", "colour"]


def palette(lab_px, k):
    if len(lab_px) < k * 20:
        return np.zeros((0, 3), np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centres = cv2.kmeans(lab_px.astype(np.float32), k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return centres


def nearest(lab_px, centres):
    if not len(centres):
        return np.full(len(lab_px), np.inf)
    return np.linalg.norm(lab_px[:, None, :] - centres[None], axis=2).min(axis=1)


def references(person, pose, parsing, g):
    _, info = try_on(person, pose, g, TryOnOptions(debug=True), parsing=parsing)
    return coverage_masks(person, pose, parsing, info["_alpha"], is_long_sleeved(g))


def score(person, out, kp, parsing, g, refs):
    t, m, e = refs
    sw = abs(kp["leftShoulder"][0] - kp["rightShoulder"][0])
    xc = (kp["leftHip"][0] + kp["rightHip"][0]) / 2
    band = slice(int(xc - 0.25 * sw), int(xc + 0.25 * sw))
    po = parse(out)
    top = (po["upper"] | (po["labels"] == 7))[:, band]          # 7 = dress
    rows_t, rows_o = np.flatnonzero(t[:, band].any(1)), np.flatnonzero(top.any(1))
    len_err = (rows_o[-1] - rows_t[-1]) / sw if len(rows_t) and len(rows_o) else np.nan
    lab_in = cv2.cvtColor(person, cv2.COLOR_RGB2LAB).astype(int)
    d = np.abs(lab_in - cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(int)).sum(2)
    grow = max(3, int(0.04 * sw))
    trousers = parsing["lower"] & ~dilate(t | m, 3 * grow)
    trousers = cv2.erode(trousers.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    skin = parsing["arms"] & ~dilate(m | e, 2 * grow)
    n_e = max(e.sum(), 1)
    lab_out = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    rgba = rgba_of(g)
    g_lab = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2LAB).astype(np.float32)[rgba[:, :, 3] > 127]
    lab_in_f = lab_in.astype(np.float32)
    pal_g = palette(g_lab[:: max(1, len(g_lab) // 4000)], 4)
    pal_o = np.concatenate([palette(lab_in_f[parsing["arms"] | parsing["face"]][::7], 2),
                            palette(lab_in_f[parsing["lower"]][::7], 3)])
    garmentlike = np.nan
    if e.any():
        px = lab_out[e]
        dg, do = nearest(px, pal_g), nearest(px, pal_o)
        garmentlike = float(((dg < do) & (dg < 25)).mean())
    lower_in = parsing["lower"]
    return {"len_err": len_err, "abs_len_err": abs(len_err),
            "E_px": e.sum() / (sw * sw),
            "E_garment": float((e & (po["upper"] | (po["labels"] == 7))).sum() / n_e) if e.any() else np.nan,
            "E_garmentlike": garmentlike,
            "E_unchanged": float((e & (d < 20)).sum() / n_e) if e.any() else np.nan,
            "trousers_repainted": float((lower_in & (d > 25)).sum() / max(lower_in.sum(), 1)),
            "trousers": _masked_ssim(person, out, trousers) if trousers.sum() > 200 else np.nan,
            "trousers_chg": float((trousers & (d > 20)).sum() / max(trousers.sum(), 1)) if trousers.sum() > 200 else np.nan,
            "skin": _masked_ssim(person, out, skin) if skin.sum() > 200 else np.nan,
            "colour": _hist_similarity(rgba_of(g), out, po["upper"] & dilate(t, grow))}


def main(methods, gdir):
    det, cache = PoseDetector(video=False), {}
    for method in methods:
        rows = []
        for f in sorted((ROOT / "results" / method).glob("*__*.jpg")):
            pid, gid = f.stem.split("__")
            if not (gdir / f"{gid}.json").exists():
                continue
            person = cv2.cvtColor(cv2.imread(str(ROOT / "persons" / f"{pid}.jpg")), cv2.COLOR_BGR2RGB)
            if pid not in cache:
                cache[pid] = (det.detect(person), parse(person))
            pose, parsing = cache[pid]
            g = load_garment(gdir, gid)
            if (pid, gid) not in cache:
                cache[(pid, gid)] = references(person, pose, parsing, g)
            out = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            r = score(person, out, pose.keypoints, parsing, g, cache[(pid, gid)])
            r.update(person=pid, garment=gid)
            rows.append(r)
            if "-v" in sys.argv:
                print(f"  {pid:10s} {gid:22s} " + " ".join(f"{k}={r[k]:.2f}" for k in KEYS))
        if rows:
            print(f"{method:30s} n={len(rows):2d} " + " ".join(f"{k}={np.nanmean([r[k] for r in rows]):.3f}" for k in KEYS))
            (ROOT / "results" / method / "coverage.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    gd = Path(sys.argv[sys.argv.index("--garment-dir") + 1]) if "--garment-dir" in sys.argv else ROOT.parent / "data" / "garments"
    ms = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("-") and sys.argv[i - 1] != "--garment-dir"]
    main(ms, gd)
