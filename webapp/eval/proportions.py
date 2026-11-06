"""Does the rendered garment take its proportions from the PRODUCT or from the
person's OLD top (i.e. from our mask)?

    .venv/bin/python -m eval.proportions studio_standard

Length = (lowest garment row - shoulder line) / shoulder width, measured on the
product photo (anchors), on the person's old top and on the rendered top (parsing).
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from eval.metrics import parse
from tryon_core.garment import load_garment
from tryon_core.pose import PoseDetector

ROOT = Path(__file__).resolve().parent


def product_length(g):
    a = g.anchors_px()
    sh_y = (a["leftShoulder"][1] + a["rightShoulder"][1]) / 2
    sw = abs(a["leftShoulder"][0] - a["rightShoulder"][0])
    rows = np.flatnonzero((g.alpha > 0.5).any(axis=1))
    return (rows[-1] - sh_y) / sw


def worn_length(top, kp):
    sh_y = (kp["leftShoulder"][1] + kp["rightShoulder"][1]) / 2
    sw = abs(kp["leftShoulder"][0] - kp["rightShoulder"][0])
    rows = np.flatnonzero(top.any(axis=1))
    return (rows[-1] - sh_y) / sw if len(rows) else np.nan


def main(method):
    pairs = json.loads((ROOT / "pairs.json").read_text())
    det = PoseDetector(video=False)
    P, O, R = [], [], []
    for pid in pairs["persons"]:
        person = cv2.cvtColor(cv2.imread(str(ROOT / "persons" / f"{pid}.jpg")), cv2.COLOR_BGR2RGB)
        kp = det.detect(person).keypoints
        old = worn_length(parse(person)["upper"], kp)
        for gid in pairs["garments"]:
            out = cv2.cvtColor(cv2.imread(str(ROOT / "results" / method / f"{pid}__{gid}.jpg")), cv2.COLOR_BGR2RGB)
            P.append(product_length(load_garment(ROOT.parent / "data" / "garments", gid)))
            O.append(old)
            R.append(worn_length(parse(out)["upper"], kp))
            print(f"{pid:10s} {gid:20s} product={P[-1]:.2f} old_top={O[-1]:.2f} rendered={R[-1]:.2f}")
    P, O, R = map(np.array, (P, O, R))
    print(f"\ncorr(rendered, old top) = {np.corrcoef(R, O)[0, 1]:+.2f}   corr(rendered, product) = {np.corrcoef(R, P)[0, 1]:+.2f}")
    # partial picture: linear fit rendered ~ a*old + b*product + c
    A = np.c_[O, P, np.ones_like(O)]
    coef, *_ = np.linalg.lstsq(A, R, rcond=None)
    print(f"fit: rendered = {coef[0]:.2f} x old_top + {coef[1]:.2f} x product + {coef[2]:.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "studio_standard")
