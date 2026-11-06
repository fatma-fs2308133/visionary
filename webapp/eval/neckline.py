"""Neckline leak: how much of the ORIGINAL top's collar zone survives unchanged.

    .venv/bin/python -m eval.neckline hybrid studio_standard

Collar zone = old top pixels above the shoulder line + 0.25 x shoulder width.
A pixel "survives" if its Lab colour changed by less than 20.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from eval.metrics import parse
from tryon_core.pose import PoseDetector

ROOT = Path(__file__).resolve().parent


def main(methods):
    pairs = json.loads((ROOT / "pairs.json").read_text())
    det = PoseDetector(video=False)
    zones, lab_in = {}, {}
    for pid in pairs["persons"]:
        person = cv2.cvtColor(cv2.imread(str(ROOT / "persons" / f"{pid}.jpg")), cv2.COLOR_BGR2RGB)
        kp = det.detect(person).keypoints
        ls, rs = np.array(kp["leftShoulder"][:2]), np.array(kp["rightShoulder"][:2])
        y = int((ls[1] + rs[1]) / 2 + 0.25 * np.linalg.norm(ls - rs))
        z = parse(person)["upper"].copy()
        z[y:] = False
        zones[pid], lab_in[pid] = z, cv2.cvtColor(person, cv2.COLOR_RGB2LAB).astype(int)
    for m in methods:
        vals, bad = [], []
        for pid in pairs["persons"]:
            for gid in pairs["garments"]:
                out = cv2.imread(str(ROOT / "results" / m / f"{pid}__{gid}.jpg"))
                if out is None:
                    continue
                lab = cv2.cvtColor(cv2.resize(out, zones[pid].shape[::-1]), cv2.COLOR_BGR2LAB).astype(int)
                same = np.abs(lab - lab_in[pid]).sum(2) < 20
                v = float((same & zones[pid]).sum() / max(zones[pid].sum(), 1))
                vals.append(v)
                if v > 0.10:
                    bad.append(f"{pid}/{gid}={v:.2f}")
        print(f"{m:18s} n={len(vals):2d} collar-zone survives: mean={np.mean(vals):.3f}  "
              f"pairs >10%: {len(bad)}  {' '.join(bad)}")


if __name__ == "__main__":
    main(sys.argv[1:])
