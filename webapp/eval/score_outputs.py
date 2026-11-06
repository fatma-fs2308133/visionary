"""Score any method's saved outputs with the output-based metrics.

    .venv/bin/python -m eval.score_outputs idm [classical hybrid ...] [--subset]

--subset restricts every method to pairs.json "ai_subset" (fair comparison
with the hosted models, which only ran on that subset).
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from eval.metrics import output_metrics, parse
from eval.run_classical import rgba_of
from tryon_core.pose import PoseDetector
from tryon_core.garment import load_garment

ROOT = Path(__file__).resolve().parent
KEYS = ["ssim_keep", "ssim_face_only", "ssim_face", "garment_hist", "old_unchanged"]


def main(methods, subset):
    pairs = json.loads((ROOT / "pairs.json").read_text())
    allowed = {tuple(p) for p in pairs["ai_subset"]} if subset else None
    if "--garments" in sys.argv:                 # e.g. --garments gingham_shirt,velvet_wrap_top
        keep = sys.argv[sys.argv.index("--garments") + 1].split(",")
        allowed = {(p, g) for p in pairs["persons"] for g in keep}
    parsed, poses, det = {}, {}, PoseDetector(video=False)
    for m in methods:
        rows = []
        for f in sorted((ROOT / "results" / m).glob("*__*.jpg")):
            if f.stem.endswith("_diag"):
                continue
            pid, gid = f.stem.split("__")
            if allowed and (pid, gid) not in allowed:
                continue
            person = cv2.cvtColor(cv2.imread(str(ROOT / "persons" / f"{pid}.jpg")), cv2.COLOR_BGR2RGB)
            if pid not in parsed:
                parsed[pid] = parse(person)
                poses[pid] = det.detect(person).keypoints
            out = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            g = load_garment(ROOT.parent / "data" / "garments", gid)
            rows.append(output_metrics(person, out, rgba_of(g), parsed[pid], poses[pid]))
        if rows:
            print(f"{m:16s} n={len(rows):2d}  " + "  ".join(f"{k}={np.nanmean([r[k] for r in rows]):.3f}" for k in KEYS))


if __name__ == "__main__":
    args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] != "--garments"]
    main(args, "--subset" in sys.argv)
