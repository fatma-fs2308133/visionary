"""Benchmark the hybrid (our geometry/mask + local CatVTON) on the fixed pairs.

    .venv/bin/python -m eval.run_hybrid [strength] [--subset]

strength 1.0 = CatVTON from noise inside our mask; < 1 = warm start from our
classical composite (SDEdit). Runs locally on Apple MPS.
"""

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from eval.metrics import coverage_metrics, output_metrics, parse
from eval.run_classical import rgba_of
from hybrid.catvton import CatVTON
from hybrid.pipeline import hybrid_try_on
from tryon_core.garment import load_garment
from tryon_core.pose import PoseDetector

ROOT = Path(__file__).resolve().parent
GARMENTS = ROOT.parent / "data" / "garments"


def main(strength=1.0, subset=False, steps=20):
    pairs = json.loads((ROOT / "pairs.json").read_text())
    todo = pairs["ai_subset"] if subset else [(p, g) for p in pairs["persons"] for g in pairs["garments"]]
    name = "hybrid" if strength >= 1 else f"hybrid_warm{int(strength * 100)}"
    out_dir = ROOT / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model, det = CatVTON(), PoseDetector(video=False)
    rows, cache = [], {}
    for pid, gid in todo:
        if pid not in cache:
            person = cv2.cvtColor(cv2.imread(str(ROOT / "persons" / f"{pid}.jpg")), cv2.COLOR_BGR2RGB)
            cache = {pid: (person, det.detect(person), parse(person))}
        person, pose, parsing = cache[pid]
        g = load_garment(GARMENTS, gid)
        src = cv2.imread(str(GARMENTS / f"{gid}_source.jpg"))
        out, mask, secs, _ = hybrid_try_on(person, pose, parsing, g, src, model, strength=strength, steps=steps)
        m = {"person": pid, "garment": gid, "seconds": round(secs, 1)}
        m.update({k: v for k, v in coverage_metrics(mask.astype(np.float32), parsing, pose.keypoints).items()
                  if k in ("spill_face",)})
        m.update(output_metrics(person, out, rgba_of(g), parsing))
        rows.append(m)
        cv2.imwrite(str(out_dir / f"{pid}__{gid}.jpg"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        print(f"{pid:10s} {gid:20s} {secs:5.1f}s ssim_face={m['ssim_face']:.3f} hist={m['garment_hist']:.3f} "
              f"old={m['old_unchanged']:.3f} seam={m['edge_seam']:.2f}", flush=True)
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keys = ["ssim_keep", "ssim_face", "garment_hist", "old_unchanged", "edge_seam", "seconds"]
    print(f"{name} MEAN " + "  ".join(f"{k}={np.nanmean([r[k] for r in rows]):.3f}" for k in keys))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(float(args[0]) if args else 1.0, subset="--subset" in sys.argv)
