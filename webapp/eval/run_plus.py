"""Benchmark the instant (classical) pipeline WITH parsing: TPS edge fit +
occlusion + no spill, i.e. what the app's instant render and live preview do.

    .venv/bin/python -m eval.run_plus [--no-tps]
"""

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from eval.metrics import coverage_metrics, output_metrics, parse
from eval.run_classical import diagnostic, rgba_of, torso_width_ratio
from tryon_core.pipeline import TryOnOptions, try_on
from tryon_core.garment import load_garment
from tryon_core.pose import PoseDetector

ROOT = Path(__file__).resolve().parent


def main(use_tps=True):
    name = "instant" if use_tps else "instant_notps"
    out_dir = ROOT / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = json.loads((ROOT / "pairs.json").read_text())
    det = PoseDetector(video=False)
    rows = []
    for pid in pairs["persons"]:
        person = cv2.cvtColor(cv2.imread(str(ROOT / "persons" / f"{pid}.jpg")), cv2.COLOR_BGR2RGB)
        pose = det.detect(person)
        parsing = parse(person)
        for gid in pairs["garments"]:
            g = load_garment(ROOT.parent / "data" / "garments", gid)
            t = time.perf_counter()
            out, info = try_on(person, pose, g, TryOnOptions(tps=use_tps, debug=True), parsing=parsing)
            alpha = info.get("_alpha", np.zeros(person.shape[:2], np.float32))
            m = {"person": pid, "garment": gid, "seconds": round(time.perf_counter() - t, 3)}
            m.update(coverage_metrics(alpha, parsing, pose.keypoints))
            m["width_ratio"] = torso_width_ratio(alpha, parsing, pose.keypoints)
            m.update(output_metrics(person, out, rgba_of(g), parsing))
            rows.append(m)
            cv2.imwrite(str(out_dir / f"{pid}__{gid}.jpg"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / f"{pid}__{gid}_diag.jpg"),
                        cv2.cvtColor(diagnostic(out, alpha, parsing, pose.keypoints), cv2.COLOR_RGB2BGR))
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keys = ["old_visible", "spill_bg", "spill_face", "forearm_covered", "iou_upper", "width_ratio",
            "ssim_keep", "ssim_face", "garment_hist", "old_unchanged", "edge_seam", "seconds"]
    print(f"{name} MEAN " + "  ".join(f"{k}={np.nanmean([r[k] for r in rows]):.3f}" for k in keys))


if __name__ == "__main__":
    main(use_tps="--no-tps" not in sys.argv)
