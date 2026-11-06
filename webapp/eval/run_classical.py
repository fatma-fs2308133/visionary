"""Benchmark the current classical pipeline on the fixed pairs.

    .venv/bin/python -m eval.run_classical

Writes eval/results/classical/<person>__<garment>.jpg, a diagnostic overlay
per pair (red = original top still visible, blue = garment on background /
face / hair, yellow = garment over the forearms) and metrics.csv.
"""

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from eval.metrics import coverage_metrics, output_metrics, parse
from tryon_core.garment import load_garment
from tryon_core.pipeline import TryOnOptions, try_on
from tryon_core.pose import PoseDetector

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "classical"
GARMENTS = ROOT.parent / "data" / "garments"


def rgba_of(g):
    return np.dstack([(g.rgb * 255).astype(np.uint8), (g.alpha * 255).astype(np.uint8)])


def torso_width_ratio(alpha, parsing, kp):
    """Garment width / real top width on the chest row (between shoulder and hip)."""
    y = int((kp["leftShoulder"][1] + kp["rightShoulder"][1]) * 0.25 + (kp["leftHip"][1] + kp["rightHip"][1]) * 0.25)
    g = np.flatnonzero(alpha[y] > 0.5)
    t = np.flatnonzero(parsing["upper"][y] | parsing["arms"][y])
    if not len(g) or not len(t):
        return float("nan")
    return float((g[-1] - g[0]) / max(t[-1] - t[0], 1))


def main():
    pairs = json.loads((ROOT / "pairs.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    det = PoseDetector(video=False)
    rows = []
    for pid in pairs["persons"]:
        person = cv2.cvtColor(cv2.imread(str(ROOT / "persons" / f"{pid}.jpg")), cv2.COLOR_BGR2RGB)
        t = time.perf_counter()
        pose = det.detect(person)
        pose_s = time.perf_counter() - t
        parsing = parse(person)
        for gid in pairs["garments"]:
            g = load_garment(GARMENTS, gid)
            t = time.perf_counter()
            out, info = try_on(person, pose, g, TryOnOptions(debug=True))
            run_s = time.perf_counter() - t + pose_s
            alpha = info.get("_alpha", np.zeros(person.shape[:2], np.float32))
            m = {"person": pid, "garment": gid, "ok": info["ok"], "seconds": round(run_s, 3)}
            m.update(coverage_metrics(alpha, parsing, pose.keypoints))
            m["width_ratio"] = torso_width_ratio(alpha, parsing, pose.keypoints) if info["ok"] else float("nan")
            m.update(output_metrics(person, out, rgba_of(g), parsing))
            rows.append(m)
            cv2.imwrite(str(OUT / f"{pid}__{gid}.jpg"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(OUT / f"{pid}__{gid}_diag.jpg"), cv2.cvtColor(diagnostic(out, alpha, parsing, pose.keypoints), cv2.COLOR_RGB2BGR))
            print(f"{pid:10s} {gid:20s} old_visible={m['old_visible']:.2f} spill_bg={m['spill_bg']:.2f} "
                  f"spill_face={m['spill_face']:.2f} forearm={m['forearm_covered']:.2f} width={m['width_ratio']:.2f}")
    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keys = ["old_visible", "spill_bg", "spill_face", "forearm_covered", "iou_upper", "width_ratio",
            "ssim_keep", "ssim_face", "garment_hist", "old_unchanged", "edge_seam", "seconds"]
    print("\nMEAN " + "  ".join(f"{k}={np.nanmean([r[k] for r in rows]):.3f}" for k in keys))


def diagnostic(out, alpha, parsing, kp):
    from eval.metrics import forearms
    d = out.astype(np.float32)
    g = alpha > 0.5

    def tint(mask, color):
        d[mask] = 0.45 * d[mask] + 0.55 * np.array(color, np.float32)

    tint(parsing["upper"] & ~g, (230, 40, 40))                                 # old top visible
    tint(g & (parsing["background"] | parsing["face_hair"]), (40, 90, 230))    # spill
    tint(g & forearms(parsing, kp), (240, 200, 30))                            # forearms covered
    return d.astype(np.uint8)


if __name__ == "__main__":
    main()
