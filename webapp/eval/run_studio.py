"""Benchmark the integrated studio render (local hybrid) on the fixed pairs.

    .venv/bin/python -m eval.run_studio [standard|detail] [--only gid,gid]
        [--persons pid,pid] [--garment-dir eval/garments_short --garments gid,gid]
        [--coverage mask|garment] [--tag _x] [--resume]

Saves results/studio_<quality>/ (with colour correction) and
results/studio_<quality>_nofix/ (same generation, no colour correction).
"""

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from eval.metrics import output_metrics, parse
from eval.run_classical import rgba_of
from hybrid.catvton import CatVTON
from hybrid.pipeline import studio_render
from tryon_core.garment import load_garment
from tryon_core.parsing import default_cloth_parser
from tryon_core.pose import PoseDetector

ROOT = Path(__file__).resolve().parent
GARMENTS = ROOT.parent / "data" / "garments"
QUALITY = {"standard": ("vitonhd", "sliced"), "detail": ("mix", "sliced")}


def arg(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def main(quality, only=None):
    pairs = json.loads((ROOT / "pairs.json").read_text())
    gdir = Path(arg("--garment-dir")) if arg("--garment-dir") else GARMENTS
    garments = arg("--garments").split(",") if arg("--garments") else pairs["garments"]
    coverage = arg("--coverage") or "garment"
    ckpt, attn = QUALITY[quality]
    model, det, parser = CatVTON(checkpoint=ckpt, attention=attn), PoseDetector(video=False), default_cloth_parser()
    tag = arg("--tag") or ""
    dirs = {k: ROOT / "results" / f"studio_{quality}{tag}{k}" for k in ("", "_nofix")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    rows = []
    persons = arg("--persons").split(",") if arg("--persons") else pairs["persons"]
    for pid in persons:
        person = cv2.cvtColor(cv2.imread(str(ROOT / "persons" / f"{pid}.jpg")), cv2.COLOR_BGR2RGB)
        pose, parsing = det.detect(person), parse(person)
        for gid in garments:
            if only and gid not in only:
                continue
            if (dirs[""] / f"{pid}__{gid}.jpg").exists() and "--resume" in sys.argv:
                continue                     # already rendered (score with eval.score_outputs)
            g = load_garment(gdir, gid)
            src = cv2.cvtColor(cv2.imread(str(gdir / f"{gid}_source.jpg")), cv2.COLOR_BGR2RGB)
            out, info = studio_render(person, pose, parsing, g, src, model, steps=25, parser=parser, coverage=coverage)
            np.savez_compressed(dirs[""] / f"{pid}__{gid}_masks.npz",
                                **{k: info[k] for k in ("mask", "target", "exposed") if k in info})
            for k, img in (("", out), ("_nofix", info["uncorrected"])):
                cv2.imwrite(str(dirs[k] / f"{pid}__{gid}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            m = {"person": pid, "garment": gid, "seconds": info["seconds"]}
            m.update(output_metrics(person, out, rgba_of(g), parsing))
            m["garment_hist_nofix"] = output_metrics(person, info["uncorrected"], rgba_of(g), parsing)["garment_hist"]
            rows.append(m)
            print(f"{pid:10s} {gid:20s} {info['seconds']:6.1f}s face={m['ssim_face']:.3f} hist={m['garment_hist']:.3f} "
                  f"(nofix {m['garment_hist_nofix']:.3f}) old={m['old_unchanged']:.3f}", flush=True)
    if not rows:
        return
    with open(dirs[""] / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keys = ["ssim_keep", "ssim_face", "garment_hist", "garment_hist_nofix", "old_unchanged", "seconds"]
    print(f"studio_{quality} MEAN " + "  ".join(f"{k}={np.nanmean([r[k] for r in rows]):.3f}" for k in keys))


if __name__ == "__main__":
    only = arg("--only").split(",") if arg("--only") else None
    main(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "standard", only)
