"""Run pretrained diffusion try-on models (hosted HF Spaces) on the fixed AI subset.

    .venv/bin/python -m eval.run_hosted [method ...]      methods: idm leffa fashn fitdit

Same person photo and same garment product photo for every method.
Results: eval/results/<method>/<person>__<garment>.jpg and hosted.csv (time,
errors). Set HF_TOKEN for more ZeroGPU quota.
"""

import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT.parent / "data" / "garments"


def _client(space):
    from gradio_client import Client
    return Client(space, token=os.environ.get("HF_TOKEN") or None, verbose=False)


def _path(x):
    """gradio returns a filepath, a dict with 'path', or a gallery list."""
    if isinstance(x, (list, tuple)):
        x = x[0]
    if isinstance(x, dict):
        x = x.get("path") or x.get("image", {}).get("path")
    return x


def run_idm(person, garment, desc):
    from gradio_client import handle_file
    out, _ = _client("yisol/IDM-VTON").predict(
        {"background": handle_file(person), "layers": [], "composite": None},
        handle_file(garment), desc, True, True, 30, 42, api_name="/tryon")
    return _path(out)


def run_leffa(person, garment, desc):
    from gradio_client import handle_file
    out = _client("franciszzj/Leffa").predict(
        handle_file(person), handle_file(garment), False, 30, 2.5, 42,
        "viton_hd", "upper_body", False, api_name="/leffa_predict_vt")
    return _path(out[0])


def run_fashn(person, garment, desc):
    from gradio_client import handle_file
    out = _client("fashn-ai/fashn-vton-1.5").predict(
        handle_file(person), handle_file(garment), "tops", "flat-lay", 30, 1.5, 42, True, api_name="/try_on")
    return _path(out)


def run_fitdit(person, garment, desc):
    from gradio_client import handle_file
    c = _client("BoyuanJiang/FitDiT")
    masked, pose = c.predict(handle_file(person), "Upper-body", 0, 0, 0, 0, api_name="/generate_mask")
    out = c.predict(handle_file(person), handle_file(garment), masked, handle_file(_path(pose)),
                    20, 2, 42, 1, "768x1024", api_name="/process")
    return _path(out)


METHODS = {"idm": run_idm, "leffa": run_leffa, "fashn": run_fashn, "fitdit": run_fitdit}


def main(methods):
    pairs = json.loads((ROOT / "pairs.json").read_text())["ai_subset"]
    names = json.loads((ROOT.parent / "data" / "catalog" / "catalog.json").read_text())
    log = ROOT / "results" / "hosted.csv"
    new = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["method", "person", "garment", "seconds", "status"])
        for m in methods:
            (ROOT / "results" / m).mkdir(parents=True, exist_ok=True)
            for pid, gid in pairs:
                dst = ROOT / "results" / m / f"{pid}__{gid}.jpg"
                if dst.exists():
                    continue
                person = str(ROOT / "persons" / f"{pid}.jpg")
                garment = str(CATALOG / f"{gid}_source.jpg")
                titles = [v if isinstance(v, str) else v["name"] for k, v in names.items() if not k.startswith("_")]
                desc = next((t for t in titles if t.lower().replace("-", "_").replace(" ", "_") == gid), gid)
                t = time.perf_counter()
                try:
                    res = METHODS[m](person, garment, desc)
                    img = cv2.imread(res)
                    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    status = "ok"
                except Exception as err:             # quota, queue, Space errors
                    status = f"error: {str(err)[:160]}"
                secs = round(time.perf_counter() - t, 1)
                print(f"{m:7s} {pid:10s} {gid:20s} {secs:6.1f}s  {status}", flush=True)
                w.writerow([m, pid, gid, secs, status])
                f.flush()
                if "quota" in status.lower() or "exceeded" in status.lower():
                    print(f"  {m}: GPU quota exhausted, skipping the rest")
                    break


if __name__ == "__main__":
    main(sys.argv[1:] or list(METHODS))
