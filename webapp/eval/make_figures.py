"""Side-by-side figures from saved results (identical inputs per row).

    .venv/bin/python -m eval.make_figures
"""

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = RES / "figures"
TILE = (300, 400)


def tile(path, label):
    img = cv2.imread(str(path)) if path and Path(path).exists() else None
    if img is None:
        img = np.full((TILE[1], TILE[0], 3), 235, np.uint8)
        cv2.putText(img, "not run", (95, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    else:
        img = cv2.resize(img, TILE, interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (0, 0), (TILE[0], 28), (20, 20, 20), -1)
    cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def garment_tile(gid):
    return tile(ROOT.parent / "data" / "garments" / f"{gid}_source.jpg", "garment")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    pairs = json.loads((ROOT / "pairs.json").read_text())
    methods = [("classical", "classical (now)"), ("classical_plus", "classical+ (TPS/occl.)"),
               ("hybrid", "hybrid (ours+CatVTON)"), ("hybrid_warm70", "hybrid warm-start 0.7"),
               ("idm", "IDM-VTON (hosted)")]
    rows = []
    for pid, gid in pairs["ai_subset"]:
        row = [tile(ROOT / "persons" / f"{pid}.jpg", "person"), garment_tile(gid)]
        row += [tile(RES / m / f"{pid}__{gid}.jpg", lab) for m, lab in methods]
        rows.append(np.hstack(row))
    cv2.imwrite(str(FIG / "comparison_subset.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])

    # before/after per person: identical inputs in every row
    cols = [("classical", "BEFORE classical"), ("instant", "AFTER instant"),
            ("hybrid", "BEFORE hybrid PoC"), ("studio_standard", "AFTER studio std"),
            ("studio_detail", "AFTER studio detail")]
    for pid in pairs["persons"]:
        rows = []
        for gid in pairs["garments"]:
            row = [tile(ROOT / "persons" / f"{pid}.jpg", "person"), garment_tile(gid)]
            row += [tile(RES / m / f"{pid}__{gid}.jpg", lab) for m, lab in cols]
            rows.append(np.hstack(row))
        cv2.imwrite(str(FIG / f"before_after_{pid}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])

    for m in ("classical", "classical_plus", "hybrid", "instant", "studio_standard"):
        grid = []
        for pid in pairs["persons"]:
            grid.append(np.hstack([tile(ROOT / "persons" / f"{pid}.jpg", pid)] +
                                  [tile(RES / m / f"{pid}__{g}.jpg", g) for g in pairs["garments"]]))
        cv2.imwrite(str(FIG / f"grid_{m}.jpg"), np.vstack(grid), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("figures in", FIG)


if __name__ == "__main__":
    main()
