"""Before/after sheet for the garment-coverage change (identical inputs).

    .venv/bin/python -m eval.coverage_figure OUT.jpg BEFORE AFTER GARMENT_DIR pid__gid [pid__gid ...]

Each row: person | product photo | BEFORE | AFTER (| AFTER's masks:
green = garment target, blue = CatVTON region, red = exposed region pinned
to trousers / left free for skin).
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
H = 360


def fit(img, h=H):
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h), interpolation=cv2.INTER_AREA)


def label(img, text):
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main(out, before, after, gdir, pairs):
    rows = []
    for pg in pairs:
        pid, gid = pg.split("__")
        person = cv2.imread(str(ROOT / "persons" / f"{pid}.jpg"))
        cells = [label(fit(person), pid), label(fit(cv2.imread(str(Path(gdir) / f"{gid}_source.jpg"))), gid)]
        for name, d in (("before", before), ("after", after)):
            cells.append(label(fit(cv2.imread(str(ROOT / "results" / d / f"{pg}.jpg"))), name))
        z = np.load(ROOT / "results" / after / f"{pg}_masks.npz")
        ov = person.astype(np.float32)
        for key, colour, a in (("mask", (255, 120, 0), 0.4), ("target", (0, 200, 0), 0.4), ("exposed", (0, 0, 255), 0.6)):
            if key in z:
                ov[z[key]] = ov[z[key]] * (1 - a) + np.array(colour) * a
        cells.append(label(fit(ov.astype(np.uint8)), "masks (after)"))
        rows.append(np.hstack(cells))
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
