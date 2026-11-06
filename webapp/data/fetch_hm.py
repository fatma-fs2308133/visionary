"""Download the H&M product photos used by the garment library.

    .venv/bin/python data/fetch_hm.py

They are not in git: H&M released them for the Kaggle competition "H&M
Personalized Fashion Recommendations" (2022) for non-commercial / academic
research, which does not allow republishing them in a public repository.
This fetches the same images from a public Hugging Face mirror
(tomytjandra/h-and-m-fashion-caption, row index = file name). Restart the
server afterwards and it prepares them (cut-out + anchors) on startup.
"""

import json
from pathlib import Path

import httpx

CATALOG = Path(__file__).resolve().parent / "catalog"
ROWS = "https://datasets-server.huggingface.co/rows"


def main():
    names = json.loads((CATALOG / "catalog.json").read_text())
    for file in names:
        if not file.startswith("hm_") or (CATALOG / file).exists():
            continue
        row = int(file[3:8])
        r = httpx.get(ROWS, params={"dataset": "tomytjandra/h-and-m-fashion-caption", "config": "default",
                                    "split": "train", "offset": row, "length": 1}, timeout=120)
        src = r.json()["rows"][0]["row"]["image"]["src"]
        (CATALOG / file).write_bytes(httpx.get(src, timeout=120).content)
        print("downloaded", file, names[file]["name"])


if __name__ == "__main__":
    main()
