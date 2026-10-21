"""Extract the raster images of Figs 4-7 from the Qu & Mudawar PDF at native resolution.

Run:  .venv/Scripts/python digitise/extract_figs.py
"""
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data/paper/QuMudawar2002_IJHMT.pdf"
OUT = ROOT / "data/paper/figs"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(PDF)
    for page_no in (7, 8, 9):  # 1-based: Fig 4 on p7, Fig 5 on p8, Figs 6-7 on p9
        page = doc[page_no - 1]
        for j, img in enumerate(page.get_images(full=True)):
            data = doc.extract_image(img[0])
            rect = page.get_image_rects(img[0])
            name = OUT / f"p{page_no:02d}_{j}.{data['ext']}"
            name.write_bytes(data["image"])
            print(name.name, data["width"], data["height"], rect)


if __name__ == "__main__":
    main()
