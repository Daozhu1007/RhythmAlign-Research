# Build labeled contact sheets from frames/*.jpg
# usage: python 01_montage.py out.png idx1 idx2 ...   (or "every:step")
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

R4 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TILE_DIR = os.path.join(R4, "frames")
TILE_W, TILE_H = 480, 270
COLS, ROWS = 5, 5
LABEL_H = 22


def tile(i):
    im = Image.open(os.path.join(TILE_DIR, f"f{i:04d}.jpg")).resize((TILE_W, TILE_H))
    d = ImageDraw.Draw(im)
    t = i / 60.0
    d.rectangle([0, 0, 74, LABEL_H - 2], fill=(0, 0, 0))
    d.text((4, 3), f"{t:5.2f}s", fill=(255, 255, 0))
    return im


def main():
    out = sys.argv[1]
    spec = sys.argv[2]
    if spec.startswith("every:"):
        step = int(spec.split(":")[1])
        idxs = list(range(0, 901, step))
    else:
        idxs = [int(x) for x in spec.split(",")]
    sheets = [idxs[i:i + COLS * ROWS] for i in range(0, len(idxs), COLS * ROWS)]
    base, ext = os.path.splitext(out)
    for si, chunk in enumerate(sheets):
        sheet = Image.new("RGB", (COLS * TILE_W, ROWS * (TILE_H + LABEL_H)), (20, 20, 20))
        for j, i in enumerate(chunk):
            r, c = divmod(j, COLS)
            sheet.paste(tile(i), (c * TILE_W, r * (TILE_H + LABEL_H)))
        p = f"{base}_{si:02d}{ext}" if len(sheets) > 1 else out
        sheet.save(p, quality=88)
        print("wrote", p, f"({len(chunk)} tiles)")


if __name__ == "__main__":
    main()
