# R4 - merge audio band onsets into candidate event list; build 60fps visual
# verification strips (3 frames each) packed into sheets for manual labeling.
import json
import os

import numpy as np
from PIL import Image, ImageDraw

R4 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(R4, "work")
TILE_DIR = os.path.join(R4, "frames")
FPS = 60.04

def merge(onsets, gap=0.055):
    xs = sorted(onsets)
    out = []
    last = -9
    for x in xs:
        if x - last > gap:
            out.append(x)
        last = x
    return out

def main():
    A = json.load(open(os.path.join(WORK, "audio_onsets.json")))
    hi = A["2000_6000"] + A["6000_14000"]
    mid = A["500_2000"]
    cand = merge(hi + mid, 0.055)
    print("candidates:", len(cand))
    # strip sheets: 3 frames (t-33ms, t, t+33ms) at 384x216, 3 strips per sheet row-pair...
    TW, TH = 384, 216
    COLS = 3
    per_sheet = 4  # 4 strips per sheet (12 tiles)
    for si in range(0, len(cand), per_sheet):
        chunk = cand[si:si + per_sheet]
        sheet = Image.new("RGB", (COLS * TW, per_sheet * (TH + 22)), (8, 8, 8))
        for j, t in enumerate(chunk):
            i0 = int(round(t * FPS))
            for k, off in enumerate([-1, 0, 1]):
                fi = min(max(i0 + off, 0), 900)
                im = Image.open(os.path.join(TILE_DIR, f"f{fi:04d}.jpg")).resize((TW, TH))
                d = ImageDraw.Draw(im)
                d.rectangle([0, 0, 96, 20], fill=(0, 0, 0))
                d.text((3, 2), f"#{si+j} {t-0.033*(1-off):.3f}{'*' if off==0 else ' '}",
                       fill=(0, 255, 255) if off == 0 else (200, 200, 200))
                sheet.paste(im, (k * TW, j * (TH + 22)))
        p = os.path.join(WORK, f"strip_{si:03d}.jpg")
        sheet.save(p, quality=90)
    json.dump([float(x) for x in cand], open(os.path.join(WORK, "cand_events.json"), "w"), indent=1)
    print("strips:", (len(cand) + per_sheet - 1) // per_sheet)

if __name__ == "__main__":
    main()
