# R5.5 Phase 7 - BLIND Holdout-C oracle apparatus (R5 60_holdout_oracle.py
# adapted; no detector predictions used anywhere in this flow).
#   pass 1: 10 fps contact sheets (whole window)
#   pass 2: 30 fps contact sheets
#   pass 3: audio-onset candidates (music-removed comb envelope, R4 family)
#           -> 3-frame 60 fps zoom strips for visual verification
# The oracle JSON (63_write_oracle_c.py) must be completed from visual review
# BEFORE the frozen R5.5 detector is ever run on this window (Phase 8).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw
from scipy import signal as sig

from common55 import (R55_OUT, R55_WORK, load_json, save_json)
from r5_common import SR, comb_envelope, iter_frames

FPS = 60.04
DUR = 18.0
TILE_DIR = os.path.join(R55_WORK, "hc_frames")


def extract_tiles():
    os.makedirs(TILE_DIR, exist_ok=True)
    n = 0
    for i, fr in iter_frames(os.path.join(R55_OUT, "holdoutC_video.mp4"), w=480):
        Image.fromarray(fr.astype(np.uint8)).save(
            os.path.join(TILE_DIR, f"f{i:04d}.jpg"), quality=88)
        n = i + 1
    print("tiles:", n)
    return n


def comb_candidates():
    comb, _ = comb_envelope(os.path.join(R55_WORK, "holdoutC_ctx.wav"),
                            os.path.join(R55_WORK, "holdoutC_ref_warp.wav"), 3.0, DUR)
    np.save(os.path.join(R55_WORK, "holdoutC_comb.npy"), comb)
    med = np.median(comb)
    mad = np.median(np.abs(comb - med)) + 1e-9
    pk, _ = sig.find_peaks(comb, height=med + 4.0 * mad, distance=55)
    on = sorted(set(round(float(k / SR), 3) for k in pk))
    merged, last = [], -9
    for x in on:
        if x - last > 0.055:
            merged.append(x)
        last = x
    print("audio candidates:", len(merged))
    return merged


def sheets():
    tiles = sorted(f for f in os.listdir(TILE_DIR) if f.endswith(".jpg"))
    n = len(tiles)
    p1 = list(range(0, n, 6))   # 10 fps
    p2 = list(range(0, n, 2))   # 30 fps
    for name, idxs in (("p1_10fps", p1), ("p2_30fps", p2)):
        per = 30
        TW, TH = 384, 216
        COLS = 5
        for si in range(0, len(idxs), per):
            chunk = idxs[si:si + per]
            rows = (len(chunk) + COLS - 1) // COLS
            sh = Image.new("RGB", (COLS * TW, rows * (TH + 18)), (8, 8, 8))
            for j, fi in enumerate(chunk):
                im = Image.open(os.path.join(TILE_DIR, f"f{fi:04d}.jpg")).resize((TW, TH))
                d = ImageDraw.Draw(im)
                d.rectangle([0, 0, 110, 16], fill=(0, 0, 0))
                d.text((3, 2), f"{fi/FPS:.2f}s", fill=(0, 255, 255))
                sh.paste(im, ((j % COLS) * TW, (j // COLS) * (TH + 18)))
            sh.save(os.path.join(R55_WORK, f"hc_{name}_{si//per:02d}.jpg"), quality=85)
        print(name, "sheets:", (len(idxs) + per - 1) // per)


def strips(cands):
    per = 4
    TW, TH = 384, 216
    COLS = 3
    for si in range(0, len(cands), per):
        chunk = cands[si:si + per]
        sh = Image.new("RGB", (COLS * TW, per * (TH + 20)), (8, 8, 8))
        for j, t in enumerate(chunk):
            i0 = int(round(t * FPS))
            for k, off in enumerate((-1, 0, 1)):
                fi = min(max(i0 + off, 0), 1100)
                p = os.path.join(TILE_DIR, f"f{fi:04d}.jpg")
                if not os.path.exists(p):
                    continue
                im = Image.open(p).resize((TW, TH))
                d = ImageDraw.Draw(im)
                d.rectangle([0, 0, 130, 16], fill=(0, 0, 0))
                d.text((3, 2), f"#{si+j} {t - 0.0167*(1-off):.3f}{'*' if off==0 else ' '}",
                       fill=(0, 255, 255) if off == 0 else (200, 200, 200))
                sh.paste(im, (k * TW, j * (TH + 20)))
        sh.save(os.path.join(R55_WORK, f"hc_strip_{si:03d}.jpg"), quality=88)
    print("strips:", (len(cands) + per - 1) // per)


if __name__ == "__main__":
    extract_tiles()
    cands = comb_candidates()
    save_json(os.path.join(R55_WORK, "holdoutC_audio_candidates.json"),
              {"method": "R4 family: multi-band residual comb envelope, med+4MAD peaks, 55 ms merge",
               "av_off_prior_ms": 0.0, "candidates_s": cands})
    sheets()
    strips(cands)
    print("annotation materials ready - oracle JSON must be written from visual "
          "review BEFORE running the frozen detector on this window")
