# R5 Phase 8 - BLIND holdout oracle apparatus (no detector predictions used).
# Builds the R4-style annotation materials for the holdout window:
#   pass 1: 10 fps contact sheets (whole window)
#   pass 2: 30 fps contact sheets
#   pass 3: audio-onset candidates (music-removed comb envelope, R4 detector
#           family) -> 3-frame 60 fps zoom strips for visual verification
# The oracle JSON is written by the annotator AFTER reviewing these materials
# and BEFORE the frozen detector is ever run on this window.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw
from scipy import signal as sig

from r5_common import (OUT, WORK, LOG, SR, load_json, save_json, comb_envelope,
                       iter_frames)

FPS = 60.04
HO0 = load_json(os.path.join(OUT, "holdout_selection.json"))["holdout_start_s"]
DUR = load_json(os.path.join(OUT, "holdout_selection.json"))["duration_s"]
TILE_DIR = os.path.join(WORK, "ho_frames")


def extract_tiles():
    os.makedirs(TILE_DIR, exist_ok=True)
    n = 0
    for i, fr in iter_frames(os.path.join(OUT, "holdout_video.mp4"), w=480):
        Image.fromarray(fr.astype(np.uint8)).save(
            os.path.join(TILE_DIR, f"f{i:04d}.jpg"), quality=88)
        n = i + 1
    print("tiles:", n)
    return n


def comb_candidates():
    comb, _ = comb_envelope(os.path.join(WORK, "holdout_ctx.wav"),
                            os.path.join(WORK, "holdout_ref_warp.wav"), 3.0, DUR)
    np.save(os.path.join(WORK, "holdout_comb.npy"), comb)
    med = np.median(comb)
    mad = np.median(np.abs(comb - med)) + 1e-9
    pk, _ = sig.find_peaks(comb, height=med + 4.0 * mad,
                           distance=max(1, int(0.06 * SR / 48000 * 48000)))
    on = sorted(set(round(float(comb[k] * 0 + k / SR), 3) for k in pk))
    # merge within 55 ms (R4 06_candidates rule)
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
    # pass 1: every 6th frame (10 fps), 30 tiles/sheet
    p1 = list(range(0, n, 6))
    # pass 2: every 2nd frame (30 fps), 30 tiles/sheet
    p2 = list(range(0, n, 2))
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
            sh.save(os.path.join(WORK, f"ho_{name}_{si//per:02d}.jpg"), quality=85)
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
                fi = min(max(i0 + off, 0), 1080)
                p = os.path.join(TILE_DIR, f"f{fi:04d}.jpg")
                if not os.path.exists(p):
                    continue
                im = Image.open(p).resize((TW, TH))
                d = ImageDraw.Draw(im)
                d.rectangle([0, 0, 130, 16], fill=(0, 0, 0))
                d.text((3, 2), f"#{si+j} {t - 0.0167*(1-off):.3f}{'*' if off==0 else ' '}",
                       fill=(0, 255, 255) if off == 0 else (200, 200, 200))
                sh.paste(im, (k * TW, j * (TH + 20)))
        sh.save(os.path.join(WORK, f"ho_strip_{si:03d}.jpg"), quality=88)
    print("strips:", (len(cands) + per - 1) // per)


if __name__ == "__main__":
    import sys as _s
    n = extract_tiles()
    cands = comb_candidates()
    save_json(os.path.join(WORK, "holdout_audio_candidates.json"),
              {"method": "R4 family: multi-band residual comb envelope, med+4MAD peaks, 55 ms merge",
               "av_off_prior_ms": 0.0, "candidates_s": cands})
    sheets()
    strips(cands)
    print("annotation materials ready - oracle JSON must be written from visual "
          "review BEFORE running 70_holdout_run.py")
