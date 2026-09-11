# R5 diagnostic - 3-frame zoom strips for inspecting region-interior FPs / FNs.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw
from r5_common import iter_frames, R1_OUT, JZ, load_json, save_json
from importlib import import_module
d20 = import_module("20_detector_dev")
FPS = 60.04

OUT_DIR = os.path.join(d20.OUT, "strips_dev")
os.makedirs(OUT_DIR, exist_ok=True)


def make_strip(vid_idx_map, t, tag, fi_lookup):
    """3 frames around t, cropped to the glass area, packed side by side."""
    i0 = int(round(t * FPS))
    tiles = []
    x0, y0, x1, y1 = 380, 60, 960, 540  # glass + JZ area
    for off in (-1, 0, 1):
        fi = min(max(i0 + off, 0), 900)
        fr = fi_lookup(fi)
        im = Image.fromarray(fr[y0:y1, x0:x1].astype(np.uint8)).resize((384, 320))
        d = ImageDraw.Draw(im)
        d.text((3, 2), f"{tag} t={t:.3f} {off:+d}f", fill=(0, 255, 255) if off == 0 else (180, 180, 180))
        tiles.append(im)
    sheet = Image.new("RGB", (384 * 3 + 8, 320), (0, 0, 0))
    for j, im in enumerate(tiles):
        sheet.paste(im, (j * 388, 0))
    return sheet


def main():
    VID = os.path.join(R1_OUT, "golden_video.mp4")
    # cache only the frames we need (uint8)
    frames_uint8 = {}

    def fi_lookup(fi):
        return frames_uint8[fi]

    series = d20.load_dev()
    oracle, contacts, noncontacts = d20.oracle_events()
    regions = [rg for rg in oracle["regions"] if not rg["id"].startswith("G")]

    def in_region(t):
        return any(rg["t0"] - 0.05 <= t <= rg["t1"] + 0.05 for rg in regions)

    score, cands = d20_detect(series)
    tp = np.array([c["t_video"] for c in cands])
    tc = np.array([e["t_video"] for e in contacts])
    fp_out, fn_out = [], []
    for c in cands:
        t = c["t_video"]
        if np.min(np.abs(tc - t)) > 0.050 and in_region(t):
            fp_out.append(t)
    for e in contacts:
        t = e["t_video"]
        if np.min(np.abs(tp - t)) > 0.050 and in_region(t):
            fn_out.append(t)
    fp_sample = [fp_out[i] for i in np.linspace(0, len(fp_out) - 1, min(8, len(fp_out))).astype(int)]
    fn_sample = [fn_out[i] for i in np.linspace(0, len(fn_out) - 1, min(8, len(fn_out))).astype(int)]
    want = set()
    for t in fp_sample + fn_sample:
        i0 = int(round(t * FPS))
        want.update(range(i0 - 1, i0 + 2))
    for i, fr in iter_frames(VID):
        if i in want:
            frames_uint8[i] = fr.astype(np.uint8)
        if len(frames_uint8) == len(want):
            break
    for k, t in enumerate(fp_sample):
        make_strip(None, t, f"FP{k:02d}", fi_lookup).save(os.path.join(OUT_DIR, f"fp_{k:02d}.png"))
    for k, t in enumerate(fn_sample):
        make_strip(None, t, f"FN{k:02d}", fi_lookup).save(os.path.join(OUT_DIR, f"fn_{k:02d}.png"))
    save_json(os.path.join(d20.LOG, "fp_fn_samples.json"),
              {"fp": [round(float(t), 3) for t in fp_sample],
               "fn": [round(float(t), 3) for t in fn_sample]})
    print("strips saved:", len(fp_sample), "FP,", len(fn_sample), "FN")


def d20_detect(series):
    from r5_common import detect_candidates
    return detect_candidates(series, FPS, {"threshold": 0.40, "slope_min": 0.15})


if __name__ == "__main__":
    main()
