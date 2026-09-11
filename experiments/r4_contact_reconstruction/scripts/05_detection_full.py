# R4 - full-clip detection pass: judgment-text onsets (960x540 decode) + LED sector
# flashes (from saved 480x270 tiles, elliptical button geometry). Candidate generators.
import json
import os
import subprocess

import numpy as np
import imageio_ffmpeg
from PIL import Image
from scipy import signal as sig

FF = imageio_ffmpeg.get_ffmpeg_exe()
VID = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs\golden_video.mp4"
R4 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(R4, "work")
TILE_DIR = os.path.join(R4, "frames")
FPS = 60.04

# ---- judgment zone features at 960x540 ----
ZX0, ZY0, ZX1, ZY1 = 260, 200, 450, 330  # generous text zone (rotated 960 coords)

def judge_features():
    cmd = [FF, "-hide_banner", "-nostdin", "-i", VID,
           "-vf", "transpose=2,scale=960:-2", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    W, H = 960, 540
    n = W * H * 3
    feats = []
    idx = 0
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32)
        z = fr[ZY0:ZY1, ZX0:ZX1]
        r, g, b = z[..., 0], z[..., 1], z[..., 2]
        # CP yellow-green: r~g high, b low. PERFECT orange: r high, g mid, b low.
        warm = (r > 130) & (b < 0.75 * np.maximum(r, 1)) & ((r + g) > 250)
        white = (r > 200) & (g > 200) & (b > 180)
        feats.append({"i": idx, "warm": int(warm.sum()), "white": int(white.sum()),
                      "warmf": float(warm.mean())})
        idx += 1
    proc.wait()
    return feats

def main():
    feats = judge_features()
    print("judge frames:", len(feats))
    warm = np.array([f["warm"] for f in feats], float)
    white = np.array([f["white"] for f in feats], float)
    json.dump(feats, open(os.path.join(WORK, "judge_feats2.json"), "w"))
    np.save(os.path.join(WORK, "judge_series.npy"), np.stack([warm, white]))

    # ---- LED sector brightness from tiles, elliptical geometry ----
    W480, H480 = 480, 270
    cx, cy = 255.0, 131.0
    a, b = 133.0, 95.0  # glass ellipse semi-axes (480x270)
    yy, xx = np.mgrid[0:H480, 0:W480]
    dx, dy = xx - cx, yy - cy
    rad = np.sqrt((dx / a) ** 2 + (dy / b) ** 2)
    band = (rad > 1.02) & (rad < 1.45)
    sectors = {}
    for k in range(8):
        ang = np.deg2rad(45 * k)  # 0=E, 45=SE, 90=S, ... (atan2(dy,dx) style, y down)
        da = dy * np.cos(ang) - dx * np.sin(ang)
        dr = dx * np.cos(ang) + dy * np.sin(ang)
        half = np.abs(np.arctan2(da, dr)) < np.deg2rad(20.0)
        sectors[f"s{45*k}"] = band & half
    names = {0: "E", 45: "SE", 90: "S", 135: "SW", 180: "W", 225: "NW", 270: "N", 315: "NE"}
    S = np.zeros((901, 8))
    key = ["s0", "s45", "s90", "s135", "s180", "s225", "s270", "s315"]
    for i in range(901):
        im = np.asarray(Image.open(os.path.join(TILE_DIR, f"f{i:04d}.jpg")), dtype=np.float32)
        for si, k in enumerate(key):
            m = sectors[k]
            S[i, si] = im[..., 0][m].mean() * 0.5 + im[..., 1][m].mean() * 0.5  # warm-ish brightness
    np.save(os.path.join(WORK, "led_series.npy"), S)
    json.dump({"names": [names[45*k] for k in range(8)]},
              open(os.path.join(WORK, "led_names.json"), "w"))
    print("led series saved")

if __name__ == "__main__":
    main()
