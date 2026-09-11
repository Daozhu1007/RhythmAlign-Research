# R4 Phase 1 helper - decode all frames of golden_video.mp4 (rotated upright),
# save JPEG tiles + per-frame motion/LED metrics for contact screening.
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
VID = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs\golden_video.mp4"
R4 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TILE_DIR = os.path.join(R4, "frames")
WORK = os.path.join(R4, "work")

W, H = 480, 270  # rotated working size
CX, CY = 257.0, 131.0  # screen circle center (scaled from 960x540 estimate)
R_IN, R_OUT = 100.0, 142.0  # bezel LED band (screen glass ends ~ r=88-95)

SECTORS = []
for k in range(8):
    a = np.deg2rad(-90 + 45 * k)  # k=0 top, clockwise
    SECTORS.append((k, a))

def main():
    os.makedirs(TILE_DIR, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    cmd = [FF, "-hide_banner", "-nostdin", "-i", VID,
           "-vf", "transpose=2,scale=480:-2",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = W * H * 3
    prev_gray = None
    metrics = []
    idx = 0
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32)
        gray = fr.mean(axis=2)
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
        diff = 0.0 if prev_gray is None else float(np.mean(np.abs(gray - prev_gray)))
        bez = (r >= R_IN) & (r <= R_OUT)
        bezel = {
            "motion": float(np.mean(np.abs(gray - prev_gray)[bez])) if prev_gray is not None else 0.0,
        }
        for k, a in SECTORS:
            da = (xx - CX) * np.sin(a) - (yy - CY) * np.cos(a)  # >0 clockwise from top
            dr = (xx - CX) * np.cos(a) + (yy - CY) * np.sin(a)
            half = np.abs(np.arctan2(da, dr)) < np.deg2rad(22.5)
            m = bez & half
            bezel[f"b{int(np.round(np.degrees(a)) % 360)}"] = float(fr[m].mean())
        Image.fromarray(fr.astype(np.uint8)).save(
            os.path.join(TILE_DIR, f"f{idx:04d}.jpg"), quality=88)
        metrics.append({"i": idx, "diff": diff, **bezel})
        prev_gray = gray
        idx += 1
    proc.wait()
    print("frames:", idx)
    with open(os.path.join(WORK, "frame_metrics.json"), "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    main()
