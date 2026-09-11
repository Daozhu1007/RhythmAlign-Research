# R4 Phase 1 - per-frame judgment-text feature extraction + onset candidates.
import json
import os
import subprocess

import numpy as np
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
VID = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs\golden_video.mp4"
R4 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(R4, "work")
FPS = 60.04

# judgment zone in rotated 960x540 coords
ZX0, ZY0, ZX1, ZY1 = 275, 215, 410, 300  # text + FAST/SLOW region

def main():
    cmd = [FF, "-hide_banner", "-nostdin", "-i", VID,
           "-vf", "transpose=2,scale=960:-2",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    W, H = 960, 540
    n = W * H * 3
    feats = []
    prev = None
    idx = 0
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3)[:, :, ::-1].astype(np.float32)  # RGB-> RGB (ffmpeg gives RGB)
        z = fr[ZY0:ZY1, ZX0:ZX1]
        r, g, b = z[..., 0], z[..., 1], z[..., 2]
        # CRITICAL PERFECT yellow-green: high R and G, much lower B
        ymask = (r > 150) & (g > 150) & (b < 0.55 * (r + g) / 2) & (np.abs(r - g) < 90)
        wmask = (r > 210) & (g > 210) & (b > 200)
        diff = 0.0 if prev is None else float(np.mean(np.abs(z - prev)))
        feats.append({"i": idx, "y": int(ymask.sum()), "w": int(wmask.sum()),
                      "yb": float(ymask.mean()), "diff": diff})
        prev = z
        idx += 1
    proc.wait()
    print("frames:", idx)
    with open(os.path.join(WORK, "judge_feats.json"), "w") as f:
        json.dump(feats, f)

if __name__ == "__main__":
    main()
