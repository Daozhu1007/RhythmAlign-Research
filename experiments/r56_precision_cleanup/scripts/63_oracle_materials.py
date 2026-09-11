# R5.6 Holdout D - step 4: VIDEO-FIRST blind annotation materials.
# Pass 1: 10 fps contact sheets (whole window, structure + coarse events)
# Pass 2: 30 fps contact sheets (judgment-time precision)
# Pass 3: 60 fps zoom strips for dense bursts / ambiguous moments (generated
#         on demand at specified times)
# Pass 4 (LAST): audio-comb candidates for gap-checking only - the oracle is
#         NOT allowed to be a superset of audio candidates.
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common56 import R56_OUT, R56_WORK, load_json, save_json
from r5_common import FF

VID = os.path.join(R56_OUT, "holdoutD_video.mp4")


def sheets(prefix, fps, tile, t0=0.0, t1=18.0, scale=288):
    out = os.path.join(R56_WORK, f"{prefix}_%02d.jpg")
    cmd = [FF, "-hide_banner", "-nostdin", "-y",
           "-ss", f"{t0}", "-i", VID, "-t", f"{t1 - t0}",
           "-vf", f"fps={fps},scale={scale}:-2,tile={tile}",
           "-q:v", "3", out]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    n = 0
    while os.path.exists(out % (n + 1)):
        n += 1
    print(f"{prefix}: {n} sheets ({tile} = {tile.split('x')[0]}x{tile.split('x')[1]} "
          f"tiles @ {fps} fps)")
    return n


def strips(prefix, times, scale=480):
    """5-frame 60fps strip centered on t (video seconds): frames at
    t-66ms, -33ms, 0, +33ms, +66ms (every 2nd frame, tile vertically)."""
    meta = []
    for i, t in enumerate(times):
        out = os.path.join(R56_WORK, f"{prefix}_{i:03d}.jpg")
        cmd = [FF, "-hide_banner", "-nostdin", "-y",
               "-ss", f"{max(t - 4 / 60.0, 0):.4f}", "-i", VID,
               "-frames:v", "9",
               "-vf", "select=not(mod(n\\,2)),scale=480:-2,tile=1x5",
               "-frames:v", "1", "-q:v", "3", out]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        meta.append({"i": i, "t_center": round(t, 3)})
    print(f"{prefix}: {len(times)} strips")
    return meta


if __name__ == "__main__":
    n1 = sheets("hdp1_10fps", 10, "5x5")
    n2 = sheets("hdp2_30fps", 30, "5x6")
    save_json(os.path.join(R56_WORK, "holdoutD_pass12_meta.json"),
              {"pass1": {"n_sheets": n1, "fps": 10, "tile": "5x5"},
               "pass2": {"n_sheets": n2, "fps": 30, "tile": "5x6"}})
