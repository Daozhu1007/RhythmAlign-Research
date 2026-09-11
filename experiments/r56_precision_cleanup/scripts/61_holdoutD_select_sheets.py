# R5.6 Holdout D - step 2: 10 fps selection contact sheets for the two allowed
# candidate zones, for VISUAL shortlisting (video-first workflow).
import os
import subprocess

from common56 import R1_WORK, R56_WORK, load_json, save_json
from r5_common import FF, HANCAM_MP4

ZONES = {"zoneA": (58.0, 84.0), "zoneB": (102.0, 136.0)}
DENS = load_json(os.path.join(R56_WORK, "holdoutD_density.json"))


def main():
    meta = {}
    for name, (t0, t1) in ZONES.items():
        out = os.path.join(R56_WORK, f"hd_sel_{name}_%02d.jpg")
        cmd = [FF, "-hide_banner", "-nostdin", "-y",
               "-ss", f"{t0}", "-i", HANCAM_MP4, "-t", f"{t1 - t0}",
               "-vf", "fps=10,scale=288:-2,tile=5x5", "-q:v", "4", out]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        # count produced sheets
        n = 0
        while os.path.exists(out % (n + 1)):
            n += 1
        meta[name] = {"t0": t0, "t1": t1, "sheets": n,
                      "tiles_per_sheet": 25, "fps": 10}
        print(f"{name}: [{t0},{t1}] -> {n} sheets (5x5 tiles @10 fps, 288 px wide)")
    save_json(os.path.join(R56_WORK, "holdoutD_selection_sheets_meta.json"), meta)


if __name__ == "__main__":
    main()
