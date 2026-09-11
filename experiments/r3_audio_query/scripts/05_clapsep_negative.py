# R3 Phase 5 / Experiment N1 - CLAPSep positive + negative control.
#   positive: Q1 (best player-interaction query from Phase 4)
#   negative: representative short segment of the CLEAN aligned music reference
#             (ref_warp_fixed, golden window) - explicitly tell the model
#             "do not keep this music".
# N2 (negative = isolated taiko event) is NOT executed: no segment in the
# golden sample can be confidently labelled taiko-only (taiko candidates are
# part of the song mix), and the brief forbids guessing labels.
import json
import os
import subprocess
import sys
import time

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf
import torch

R3 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(R3, "outputs")
LOG_DIR = os.path.join(R3, "logs")
R1_OUT = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs"
R1_WORK = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\work"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

POS_Q = os.path.join(R3, "queries", "query_q1.wav")
NEG_WIN = (8.0, 10.0)   # golden-timeline seconds of loudest aligned-music section


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, os.path.dirname(__file__))
    import clapsep_lib

    # negative query: cut clean music segment straight from the aligned reference
    neg_path = os.path.join(R3, "queries", "neg_music.wav")
    if not os.path.exists(neg_path):
        s0 = (3.0 + NEG_WIN[0])  # ref_warp_fixed carries 3 s pre-context
        subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", f"{s0:.3f}",
                        "-i", os.path.join(R1_WORK, "ref_warp_fixed.wav"),
                        "-t", f"{NEG_WIN[1] - NEG_WIN[0]:.3f}", neg_path], check=True)
        print(f"negative query: ref_warp_fixed[{NEG_WIN[0]}..{NEG_WIN[1]}] -> queries/neg_music.wav")

    model = clapsep_lib.load_clapsep(device)
    mixture, fs = librosa.load(os.path.join(R1_OUT, "golden_raw.wav"), sr=32000, mono=True)

    e_pos = clapsep_lib.embed_audio_query(model, POS_Q)
    e_neg = clapsep_lib.embed_audio_query(model, neg_path)
    print(f"pos norm {np.linalg.norm(e_pos):.3f} | neg norm {np.linalg.norm(e_neg):.3f}")

    target = clapsep_lib.separate(model, mixture, e_pos, e_neg, device)
    residual = (mixture - target).astype(np.float32)
    sf.write(os.path.join(OUT_DIR, "clapsep_n1_target.wav"), target, fs, subtype="FLOAT")
    sf.write(os.path.join(OUT_DIR, "clapsep_n1_residual.wav"), residual, fs, subtype="FLOAT")

    meta = {"name": "clapsep_n1", "positive": POS_Q, "negative": neg_path,
            "negative_window_s": list(NEG_WIN),
            "mixture_rms_db": db(mixture), "target_rms_db": db(target),
            "energy_ratio_db": db(target) - db(mixture)}
    with open(os.path.join(LOG_DIR, "05_clapsep_n1.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"clapsep_n1: target {db(target):.1f} dB vs mixture {db(mixture):.1f} dB "
          f"({db(target) - db(mixture):+.1f} dB) | done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
