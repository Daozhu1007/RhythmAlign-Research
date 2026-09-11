# R2 - final-mix previews: clean aligned reference + extracted target stem.
#
#   final_mix = 0.5 * ref_warp_fixed (48 kHz stereo, R1 aligned clean reference)
#             + 0.5 * target        (32 kHz mono model output, upsampled, both ch)
#
# Same 0.5/0.5 weighting convention as R1 final mixes. No limiter/compressor.
# The raw unprocessed target stem is kept as-is (written by 10_audiosep_run.py).
# If the model output level deviates from the raw recording's interaction level
# by more than 6 dB, an additional gain-adjusted LISTENING PREVIEW is written
# with the gain documented in logs/final_mix.json.
import json
import os
import sys

import numpy as np
import soundfile as sf
import librosa
from scipy import signal as sig

R2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(R2_DIR, "outputs")
LOG_DIR = os.path.join(R2_DIR, "logs")
R1_OUT = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs"
R1_WORK = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\work"
SR = 48000


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


def main():
    run_name = sys.argv[1]  # e.g. audiosep_raw_p1
    target32 = librosa.load(os.path.join(OUT_DIR, f"{run_name}_target.wav"), sr=SR, mono=True)[0]
    ref48, _ = sf.read(os.path.join(R1_WORK, "ref_warp_fixed.wav"), dtype="float32", always_2d=True)
    raw48, _ = sf.read(os.path.join(R1_OUT, "golden_raw.wav"), dtype="float32", always_2d=True)
    s0, s1 = int(3.0 * SR), int(18.0 * SR)   # golden window inside ctx-aligned files
    ref = ref48[s0:s1]
    raw = raw48[s0:s1]

    assert len(ref) == len(target32), f"length mismatch ref {len(ref)} vs target {len(target32)}"
    tgt = np.stack([target32, target32], axis=1)

    mix = 0.5 * ref + 0.5 * tgt
    out = os.path.join(OUT_DIR, f"{run_name}_final_mix.wav")
    sf.write(out, mix.astype(np.float32), SR, subtype="FLOAT")

    meta = {
        "run": run_name,
        "recipe": "0.5*ref_warp_fixed + 0.5*target(32k->48k mono->stereo)",
        "final_mix": out,
        "rms_db": {"ref": db(ref), "target": db(tgt), "final_mix": db(mix), "raw_recording": db(raw)},
    }

    # level check vs the raw recording's own content: compare the target stem
    # level to the mixture level in the same event windows (rough sanity bound)
    dev = db(tgt) - db(raw)
    meta["target_vs_raw_level_deviation_db"] = dev
    if abs(dev) > 6.0:
        gain = 10 ** ((db(raw) - db(tgt)) / 2)   # split the difference, preview only
        prev = 0.5 * ref + 0.5 * gain * tgt
        pout = os.path.join(OUT_DIR, f"{run_name}_final_mix_gainpreview.wav")
        sf.write(pout, prev.astype(np.float32), SR, subtype="FLOAT")
        meta["gain_preview"] = {"file": pout, "target_gain": float(gain),
                                "note": "listening preview only; raw target stem preserved"}

    with open(os.path.join(LOG_DIR, "final_mix.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
