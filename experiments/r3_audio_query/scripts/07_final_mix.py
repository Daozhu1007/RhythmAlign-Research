# R3 final-mix previews: aligned pristine clean reference + extracted target.
#   final_mix = 0.5 * ref_warp_fixed (48 kHz stereo, R1 aligned clean music)
#             + 0.5 * target        (32 kHz mono model output, upsampled, both ch)
# Fixed documented 0.5/0.5 gain (R1/R2 convention). No limiter/compressor.
# Raw target stems are kept untouched next to the mixes.
import json
import os

import librosa
import numpy as np
import soundfile as sf

R3 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(R3, "outputs")
LOG_DIR = os.path.join(R3, "logs")
R1_OUT = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs"
R1_WORK = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\work"
SR = 48000
RUNS = ["clapsep_q1", "clapsep_n1"]   # best 2 R3 targets (R2 best audiosep_raw_p2 already has one)


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


def main():
    ref48, _ = sf.read(os.path.join(R1_WORK, "ref_warp_fixed.wav"), dtype="float32", always_2d=True)
    raw48, _ = sf.read(os.path.join(R1_OUT, "golden_raw.wav"), dtype="float32", always_2d=True)
    ref = ref48[int(3.0 * SR):int(18.0 * SR)]
    raw = raw48[int(3.0 * SR):int(18.0 * SR)]

    metas = []
    for run in RUNS:
        target32 = librosa.load(os.path.join(OUT_DIR, f"{run}_target.wav"), sr=SR, mono=True)[0]
        assert len(ref) == len(target32), (len(ref), len(target32))
        tgt = np.stack([target32, target32], axis=1)
        mix = 0.5 * ref + 0.5 * tgt
        out = os.path.join(OUT_DIR, f"{run}_final_mix.wav")
        sf.write(out, mix.astype(np.float32), SR, subtype="FLOAT")
        meta = {"run": run,
                "recipe": "0.5*ref_warp_fixed + 0.5*target(32k->48k mono->stereo), fixed gain, no limiter",
                "final_mix": out,
                "rms_db": {"ref": db(ref), "target": db(tgt), "final_mix": db(mix),
                           "raw_recording": db(raw)},
                "target_vs_raw_level_deviation_db": db(tgt) - db(raw)}
        dev = abs(meta["target_vs_raw_level_deviation_db"])
        if dev > 6.0:
            gain = 10 ** ((db(raw) - db(tgt)) / 2)
            prev = 0.5 * ref + 0.5 * gain * tgt
            pout = os.path.join(OUT_DIR, f"{run}_final_mix_gainpreview.wav")
            sf.write(pout, prev.astype(np.float32), SR, subtype="FLOAT")
            meta["gain_preview"] = {"file": pout, "target_gain": float(gain),
                                    "note": "listening preview only; raw target stem preserved"}
        metas.append(meta)
        print(f"{run}: final mix rms {db(mix):.1f} dB | target-vs-raw dev "
              f"{meta['target_vs_raw_level_deviation_db']:+.1f} dB | preview: {dev > 6.0}")

    with open(os.path.join(LOG_DIR, "07_final_mix.json"), "w", encoding="utf-8") as f:
        json.dump(metas, f, indent=2)
    print("saved final mixes + logs/07_final_mix.json")


if __name__ == "__main__":
    main()
