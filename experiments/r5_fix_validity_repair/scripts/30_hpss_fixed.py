# R5-FIX PART 3 - the ONE corrected HPSS control on the golden window.
# The historical C2 put the TIME-axis median (harmonic estimate) in the
# numerator of its "percussive" mask, i.e. it enhanced sustained structure.
# The corrected control uses the standard Fitzgerald median-filter HPSS
# orientation with the SAME frozen parameters as the historical build:
#   STFT: n_fft 2048, hop 512 (scipy.signal, same as R4 20_build.py)
#   harmonic estimate H = median over TIME   -> size (1, 31)   (horizontal structures)
#   percussive estimate P = median over FREQ -> size (31, 1)  (vertical structures)
#   percussive soft mask M = P / (H + P + 1e-9)   [numerator = percussive estimate]
#   gain g = clip(0.35 + 0.65 * M^0.8, 0.35, 1.0)   [same floor/power/emphasis as R4]
#   perc = istft(STFT(raw_ch) * g) per channel;  C2_fixed = env_fixed * perc
#   final mix = 0.5 * ref + 0.5 * C2_fixed
# Single implementation, NO parameter sweep, NO tuning.
import os
import sys

import numpy as np
import soundfile as sf
from scipy import ndimage as ndi
from scipy import signal as sig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixenv as fx

N_FFT, HOP, MED, FLOOR, MASK_POW, EMPH = 2048, 512, 31, 0.35, 0.8, 0.65


def percussive_blend_FIXED(raw):
    mono = raw.mean(axis=1)
    _, _, Z = sig.stft(mono, fs=fx.SR, nperseg=N_FFT, noverlap=N_FFT - HOP)
    H = ndi.median_filter(np.abs(Z), size=(1, MED), mode="reflect")   # time-axis median
    P = ndi.median_filter(np.abs(Z), size=(MED, 1), mode="reflect")   # frequency-axis median
    M = P / (H + P + 1e-9)                                            # percussive numerator
    g = np.clip(FLOOR + EMPH * np.power(M, MASK_POW), FLOOR, 1.0)
    out = np.zeros_like(raw)
    for c in range(raw.shape[1]):
        _, _, Zc = sig.stft(raw[:, c], fs=fx.SR, nperseg=N_FFT, noverlap=N_FFT - HOP)
        _, xr = sig.istft(Zc * g, fs=fx.SR, nperseg=N_FFT, noverlap=N_FFT - HOP)
        out[: min(len(xr), len(raw)), c] = xr[: len(raw)]
    return out, g


def main():
    raw = sf.read(os.path.join(fx.R1_OUT, "golden_raw.wav"), dtype="float64", always_2d=True)[0]
    ref = sf.read(os.path.join(fx.R1_WORK, "ref_warp_fixed.wav"), dtype="float64", always_2d=True)[0][3 * fx.SR:18 * fx.SR]
    env = np.load(os.path.join(fx.FIX_WORK, "env_golden_oracle_fixed.npy"))

    perc, g = percussive_blend_FIXED(raw)
    c2 = env[:, None] * perc
    mix = 0.5 * ref + 0.5 * c2
    sf.write(os.path.join(fx.FIX_OUT, "golden_C2_fixed_interaction.wav"),
             c2.astype(np.float32), fx.SR, subtype="FLOAT")
    sf.write(os.path.join(fx.FIX_OUT, "golden_C2_fixed_final_mix.wav"),
             mix.astype(np.float32), fx.SR, subtype="FLOAT")

    # arithmetic verification (same convention as PART 2)
    ri = sf.read(os.path.join(fx.FIX_OUT, "golden_C2_fixed_interaction.wav"),
                 dtype="float32", always_2d=True)[0].astype(np.float64)
    rm = sf.read(os.path.join(fx.FIX_OUT, "golden_C2_fixed_final_mix.wav"),
                 dtype="float32", always_2d=True)[0].astype(np.float64)
    err = float(np.max(np.abs(rm - (0.5 * ref + 0.5 * ri))))

    # mask sanity: on a click-like vertical event the mask should now be higher
    # than the historical reversed mask would have been at that same frame.
    out = {
        "phase": "R5-FIX PART 3 corrected HPSS control (single implementation, no sweep)",
        "definition": {
            "stft": {"n_fft": N_FFT, "hop": HOP, "library": "scipy.signal (as R4)"},
            "harmonic_estimate": f"median_filter(|Z|, size=(1,{MED})) along TIME",
            "percussive_estimate": f"median_filter(|Z|, size=({MED},1)) along FREQUENCY",
            "mask": "M = P / (H + P + 1e-9)  [percussive estimate in numerator]",
            "gain": f"g = clip({FLOOR} + {EMPH} * M^{MASK_POW}, {FLOOR}, 1.0)",
            "composition": "C2_fixed = corrected_C1_envelope * percussive_blend(raw); final = 0.5*ref + 0.5*C2_fixed",
        },
        "historical_defect": "R4 C2 numerator was the TIME-axis median (harmonic estimate): the mask enhanced sustained structure instead of transients.",
        "mask_gain_stats": {"mean": float(g.mean()), "min": float(g.min()), "max": float(g.max())},
        "mean_env": float(env.mean()),
        "final_mix_arith_max_err_readback_float32": err,
        "outputs": ["golden_C2_fixed_interaction.wav", "golden_C2_fixed_final_mix.wav"],
        "scope": ("This experiment answers ONLY whether the corrected percussive HPSS "
                  "control remains inferior to corrected C1. It does NOT revive the HPSS research branch."),
    }
    assert err <= 1.2e-7   # <= 1 float32 ULP at full scale (1.19e-7)
    fx.save_json(os.path.join(fx.FIX_LOG, "part3_hpss_summary.json"), out)
    print("golden_C2_fixed written; mask mean gain", round(float(g.mean()), 4),
          "| arith err", err)


if __name__ == "__main__":
    main()
