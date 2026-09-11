"""Part-15 synthetic validation cases (shared by tests/ and the fixture pipeline).

Six required cases where the expected oracle behavior is known:
 1. Y = S                      -> bounded oracle reproduces S closely
 2. S = 0                      -> target output is zero
 3. S and N in separate TF regions -> bounded oracle separates nearly perfectly
 4. destructive interference |S|>|Y| -> bounded mask visibly fails; complex oracle recovers
 5. continuous tone target     -> no artificial chopping from STFT/ISTFT
 6. impulse target             -> transient timing/amplitude sanity
"""
from __future__ import annotations

import numpy as np

import common as C
from stft_clapsep import (ClapSepSTFT, bounded_real_mask, complex_ratio_mask,
                          run_oracles)

SR = C.WORK_SR
ST = ClapSepSTFT()


def _sig(t):
    return np.arange(int(t * SR)) / SR


def case_y_equals_s() -> dict:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(int(1.0 * SR)) * 0.1
    res = run_oracles(ST, x, x, opt_max_iter=20)
    s = res["outputs"]
    err_b = float(np.max(np.abs(s["BOUNDED_REAL_MASK_ORACLE"] - x)))
    err_o = float(np.max(np.abs(s["OPTIMIZED_BOUNDED_MASK_ORACLE"] - x)))
    err_rt = float(np.max(np.abs(s["TRUE_TARGET_STFT_ROUNDTRIP"] - x)))
    return {"case": "y_equals_s", "max_err_bounded": err_b,
            "max_err_optimized": err_o, "max_err_roundtrip": err_rt,
            "pass": bool(err_b < 5e-4 and err_o <= err_b + 1e-9 and err_rt < 5e-4)}


def case_s_zero() -> dict:
    rng = np.random.default_rng(2)
    y = rng.standard_normal(int(0.5 * SR)) * 0.1
    res = run_oracles(ST, y, np.zeros_like(y), opt_max_iter=20)
    s = res["outputs"]
    e_b = float(np.max(np.abs(s["BOUNDED_REAL_MASK_ORACLE"])))
    e_o = float(np.max(np.abs(s["OPTIMIZED_BOUNDED_MASK_ORACLE"])))
    return {"case": "s_zero", "max_abs_out_bounded": e_b,
            "max_abs_out_optimized": e_o,
            "pass": bool(e_b == 0.0 and e_o < 1e-9)}


def case_separate_regions() -> dict:
    sr = SR
    n = int(1.0 * sr)
    t = _sig(1.0)
    # target: 3 kHz tone band (time-limited); nuisance: 80 Hz + 12 kHz bands
    s = np.zeros(n)
    s[int(0.2 * sr):int(0.8 * sr)] = 0.5 * np.sin(
        2 * np.pi * 3000 * t[int(0.2 * sr):int(0.8 * sr)])
    y = s + 0.4 * np.sin(2 * np.pi * 80 * t) + 0.3 * np.sin(2 * np.pi * 12000 * t)
    res = run_oracles(ST, y, s, opt_max_iter=30)
    from evaluator import recon_snr, si_sdr
    sb = res["outputs"]["BOUNDED_REAL_MASK_ORACLE"]
    snr = recon_snr(sb, s)
    si = si_sdr(sb, s)
    return {"case": "separate_tf_regions", "recon_snr_bounded_db": snr,
            "si_sdr_bounded_db": si,
            "pass": bool(snr > 20.0 and si > 20.0)}


def case_destructive() -> dict:
    n = int(0.5 * SR)
    t = _sig(0.5)
    s = 0.3 * np.sin(2 * np.pi * 1000 * t) + 0.05 * np.sin(2 * np.pi * 3 * t)
    y = s - 0.8 * s  # Y = 0.2*S in every bin -> |S| > |Y| everywhere
    res = run_oracles(ST, y, s, opt_max_iter=60)
    from evaluator import projection_gain, recon_snr
    s_b = res["outputs"]["BOUNDED_REAL_MASK_ORACLE"]
    s_c = res["outputs"]["COMPLEX_RATIO_ORACLE"]
    s_o = res["outputs"]["OPTIMIZED_BOUNDED_MASK_ORACLE"]
    # Bounded mask saturates at 1 -> output ~ 0.2*s: correlated but 14 dB too
    # quiet (projection gain ~ -14 dB); no M in [0,1] can do better. Complex
    # ratio mask (unbounded) recovers the target amplitude exactly.
    g_b = projection_gain(s_b, s)
    g_o = projection_gain(s_o, s)
    snr_c = recon_snr(s_c, s)
    return {"case": "destructive_interference",
            "bounded_projection_gain_db": g_b,
            "optimized_projection_gain_db": g_o,
            "complex_recon_snr_db": snr_c,
            "pass": bool(g_b <= -10.0 and g_o <= -10.0 and snr_c > 30.0)}


def case_continuous_tone() -> dict:
    n = int(1.0 * SR)
    t = _sig(1.0)
    s = 0.2 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 443 * t)
    res = run_oracles(ST, s.copy(), s, opt_max_iter=20)
    s_b = res["outputs"]["BOUNDED_REAL_MASK_ORACLE"]
    s_rt = res["outputs"]["TRUE_TARGET_STFT_ROUNDTRIP"]
    # chopping metric: ripple of the analytic envelope vs roundtrip floor
    def ripple(x):
        k = int(0.01 * SR)
        m = len(x) // k
        env = np.abs(x[:m * k]).reshape(m, k).max(axis=1)
        return float((env.max() - env.min()) / (env.max() + 1e-12))
    return {"case": "continuous_tone_no_chopping",
            "max_err_bounded_vs_target": float(np.max(np.abs(s_b - s))),
            "ripple_bounded": ripple(s_b), "ripple_roundtrip": ripple(s_rt),
            "pass": bool(np.max(np.abs(s_b - s)) < 5e-4)}


def case_impulse() -> dict:
    n = int(0.5 * SR)
    s = np.zeros(n)
    pk = int(0.2 * SR)
    s[pk:pk + 3] = [0.5, -0.3, 0.2]  # short transient
    res = run_oracles(ST, s.copy(), s, opt_max_iter=20)
    s_b = res["outputs"]["BOUNDED_REAL_MASK_ORACLE"]
    pk_b = int(np.argmax(np.abs(s_b)))
    peak_ratio = float(np.max(np.abs(s_b)) / np.max(np.abs(s)))
    return {"case": "impulse_timing_amplitude",
            "peak_pos_target": pk, "peak_pos_bounded": pk_b,
            "peak_offset_samples": abs(pk_b - pk),
            "peak_amplitude_ratio": peak_ratio,
            "pass": bool(abs(pk_b - pk) <= 2 and 0.9 <= peak_ratio <= 1.1)}


ALL_CASES = [case_y_equals_s, case_s_zero, case_separate_regions,
             case_destructive, case_continuous_tone, case_impulse]


def run_all() -> dict:
    results = [fn() for fn in ALL_CASES]
    return {"part15_synthetic_validation": results,
            "all_pass": bool(all(r["pass"] for r in results)),
            "stft_config": C.CLAPSEP_STFT,
            "mask_config": C.CLAPSEP_MASK}
