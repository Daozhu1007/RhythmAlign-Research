"""S1a metric implementations. Fixed-gain everywhere; no per-output normalization.

Primary preservation metric (S0 convention): fixed-gain reconstruction SNR from NMSE
  NMSE = ||s_hat - s||^2 / (||s||^2 + eps);  SNR = -10 log10(NMSE).
SI-SDR is secondary (scale invariance can reward over-attenuation).
"""
from __future__ import annotations

import numpy as np

import common as C

EPS = C.METRIC_EPS


def nmse(s_hat: np.ndarray, s: np.ndarray) -> float:
    return float(np.sum((s_hat - s) ** 2) / (np.sum(s ** 2) + EPS))


def recon_snr(s_hat: np.ndarray, s: np.ndarray) -> float:
    return float(-10.0 * np.log10(max(nmse(s_hat, s), 1e-300)))


def si_sdr(s_hat: np.ndarray, s: np.ndarray, eps: float = 1e-8) -> float:
    """Standard scale-invariant SDR (Le Roux et al. 2019). Zero-mean first."""
    s_hat = s_hat - s_hat.mean()
    s = s - s.mean()
    alpha = np.dot(s_hat, s) / (np.dot(s, s) + eps ** 2)
    t = alpha * s
    e = s_hat - t
    return float(10.0 * np.log10((np.dot(t, t) + eps ** 2) / (np.dot(e, e) + eps ** 2)))


def projection_gain(s_hat: np.ndarray, s: np.ndarray) -> float:
    a = np.dot(s_hat, s) / (np.dot(s, s) + EPS ** 2)
    return float(20.0 * np.log10(abs(a) + EPS))


def projection_gain_signed(s_hat: np.ndarray, s: np.ndarray) -> float:
    return float(np.dot(s_hat, s) / (np.dot(s, s) + EPS ** 2))


# --------------------------------------------------------------- spectral
def _stft_mag(x, n_fft, hop, sr):
    from scipy.signal import get_window, stft as sp_stft
    w = get_window("hann", n_fft, fftbins=True)
    _, _, Z = sp_stft(x, fs=sr, window=w, nperseg=n_fft, noverlap=n_fft - hop,
                      boundary="even", padded=True)
    return np.abs(Z)


def mrstft_distance(s_hat: np.ndarray, s: np.ndarray, sr: int = C.WORK_SR) -> dict:
    """Mean of spectral convergence and log-magnitude L1 at 256/1024/2048 (quarter hop).

    Mirrors the S1b loss convention (windows 256/1024/2048, hann, quarter-window hop).
    """
    scs, l1s = [], []
    for n_fft in (256, 1024, 2048):
        hop = n_fft // 4
        Mh = _stft_mag(s_hat, n_fft, hop, sr)
        Ms = _stft_mag(s, n_fft, hop, sr)
        num = np.linalg.norm(Mh - Ms)
        scs.append(float(num / (np.linalg.norm(Ms) + EPS)))
        l1s.append(float(np.mean(np.abs(np.log(Mh + EPS) - np.log(Ms + EPS)))))
    return {"spectral_convergence_mean": float(np.mean(scs)),
            "log_mag_l1_mean": float(np.mean(l1s)),
            "per_size": {str(n): {"sc": scs[i], "log_mag_l1": l1s[i]}
                         for i, n in enumerate((256, 1024, 2048))}}


def band_energy(x: np.ndarray, sr: int, f_lo: float, f_hi: float) -> float:
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), 1.0 / sr)
    m = (fr >= f_lo) & (fr < f_hi)
    return float(np.sum(np.abs(X[m]) ** 2))


def high_frequency_retention(s_hat: np.ndarray, s: np.ndarray, sr: int = C.WORK_SR,
                             cutoff_hz: float = 8000.0) -> dict:
    """Fixed-gain band energy ratios output/target. 1.0 = exact retention."""
    out = {}
    for lo, hi in ((cutoff_hz, sr / 2), (0.0, cutoff_hz)):
        e_out = band_energy(s_hat, sr, lo, hi)
        e_ref = band_energy(s, sr, lo, hi)
        out[f"{int(lo)}_{int(hi)}_ratio"] = float(e_out / (e_ref + EPS ** 2))
    return out


# --------------------------------------------------------------- transient
def _rise_decay_times(env: np.ndarray, sr: int, peak_i: int,
                      decay_db: float = -20.0) -> tuple[float, float]:
    peak = env[peak_i]
    if peak <= EPS:
        return float("nan"), float("nan")
    thr_lo, thr_hi = 0.1 * peak, 0.9 * peak
    pre = env[max(0, peak_i - int(0.05 * sr)):peak_i + 1]
    rise = peak_i - (max(0, peak_i - int(0.05 * sr)) + int(np.argmax(pre >= thr_hi)))
    below = np.where(env[peak_i:] < peak * 10 ** (decay_db / 20))[0]
    decay = float(below[0] / sr) if len(below) else float("nan")
    return float(rise / sr), decay


def transient_fidelity(s_hat: np.ndarray, s: np.ndarray, sr: int,
                       events: list[dict]) -> dict:
    """Per-event peak ratio, rise-time and decay-time errors (events: {t0,t1})."""
    per = []
    for ev in events:
        i0, i1 = int(ev["t0"] * sr), int(ev["t1"] * sr)
        seg_r, seg_e = s_hat[i0:i1], s[i0:i1]
        if np.max(np.abs(seg_e)) <= EPS:
            continue
        pk_r = float(np.max(np.abs(seg_r)) / (np.max(np.abs(seg_e)) + EPS))
        env_e = np.abs(seg_e)
        env_r = np.abs(seg_r)
        pi = int(np.argmax(env_e))
        rise_e, dec_e = _rise_decay_times(env_e, sr, pi)
        rise_r, dec_r = _rise_decay_times(env_r, sr, int(np.argmax(env_r)))
        per.append({
            "t0": ev["t0"], "peak_ratio": pk_r,
            "rise_time_err_s": (rise_r - rise_e) if np.isfinite(rise_e) else None,
            "decay20_time_err_s": (dec_r - dec_e)
            if np.isfinite(dec_e) and np.isfinite(dec_r) else None,
        })
    if not per:
        return {"n_events": 0}
    pr = np.array([p["peak_ratio"] for p in per])
    return {
        "n_events": len(per),
        "peak_ratio_median": float(np.median(pr)),
        "peak_ratio_min": float(np.min(pr)),
        "rise_time_err_s_median": _nanmedian([p["rise_time_err_s"] for p in per]),
        "decay20_time_err_s_median": _nanmedian([p["decay20_time_err_s"] for p in per]),
        "per_event": per,
    }


def _nanmedian(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


# --------------------------------------------------- continuous/friction
def friction_fidelity(s_hat: np.ndarray, s: np.ndarray, sr: int,
                      friction_intervals: list[dict]) -> dict:
    """Envelope correlation + log-spectral distance inside friction intervals."""
    env_e, env_r = [], []
    lds = []
    for iv in friction_intervals:
        i0, i1 = int(iv["t0"] * sr), int(iv["t1"] * sr)
        se, re_ = s[i0:i1], s_hat[i0:i1]
        ae, ar = np.abs(se), np.abs(re_)
        k = max(1, int(0.005 * sr))
        env_e.append(ae[:len(ae) // k * k].reshape(-1, k).mean(axis=1))
        env_r.append(ar[:len(ar) // k * k].reshape(-1, k).mean(axis=1))
        Me, Mr = _stft_mag(se, 512, 128, sr), _stft_mag(re_, 512, 128, sr)
        lds.append(float(np.mean(np.abs(np.log(Me + EPS) - np.log(Mr + EPS)))))
    if not env_e:
        return {"n_intervals": 0}
    a = np.concatenate(env_e)
    b = np.concatenate(env_r)
    r = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
    return {"n_intervals": len(env_e), "envelope_pearson_r": r,
            "log_spectral_distance_mean": float(np.mean(lds))}


# ------------------------------------------------- suppression attribution
def exact_path_decomposition(stft, mask: np.ndarray, Y: np.ndarray, S: np.ndarray,
                             Nstft: np.ndarray, n_time: np.ndarray,
                             length: int) -> dict:
    """Exact per-path energies for a mask oracle.

    The STFT-mask path is LINEAR in Y, so R(M.*Y) = R(M.*S) + R(M.*N) exactly
    (verified per call via sum_check). Attenuation compares the masked nuisance
    path against the true time-domain nuisance at fixed gain.
    """
    x_t = stft.istft(mask * S, length)
    x_n = stft.istft(mask * Nstft, length)
    e_t, e_n = float(np.dot(x_t, x_t)), float(np.dot(x_n, x_n))
    e_nu = float(np.dot(n_time, n_time))
    return {
        "target_path_energy": e_t,
        "nuisance_path_energy": e_n,
        "nuisance_path_to_target_path_db": C.db((e_n + EPS ** 2) / (e_t + EPS ** 2)),
        "nuisance_attenuation_db": float(10.0 * np.log10((e_nu + EPS ** 2)
                                                         / (e_n + EPS ** 2))),
        "sum_check_max_abs_err": float(np.max(np.abs(
            x_t + x_n - stft.istft(mask * Y, length)))),
    }


def nuisance_projection(s_hat: np.ndarray, s: np.ndarray, nuisances: list) -> dict:
    """LSQ projection of s_hat onto [s, n1, ...]; diagnostic only (S0 data plan §7)."""
    X = np.column_stack([s] + list(nuisances))
    coef, *_ = np.linalg.lstsq(X, s_hat, rcond=None)
    resid = s_hat - X @ coef
    gram = X.T @ X
    cond = float(np.linalg.cond(gram)) if gram.size else float("inf")
    return {
        "coefficients": [float(c) for c in coef],
        "target_coefficient": float(coef[0]),
        "unexplained_energy_frac": float(np.dot(resid, resid) / (np.dot(s_hat, s_hat) + EPS ** 2)),
        "gram_condition_number": cond,
        "condition_flag": bool(cond > 1e4),
    }


def absent_false_output(s_hat: np.ndarray, n: np.ndarray) -> float:
    """Target-absent false-output energy: 10log10(||s_hat||^2 / ||n||^2)."""
    return float(10.0 * np.log10((np.dot(s_hat, s_hat) + EPS ** 2)
                                 / (np.dot(n, n) + EPS ** 2)))


def evaluate_variant(s_hat: np.ndarray, s: np.ndarray, sr: int,
                     events=None, friction=None) -> dict:
    """Preservation block for one variant against exact target s (fixed gain)."""
    out = {
        "nmse": nmse(s_hat, s),
        "recon_snr_db": recon_snr(s_hat, s),
        "si_sdr_db": si_sdr(s_hat, s),
        "projection_gain_db": projection_gain(s_hat, s),
        "mrstft": mrstft_distance(s_hat, s, sr),
        "hf_retention": high_frequency_retention(s_hat, s, sr),
    }
    if events:
        out["transient"] = transient_fidelity(s_hat, s, sr, events)
    if friction:
        out["friction"] = friction_fidelity(s_hat, s, sr, friction)
    return out
