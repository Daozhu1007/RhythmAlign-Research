# Validated delay estimation utilities for the R1 experiment.
# Convention (empirically verified on synthetic data):
#   irfft(X*conj(R), N2)[k] holds circular lag m = k (k <= N2/2) or m = k - N2 (k > N2/2),
#   where the linear correlation value for lag m is sum_n x[n] * rw[n - m].
#   scipy.correlate(x, rw, 'full')[k] uses m = k - (len(rw)-1) with the same value.
import numpy as np
from scipy import signal as sig


def next_pow2(n):
    return 1 << int(np.ceil(np.log2(n)))


def full_xcorr(xw, rwz, N2):
    """Linear cross-correlation c[m] = sum xw[n] rwz[n-m], m in [-(L-1), N-1].
    Returns (cc_full, m_offset) with cc_full[i] = c[i - m_offset]."""
    X = np.fft.rfft(xw, N2)
    R = np.fft.rfft(rwz, N2)
    G = X * np.conj(R)
    cc = np.fft.irfft(G, N2)
    Lm1 = len(rwz) - 1
    Nn = len(xw)
    cc_full = np.concatenate([cc[N2 - Lm1:N2], cc[:Nn]])
    return cc_full, Lm1


def parabolic(vals):
    k = int(np.argmax(vals))
    if 0 < k < len(vals) - 1:
        y0, y1, y2 = vals[k - 1], vals[k], vals[k + 1]
        den = y0 - 2 * y1 + y2
        dx = float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if abs(den) > 1e-15 else 0.0
    else:
        dx = 0.0
    return k, dx


def estimate_delay(x, rw, sr, n0, d_axis, method="ncc", win_hann=True):
    """x: handcam window (N,). rw: ref buffer (L,).
    Mapping: if the true delay is d (t_ref - t_hand), then
        x[n] matches rw[n0 + (d - d_axis[0]) * sr + n].
    Returns (d_hat, peak_value, second_peak_ratio)."""
    N = len(x)
    L = len(rw)
    N2 = next_pow2(N + L)
    w = sig.windows.hann(N, sym=False) if win_hann else np.ones(N)
    xw = (x - x.mean()) * w
    rwz = rw - rw.mean()
    d0 = float(d_axis[0])
    # lag m for delay d: x[n] <-> rw[n - m] => -m = n0 + (d-d0)*sr => m = -n0 - (d-d0)*sr
    m_lo = -n0 - (float(d_axis[-1]) - d0) * sr   # d max
    m_hi = -n0 - (float(d_axis[0]) - d0) * sr    # d min
    if method == "ncc":
        cc_full, mo = full_xcorr(xw, rwz, N2)
        seg = cc_full
    elif method == "gcc":
        X = np.fft.rfft(xw, N2)
        R = np.fft.rfft(rwz, N2)
        G = X * np.conj(R)
        cc = np.fft.irfft(G / (np.abs(G) + 1e-12), N2)
        Lm1 = len(rwz) - 1
        cc_full = np.concatenate([cc[N2 - Lm1:N2], cc[:N]])
        seg, mo = cc_full, Lm1
    else:
        raise ValueError(method)
    i_lo = int(np.ceil(m_lo + mo))
    i_hi = int(np.floor(m_hi + mo))
    i_lo, i_hi = max(i_lo, 1), min(i_hi, len(seg) - 2)
    window = seg[i_lo:i_hi + 1]
    k_rel, dx = parabolic(window)
    i = i_lo + k_rel
    m = i - mo + dx
    d_hat = d0 - (m + n0) / sr
    j0 = int(round(-m))
    al = rw[j0:j0 + N]
    if len(al) < N:
        return d_hat, 0.0, 0.0
    aln = (al - al.mean()) * w
    val = float(np.dot(xw, aln) / (np.linalg.norm(xw) * np.linalg.norm(aln) + 1e-12))
    seg2 = window.copy()
    sep = int(0.015 * sr)
    lo2 = max(0, k_rel - sep)
    hi2 = min(len(seg2), k_rel + sep + 1)
    seg2[lo2:hi2] = -np.inf
    ratio = 0.0
    if np.isfinite(seg2).any():
        k2_rel = int(np.argmax(seg2))
        m2 = (i_lo + k2_rel) - mo
        j2 = int(round(-m2))
        al2 = rw[j2:j2 + N]
        if len(al2) == N:
            al2n = (al2 - al2.mean()) * w
            val2 = float(np.dot(xw, al2n) / (np.linalg.norm(xw) * np.linalg.norm(al2n) + 1e-12))
            ratio = val / (abs(val2) + 1e-12)
    return d_hat, val, ratio
