# S1E shared stage: local alignment refinement + A2-style STFT per-bin ridge FIR
# cancellation of the known pristine music (adapted from r1_golden_sample/scripts/
# 05_fine_align.py + 06_cancel.py; r1 files remain untouched).
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import signal as sig

from s1e_common import WORK_DIR, SR, load_wav

R1_COARSE_OFFSET_S = 11.331337868480725   # t_ref = t_video - offset (r1 hybrid)
REF_DELAY_S = -11.374728177828736         # r1 fine-stage d*: t_ref = t_video + d*
GLOBAL_DELAY_S = REF_DELAY_S              # alias kept for readability


def bp(x, lo, hi, sr=SR):
    sos = sig.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return sig.sosfiltfilt(sos, x, axis=0)


def next_pow2(n):
    return 1 << (n - 1).bit_length()


def ncc_delay_grid(y_mono, ref_mono, sr, band, grid_ms=1.0, span_ms=25.0):
    """Local delay refinement vs the REF_DELAY_S prediction: minimize cancellation
    residual energy E(k)=Syy - cc[k]^2/rr[k] (r1 fine-align objective, linear corr).
    True lag k* = d*·SR is negative (ref index = y index + d*·SR). Returns
    (delta_ms rel. to REF_DELAY_S, normalized ncc at optimum)."""
    yb = bp(y_mono, band[0], band[1], sr)
    rb = bp(ref_mono, band[0], band[1], sr)
    n2 = next_pow2(len(yb) + len(rb))
    Y = np.fft.rfft(yb, n2)
    R = np.fft.rfft(rb, n2)
    cc = np.fft.irfft(np.conj(Y) * R, n2)          # cc[k] = sum y[m] r[m+k], circular
    Wn = np.zeros(n2); Wn[: len(yb)] = 1.0
    Wf = np.fft.rfft(Wn)
    rr = np.fft.irfft(np.conj(Wf) * (R * np.conj(R)), n2)  # rr[k] = sum r[m+k]^2 over support
    Syy = float(np.dot(yb, yb))
    k0 = int(round(REF_DELAY_S * sr))
    best = (np.inf, 0.0, 0.0)
    for o_ms in np.arange(-span_ms, span_ms + grid_ms / 2, grid_ms):
        k = (k0 + int(round(o_ms / 1000.0 * sr))) % n2
        rrw = rr[k] + 1e-30
        E = Syy - cc[k] * cc[k] / rrw
        ncc = cc[k] / np.sqrt(max(Syy * rrw, 1e-30))
        if E < best[0]:
            best = (E, float(o_ms / 1000.0), float(ncc))
    return best[1] * 1000.0, best[2]


def warp_ref(ref_native, t_ref0_samples, n_out, a=1.0, taps=16, chunk=200_000):
    """Resample ref for handcam samples [0..n_out): ref index = (a*(n/SR))*SR + t_ref0_samples."""
    out = np.zeros((n_out, ref_native.shape[1]))
    valid = np.zeros(n_out, bool)
    n = np.arange(n_out)
    pos = a * n + t_ref0_samples
    ks = np.arange(-taps, taps + 1)
    hwin = np.hanning(2 * taps + 1)
    for c0 in range(0, n_out, chunk):
        c1 = min(c0 + chunk, n_out)
        p = pos[c0:c1]
        base = np.floor(p).astype(np.int64)
        idx = base[:, None] + ks[None, :]
        d = p[:, None] - idx
        w = np.sinc(d) * hwin[None, :]
        inb = (idx >= 0) & (idx < len(ref_native))
        w *= inb
        out[c0:c1] = np.einsum("mk,mkc->mc", w, ref_native[np.clip(idx, 0, len(ref_native) - 1)])
        valid[c0:c1] = inb.all(axis=1)
    return out, valid


def stft_frames(x, W, HOP):
    n = (len(x) - W) // HOP + 1
    win = sig.windows.hann(W, sym=False)
    idx = np.arange(W)[None, :] + HOP * np.arange(n)[:, None]
    return np.fft.rfft(x[idx] * win[None, :], axis=1), win, n


def istft_frames(S, W, HOP, n_frames, n_out):
    win = sig.windows.hann(W, sym=False)
    out = np.zeros(n_out)
    wsum = np.zeros(n_out)
    frames = np.fft.irfft(S, W, axis=1)
    w2 = win ** 2
    for t in range(n_frames):
        s = t * HOP
        out[s:s + W] += frames[t] * win
        wsum[s:s + W] += w2
    interior = wsum[W:n_out - W] if n_out > 2 * W else wsum
    cola = float(np.median(interior)) if len(interior) else 1.0
    return out / max(cola, 1e-12)


def fir_fit_apply(y, refs, W, HOP, eps_reg):
    """Per-bin multi-input ridge FIR (r1 06_cancel.py A2, verbatim logic)."""
    R, win, n_frames = stft_frames(refs[:, 0], W, HOP)
    R2, _, _ = stft_frames(refs[:, 1], W, HOP)
    RR = np.stack([R, R2], axis=-1)
    n_bins = RR.shape[1]
    A = np.einsum("tfb,tfd->fbd", RR.conj(), RR) / n_frames
    tr = A[:, 0, 0] + A[:, 1, 1]
    m_hat = np.zeros((len(refs), 2))
    H = np.zeros((n_bins, 2, 2), complex)
    Y, _, _ = stft_frames(y[:, 0], W, HOP)
    Y2, _, _ = stft_frames(y[:, 1], W, HOP)
    YY = np.stack([Y, Y2], axis=-1)
    B = np.einsum("tfb,tfd->fbd", RR.conj(), YY) / n_frames
    for b in range(n_bins):
        lam = max(eps_reg * tr[b] / 2, 1e-16)   # floor keeps silent ref bins non-singular
        M = A[b] + lam * np.eye(2)
        H[b] = np.linalg.solve(M, B[b])
    S = np.einsum("tfj,fjc->tfc", RR, H)
    for c in range(2):
        m_hat[:, c] = istft_frames(S[:, :, c], W, HOP, n_frames, len(refs))
    m_hat[:W] = 0.0
    return m_hat, H


def cancel_clip(y_seg, ref_native, t0_video_s, a=1.0, W=12288, HOP=3072, eps_reg=1e-4,
                refine_band=(40, 300), span_ms=25.0):
    """Align + FIR-cancel pristine music for handcam segment starting t0_video_s.
    y_seg: (n,2) handcam. Returns dict with residual, m_hat, ref_warp, delay info.
    Fit margins: the FIR is fitted on the WHOLE provided segment incl. margins; caller
    should pass >=1 s margins on each side and trim afterwards (m_hat[:W] is zeroed)."""
    n = len(y_seg)
    # local delay refinement on the segment (mono, 40-300 Hz)
    d_ms, ncc = ncc_delay_grid(np.mean(y_seg, axis=1), np.mean(ref_native, axis=1), SR,
                               refine_band, grid_ms=1.0, span_ms=span_ms)
    # handcam sample k (video time t0 + k/SR) -> ref index (t0 + d* + delta)*SR + k
    pred_ref_index0 = (t0_video_s + REF_DELAY_S + d_ms / 1000.0) * SR
    rwarp, valid = warp_ref(ref_native, pred_ref_index0, n, a=a)
    m_hat, H = fir_fit_apply(y_seg, rwarp, W, HOP, eps_reg)
    resid = y_seg - m_hat
    return {
        "residual": resid, "m_hat": m_hat, "ref_warp": rwarp,
        "local_delay_ms": d_ms, "local_ncc": ncc, "valid_ref": valid,
        "eps_reg": eps_reg, "W": W, "HOP": HOP,
    }
