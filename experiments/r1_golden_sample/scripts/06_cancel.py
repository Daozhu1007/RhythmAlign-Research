# Phase 4 - A0..A3 reference-music cancellation on the golden context.
# All versions use the SAME golden sample and fixed gains; float WAV out, no per-version
# normalization, no compressor/limiter.
#   A0: 0.5*y + 0.5*r_warp                 (current-pipeline style mix)
#   A1: m_hat = per-channel LS gain on warped stereo ref; residual = y - m_hat
#   A2: m_hat = STFT per-bin ridge FIR (W=256 ms) fitted on r_warp_fixed
#   A3: same FIR on r_warp_affine (drift-compensated warp; eps*=0 here -> expect ~A2)
# final_mix = 0.5*residual + 0.5*warped pristine ref (same weights as A0).
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import signal as sig

from r1_common import OUT_DIR, WORK_DIR, load_wav, load_json, save_json, save_wav

SR = 48000
GOLDEN_S0 = 3.0          # ctx offset of golden window start
GOLDEN_LEN = 15.0
W = 12288                # 256 ms FIR window
HOP = W // 4


def edge_weights(n, sr, fade_s=0.5):
    w = np.ones(n)
    nf = int(fade_s * sr)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, nf)))
    w[:nf] = ramp
    w[-nf:] = ramp[::-1]
    return w


def ls_gains(y, r2, w):
    """Robust-ish global LS: y[:,c] ~ g0*r[:,0] + g1*r[:,1], weight w. Returns (2,C)."""
    C = y.shape[1]
    A = np.zeros((2, 2))
    B = np.zeros((2, C))
    for j in range(2):
        for k in range(2):
            A[j, k] = np.dot(w * r2[:, j], r2[:, k])
        for c in range(C):
            B[j, c] = np.dot(w * r2[:, j], y[:, c])
    A += 1e-12 * np.eye(2) * max(np.trace(A), 1e-30)
    return np.linalg.solve(A, B)


def stft_frames(x, W, HOP):
    n = (len(x) - W) // HOP + 1
    win = sig.windows.hann(W, sym=False)
    idx = np.arange(W)[None, :] + HOP * np.arange(n)[:, None]
    return np.fft.rfft(x[idx] * win[None, :], axis=1), win, n


def istft_frames(S, W, HOP, n_frames, n_out):
    # Constant-COLA synthesis: divide by the interior sum of w^2 (hann, hop W/4 -> 1.5).
    # Per-sample division would amplify the circular-wrap garbage at each frame head
    # (w^2 -> 0 there), which is exactly the failure mode this avoids.
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
    """Per-bin multi-input ridge FIR. refs: (n,2). Returns m_hat (n,2), H (bins,2)."""
    R, win, n_frames = stft_frames(refs[:, 0], W, HOP)
    R2, _, _ = stft_frames(refs[:, 1], W, HOP)
    RR = np.stack([R, R2], axis=-1)          # (frames, bins, 2)
    n_bins = RR.shape[1]
    A = np.einsum("tfb,tfd->fbd", RR.conj(), RR) / n_frames   # A[k,j] = sum R_k* R_j
    tr = A[:, 0, 0] + A[:, 1, 1]
    m_hat = np.zeros((len(refs), 2))
    H = np.zeros((n_bins, 2, 2), complex)
    Y, _, _ = stft_frames(y[:, 0], W, HOP)
    Y2, _, _ = stft_frames(y[:, 1], W, HOP)
    YY = np.stack([Y, Y2], axis=-1)          # (frames, bins, 2)
    B = np.einsum("tfb,tfd->fbd", RR.conj(), YY) / n_frames   # (bins,2,2)
    for b in range(n_bins):
        lam = eps_reg * tr[b] / 2
        M = A[b] + lam * np.eye(2)
        H[b] = np.linalg.solve(M, B[b])      # (2,2): H[j,c]
    # apply: spec_c(t) = sum_j H[j,c]*R_j(t)
    S = np.einsum("tfj,fjc->tfc", RR, H)   # (frames,bins,2)
    for c in range(2):
        m_hat[:, c] = istft_frames(S[:, :, c], W, HOP, n_frames, len(refs))
    # zero the unfiltered warm-up zone (first W samples lack full context)
    m_hat[:W] = 0.0
    return m_hat, H


def band_energy(x, lo, hi):
    sos = sig.butter(4, [lo, hi], btype="bandpass", fs=SR, output="sos")
    xb = sig.sosfiltfilt(sos, x, axis=0)
    return float(np.sum(xb ** 2))


def main():
    ctx, sr = load_wav(os.path.join(WORK_DIR, "golden_ctx.wav"))
    r_fix, _ = load_wav(os.path.join(WORK_DIR, "ref_warp_fixed.wav"))
    r_aff, _ = load_wav(os.path.join(WORK_DIR, "ref_warp_affine.wav"))
    assert sr == SR and len(ctx) == len(r_fix) == len(r_aff)
    n = len(ctx)
    s0, s1 = int(GOLDEN_S0 * SR), int((GOLDEN_S0 + GOLDEN_LEN) * SR)

    y = ctx
    r = r_fix
    print(f"ctx {n/SR:.1f}s; outputs cut to ctx[{s0/SR:.1f}..{s1/SR:.1f}]s (golden 15 s)")

    # ---------- A0 ----------
    a0 = 0.5 * y + 0.5 * r
    save_wav(os.path.join(OUT_DIR, "A0_current_mix.wav"), a0[s0:s1].astype(np.float32), SR)
    print("A0_current_mix.wav saved")

    # ---------- A1: delay + gain ----------
    wgt = edge_weights(n, SR)
    G = ls_gains(y * wgt[:, None], r * wgt[:, None], wgt)
    print(f"A1 gains (rows=refL,refR; cols=micL,micR):\n{G}")
    m_hat = r @ G
    e = y - m_hat
    save_wav(os.path.join(OUT_DIR, "A1_residual.wav"), e[s0:s1].astype(np.float32), SR)
    save_wav(os.path.join(OUT_DIR, "A1_removed.wav"), m_hat[s0:s1].astype(np.float32), SR)
    save_wav(os.path.join(OUT_DIR, "A1_final_mix.wav"),
             (0.5 * e + 0.5 * r)[s0:s1].astype(np.float32), SR)
    erle1 = 10 * np.log10(np.sum(y[s0:s1] ** 2) / max(np.sum(e[s0:s1] ** 2), 1e-12))
    print(f"A1 saved; full-band energy reduction over golden: {erle1:.2f} dB")

    # ---------- A2/A3: FIR ----------
    for tag, ref in (("A2", r_fix), ("A3", r_aff)):
        best = None
        for eps_reg in (1e-4, 1e-3, 1e-2):
            m_hat, H = fir_fit_apply(y, ref, W, HOP, eps_reg)
            e = y - m_hat
            # quick objective: full-band reduction on golden + removed-click leakage
            red = 10 * np.log10(np.sum(y[s0:s1] ** 2) / max(np.sum(e[s0:s1] ** 2), 1e-12))
            print(f"{tag} eps_reg={eps_reg:g}: full-band reduction {red:.2f} dB")
            if best is None or red > best[1]:
                best = (eps_reg, red, m_hat, e, H)
        eps_reg, red, m_hat, e, H = best
        save_wav(os.path.join(OUT_DIR, f"{tag}_residual.wav"), e[s0:s1].astype(np.float32), SR)
        save_wav(os.path.join(OUT_DIR, f"{tag}_removed.wav"), m_hat[s0:s1].astype(np.float32), SR)
        save_wav(os.path.join(OUT_DIR, f"{tag}_final_mix.wav"),
                 (0.5 * e + 0.5 * ref)[s0:s1].astype(np.float32), SR)
        np.save(os.path.join(WORK_DIR, f"fir_H_{tag}.npy"), H)
        print(f"{tag}: eps_reg={eps_reg:g} chosen, reduction {red:.2f} dB; outputs saved")

    print("phase 4 core outputs complete")


if __name__ == "__main__":
    main()
