# Phase 3 - fine alignment on the golden context (15 s + 3 s margins).
# Alignment is judged by the CANCELLATION OBJECTIVE, not correlation aesthetics:
#   E(d)   = weighted residual energy after optimal LS gain, vs delay d (from FFT corr)
#   E(eps) = same vs drift rate eps (explicit warps)
# Stage A: brute-force anchor (1 ms grid, 4 s windows, band 40-300 Hz).
# Stage B: objective static-delay refinement around the anchor (full 40-800 band).
# Stage C: drift-rate sweep (explicit affine warps, 25 ppm steps).
# Stage D: NCC/GCC fine-window measurements as supporting diagnostics.
# Outputs: ref_warp_fixed.wav (a=1, d*) and ref_warp_affine.wav (a=1+eps*, d*),
#          alignment_diagnostics.json, work/fine_alignment.png.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import signal as sig

from align_lib import estimate_delay, next_pow2
from r1_common import OUT_DIR, WORK_DIR, load_wav, load_json, save_json, save_wav

SR = 48000
MARGIN_S = 0.6
EST_BAND = (40, 800)
NCC_BAND = (40, 300)
GCC_BAND = (150, 8000)
WIN_S, HOP_S = 3.0, 0.5


def bp(x, lo, hi):
    sos = sig.butter(4, [lo, hi], btype="bandpass", fs=SR, output="sos")
    return sig.sosfiltfilt(sos, x, axis=0)


def sinc_warp(ref, a, b, n_out, ref_index0, taps=16, chunk=200_000):
    """handcam ctx sample n -> ref-native index (a*(n/SR) + b)*SR - ref_index0."""
    out = np.zeros((n_out, 2))
    valid = np.zeros(n_out, bool)
    n = np.arange(n_out)
    pos = (a * (n / SR) + b) * SR - ref_index0
    ks = np.arange(-taps, taps + 1)
    hwin = np.hanning(2 * taps + 1)
    for c0 in range(0, n_out, chunk):
        c1 = min(c0 + chunk, n_out)
        p = pos[c0:c1]
        base = np.floor(p).astype(np.int64)
        idx = base[:, None] + ks[None, :]
        d = p[:, None] - idx
        w = np.sinc(d) * hwin[None, :]
        inb = (idx >= 0) & (idx < len(ref))
        w *= inb
        out[c0:c1] = np.einsum("mk,mkc->mc", w, ref[np.clip(idx, 0, len(ref) - 1)])
        valid[c0:c1] = inb.all(axis=1)
    return out, valid


def residual_energy_curve(y_w, refb, w_win, d_grid, ctx_start):
    """E(d) = Syy - cc_w(d)^2 / rr_w(d). d = t_ref - t_hand; r~_d[n] = ref[(ctx_start+n/SR+d)*SR]."""
    N = len(y_w)
    n2 = next_pow2(len(refb) + N)
    Y = np.fft.rfft(y_w, n2)
    R = np.fft.rfft(refb, n2)
    c = np.fft.irfft(R * np.conj(Y), n2)      # c[k] ~ sum refb[n+k] y_w[n]
    R2 = np.fft.rfft(refb ** 2, n2)
    W = np.fft.rfft(w_win, n2)
    rr = np.fft.irfft(R2 * np.conj(W), n2)    # rr[k] ~ sum refb2[n+k] w[n]
    Syy = float(np.dot(y_w, y_w))
    Es = []
    for d in d_grid:
        m = int(round(-(ctx_start + d) * SR))
        k = m % n2
        cc = c[k]
        rrw = rr[k] + 1e-12
        Es.append(Syy - cc * cc / rrw)
    return np.array(Es)


def edge_weights(n, sr, fade_s=0.5):
    w = np.ones(n)
    nf = int(fade_s * sr)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, nf)))
    w[:nf] = ramp
    w[-nf:] = ramp[::-1]
    return w


def main():
    ctx, sr = load_wav(os.path.join(WORK_DIR, "golden_ctx.wav"))
    assert sr == SR
    r_full, _ = load_wav(os.path.join(WORK_DIR, "ref_native.wav"))
    coarse = load_json(os.path.join(WORK_DIR, "coarse_offset.json"))["hybrid"]["offset"]
    sel = load_json(os.path.join(WORK_DIR, "golden_selection.json"))
    g_start, g_dur = sel["start"], sel["duration"]
    ctx_start = g_start - 3.0
    ctx_dur = g_dur + 6.0
    pred_d = -coarse

    r0 = int(round((ctx_start + pred_d - MARGIN_S) * SR))
    r1 = int(round((ctx_start + ctx_dur + pred_d + MARGIN_S) * SR))
    assert 0 <= r0 < r1 <= len(r_full)
    r_seg = r_full[r0:r1]
    print(f"ref slice: ref-native [{r0/SR:.3f} .. {r1/SR:.3f}]s len={len(r_seg)/SR:.3f}s (coarse {coarse:.4f}s baked in)")

    xs = bp((ctx[:, 0] + ctx[:, 1]) * 0.5, *EST_BAND)
    xg = bp((ctx[:, 0] + ctx[:, 1]) * 0.5, *GCC_BAND)
    refb_full = bp((r_full[:, 0] + r_full[:, 1]) * 0.5, *EST_BAND)

    # ---- Stage A: brute-force anchor ----
    anchor_windows = [g_start - 1.0, g_start + 4.0, g_start + 9.0, g_start + 12.0]
    rs_est = refb_full
    grid = np.arange(pred_d - 0.3, pred_d + 0.3, 0.001)
    N_A = int(4 * SR)
    anchor_rows = []
    for t0 in anchor_windows:
        c0 = int(round((t0 - ctx_start) * SR))
        x = xs[c0:c0 + N_A]
        xn = x - x.mean()
        nx = np.linalg.norm(xn)
        vals = np.zeros(len(grid))
        for i, d in enumerate(grid):
            a = int(round((t0 + d) * SR))
            al = rs_est[a:a + N_A]
            if len(al) < N_A:
                continue
            aln = al - al.mean()
            vals[i] = np.dot(xn, aln) / (nx * np.linalg.norm(aln) + 1e-12)
        k = int(np.argmax(vals))
        if 0 < k < len(grid) - 1:
            y0, y1, y2 = vals[k - 1], vals[k], vals[k + 1]
            den = y0 - 2 * y1 + y2
            dx = float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if abs(den) > 1e-15 else 0.0
        else:
            dx = 0.0
        d_hat = grid[k] + dx * 0.001
        anchor_rows.append({"t": t0 + N_A / 2 / SR, "d": d_hat, "ncc": float(vals[k])})
        print(f"anchor window t0={t0:6.1f}s: d={d_hat:+.5f}s ncc={vals[k]:.3f}")
    aw = np.array([r["ncc"] for r in anchor_rows])
    ad = np.array([r["d"] for r in anchor_rows])
    anchor = float(np.sum(ad * aw) / np.sum(aw))
    print(f"anchor d_ref = {anchor*1e3:+.3f} ms")

    # ---- Stage B: objective static-delay refinement (explicit brute force) ----
    w_win = edge_weights(len(xs), SR, fade_s=0.5)
    n_ctx = len(ctx)
    yb = xs  # band-filtered ctx mono
    yb_w = yb * w_win
    r_seg_band = bp((r_seg[:, 0] + r_seg[:, 1]) * 0.5, *EST_BAND)
    d_grid = np.arange(anchor - 0.03, anchor + 0.0301, 0.0005)
    Es, Nccs = [], []
    for d in d_grid:
        rw, _ = sinc_warp(r_seg_band[:, None], 1.0, ctx_start + d, n_ctx, r0)
        rwm = rw[:, 0] * w_win
        g = float(np.dot(yb_w, rwm) / (np.dot(rwm, rwm) + 1e-12))
        e = yb_w - g * rwm
        nb = int(0.5 * SR)
        m = (len(e) // nb) * nb
        be = np.sum(e[:m].reshape(-1, nb) ** 2, axis=1)
        Es.append(float(np.median(be)))
        Nccs.append(float(np.dot(yb_w, rwm) / (np.linalg.norm(yb_w) * np.linalg.norm(rwm) + 1e-12)))
    Es = np.array(Es); Nccs = np.array(Nccs)
    # The energy objective E(d) is sign-blind: an anti-correlated lobe reduces energy
    # too. Choose the POSITIVE-correlation peak, refine parabolically on the ncc curve.
    kn = int(np.argmax(Nccs))
    if Nccs[kn] < 0.1:
        raise SystemExit(f"no positive correlation peak found (max ncc {Nccs[kn]:.3f})")
    if 0 < kn < len(Nccs) - 1:
        y0, y1, y2 = Nccs[kn - 1], Nccs[kn], Nccs[kn + 1]
        den = y0 - 2 * y1 + y2
        dxn = float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if abs(den) > 1e-15 else 0.0
    else:
        dxn = 0.0
    d_star = float(d_grid[kn] + dxn * 0.0005)
    e_at = lambda d: float(np.interp(d, d_grid, Es))
    ncc_at = lambda d: float(np.interp(d, d_grid, Nccs))
    e_edge = float(max(Es[0], Es[-1], 1e-12))
    gain_db = 10 * np.log10(e_edge / max(e_at(d_star), 1e-12))
    print(f"static-delay selection: d* = {d_star*1e3:+.4f} ms (POSITIVE ncc peak {ncc_at(d_star):.3f}; "
          f"E(d*) vs edge {gain_db:.2f} dB; anti-lobe check: ncc min {Nccs.min():.3f} at {d_grid[int(np.argmin(Nccs))]*1e3:+.3f} ms)")

    # ---- Stage C: drift-rate sweep. Two metrics:
    #   (a) sign-blind block-median E (can be fooled by anti-phase lobes)
    #   (b) sign-aware: median POSITIVE ncc of sliding windows per rate (primary)
    rates = np.arange(-100, 100.1, 25) * 1e-6
    N_CC = int(3 * SR)
    win_n = sig.windows.hann(N_CC, sym=False)
    xs_n = bp((ctx[:, 0] + ctx[:, 1]) * 0.5, *EST_BAND)
    med_ncc, E_eps = [], []
    for eps in rates:
        r_w, _ = sinc_warp(r_seg_band[:, None], 1.0 + eps, (1.0 + eps) * ctx_start + (d_star - eps * g_start),
                           n_ctx, r0)
        rwm = r_w[:, 0] * w_win
        g = float(np.dot(yb_w, rwm) / (np.dot(rwm, rwm) + 1e-12))
        e = yb_w - g * rwm
        nb = int(0.5 * SR)
        mlen = (len(e) // nb) * nb
        be = np.sum(e[:mlen].reshape(-1, nb) ** 2, axis=1)
        E_eps.append(float(np.median(be)))
        # sign-aware: positive ncc of 3 s windows (+-10 ms search)
        vals = []
        rwm_f = r_w[:, 0]
        n0s = int(0.01 * SR)
        for c0 in range(int(1.0 * SR), n_ctx - N_CC - int(1.0 * SR), int(1.0 * SR)):
            xw = xs_n[c0:c0 + N_CC]
            rw2 = rwm_f[c0 - n0s:c0 + N_CC + n0s]
            xn = xw - xw.mean()
            rn = rw2 - rw2.mean()
            c2 = sig.correlate(xn, rn, mode="full", method="fft")
            k = int(np.argmax(c2))
            a0, a1, a2 = c2[k - 1], c2[k], c2[k + 1]
            den = a0 - 2 * a1 + a2
            dx2 = 0.5 * (a0 - a2) / den if abs(den) > 1e-15 else 0.0
            j = int(round(-(k - (len(rn) - 1) + dx2)))
            al = rw2[j:j + N_CC]
            if len(al) == N_CC:
                aln = al - al.mean()
                vals.append(float(np.dot(xn, aln) / (np.linalg.norm(xn) * np.linalg.norm(aln) + 1e-12)))
        med_ncc.append(float(np.median(vals)))
    E_eps = np.array(E_eps); med_ncc = np.array(med_ncc)
    kc = int(np.argmax(med_ncc))
    eps_star = float(rates[kc])
    improvement_db = 10 * np.log10(max(E_eps[len(rates) // 2], 1e-12) / max(E_eps[int(np.argmin(E_eps))], 1e-12))
    print("drift sweep (sign-aware median positive ncc | sign-blind block-median E):")
    for eps, mn, E in zip(rates, med_ncc, E_eps):
        print(f"    {eps*1e6:+5.0f} ppm  ncc_med={mn:.4f}  E/E(0)={E / E_eps[len(rates)//2]:.4f}")
    print(f"drift: eps* = {eps_star*1e6:+.0f} ppm by median positive ncc "
          f"(improvement vs 0 ppm: {med_ncc[kc] - med_ncc[len(rates)//2]:+.4f} ncc; "
          f"sign-blind E range {(E_eps.max()/E_eps.min()-1)*100:.2f}%)")
    for eps, E in zip(rates, E_eps):
        print(f"    {eps*1e6:+5.0f} ppm -> E/E0 = {E / E_eps[0]:.5f}")

    # ---- Stage D: NCC/GCC fine windows (supporting diagnostics) ----
    sos_n = sig.butter(4, list(NCC_BAND), btype="bandpass", fs=SR, output="sos")
    xs_n = sig.sosfiltfilt(sos_n, (ctx[:, 0] + ctx[:, 1]) * 0.5)
    r0n = int(round((ctx_start + pred_d - 1.0) * SR))
    r1n = int(round((ctx_start + ctx_dur + pred_d + 1.0) * SR))
    rs_n = sig.sosfiltfilt(sos_n, (r_full[r0n:r1n, 0] + r_full[r0n:r1n, 1]) * 0.5)
    off_n = r0n
    d_lo, d_hi = anchor - 0.02, anchor + 0.02
    d_axis = np.array([d_lo, d_hi])
    n0 = int(0.3 * SR)
    NN = int(WIN_S * SR)
    rows = []
    for c0 in range(0, int(len(ctx) - NN) + 1, int(HOP_S * SR)):
        t_start = ctx_start + c0 / SR
        t_abs = ctx_start + (c0 + NN / 2) / SR
        x = xs_n[c0:c0 + NN]
        if len(x) < NN:
            continue
        st = int(round((t_start + d_lo) * SR)) - n0 - off_n
        en = int(round((t_start + d_hi) * SR)) + NN + n0
        if st < 0 or en > len(rs_n):
            continue
        d_n, v_n, r_n = estimate_delay(x, rs_n[st:en], SR, n0, d_axis, "ncc", win_hann=False)
        rows.append({"t": t_abs, "d_ncc": d_n, "ncc_peak": v_n, "ncc_ratio": r_n})
    print(f"supporting NCC windows: {len(rows)}, median peak {np.median([r['ncc_peak'] for r in rows]):.3f}")

    # ---- final warps ----
    r_fix, v_fix = sinc_warp(r_seg, 1.0, ctx_start + d_star, n_ctx, r0)
    r_aff, v_aff = sinc_warp(r_seg, 1.0 + eps_star, (1.0 + eps_star) * ctx_start + (d_star - eps_star * g_start),
                             n_ctx, r0)
    vf, va = int(0.05 * SR), n_ctx - int(0.05 * SR)
    print(f"warps: fixed valid={bool(v_fix[vf:va].all())} affine valid={bool(v_aff[vf:va].all())}")
    save_wav(os.path.join(WORK_DIR, "ref_warp_fixed.wav"), r_fix.astype(np.float32), SR)
    save_wav(os.path.join(WORK_DIR, "ref_warp_affine.wav"), r_aff.astype(np.float32), SR)

    diag = {
        "coarse_offset_s": coarse,
        "convention": "d(t) = t_ref - t_hand; t_ref = a*t_hand + b; fixed warp: a=1, b=ctx_start+d*; affine warp: a=1+eps*, b=(1+eps*)*ctx_start + d* - eps**g_start",
        "anchor_stage": {
            "method": "brute-force 1 ms grid, 4 s windows, band 40-300 Hz",
            "windows": anchor_rows,
            "d_ref_s": anchor,
        },
        "objective": {
            "static_delay": {
                "band": EST_BAND, "window_s": WIN_S, "weighting": "hann",
                "d_star_s": d_star,
                "ncc_at_d_star": ncc_at(d_star),
                "E_improvement_vs_edge_db": gain_db,
                "anti_lobe_ncc_min": float(Nccs.min()),
            },
            "drift_sweep": {
                "rates_ppm": (rates * 1e6).tolist(),
                "median_positive_ncc": med_ncc.tolist(),
                "E_ratio_vs_zero": (E_eps / E_eps[len(rates)//2]).tolist(),
                "eps_star_ppm": eps_star * 1e6,
                "ncc_improvement_vs_zero": float(med_ncc[kc] - med_ncc[len(rates)//2]),
                "improvement_db_signblind": improvement_db,
                "note": "improvement < ~0.3 dB => drift not a meaningful bottleneck here",
            },
        },
        "fine": {
            "windows_total": len(rows),
            "ncc_peak_median": float(np.median([r["ncc_peak"] for r in rows])) if rows else None,
        },
        "local_delay_measurements": rows,
    }
    save_json(os.path.join(OUT_DIR, "alignment_diagnostics.json"), diag)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].plot(d_grid * 1e3, Es / Es[0], ".-", ms=3)
    ax[0].axvline(d_star * 1e3, color="r", lw=1, label=f"d* = {d_star*1e3:.3f} ms")
    ax[0].set_xlabel("delay d [ms]"); ax[0].set_ylabel("E/E(search edge)")
    ax[0].set_title("objective static-delay sweep (band 40-800 Hz)")
    ax[0].legend()
    ax[1].plot(rates * 1e6, E_eps / E_eps[0], ".-", ms=4)
    ax[1].axvline(eps_star * 1e6, color="r", lw=1, label=f"eps* = {eps_star*1e6:+.0f} ppm")
    ax[1].set_xlabel("drift rate [ppm]"); ax[1].set_ylabel("E/E(eps=0)")
    ax[1].set_title("drift-rate sweep (band 40-800 Hz)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "fine_alignment.png"), dpi=110)
    print("alignment_diagnostics.json + plot saved")


if __name__ == "__main__":
    main()
