# Critical audit of A0-A3: music-only ERLE per band, click preservation, removed-signal
# click leakage, per-band gain analysis, FIR response, train/test path check, AGC evidence.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import librosa
from scipy import signal as sig

from r1_common import OUT_DIR, WORK_DIR, load_wav, load_json, save_json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 48000
GOLDEN_S0, GOLDEN_LEN = 3.0, 15.0


def bp(x, lo, hi):
    sos = sig.butter(4, [lo, hi], btype="bandpass", fs=SR, output="sos")
    return sig.sosfiltfilt(sos, x, axis=0)


def ssum(x):
    return float(np.sum(np.asarray(x) ** 2))


def main():
    ctx, _ = load_wav(os.path.join(WORK_DIR, "golden_ctx.wav"))
    r_fix, _ = load_wav(os.path.join(WORK_DIR, "ref_warp_fixed.wav"))
    s0, s1 = int(GOLDEN_S0 * SR), int((GOLDEN_S0 + GOLDEN_LEN) * SR)
    y = ctx[s0:s1]
    versions = {}
    for tag in ("A1", "A2", "A3"):
        e, _ = load_wav(os.path.join(OUT_DIR, f"{tag}_residual.wav"))
        rem, _ = load_wav(os.path.join(OUT_DIR, f"{tag}_removed.wav"))
        versions[tag] = {"e": e, "rem": rem}
    ym = (y[:, 0] + y[:, 1]) * 0.5
    rm = (r_fix[s0:s1, 0] + r_fix[s0:s1, 1]) * 0.5

    # ---------- events / masks (same detector family as selection) ----------
    N, HOP = 1024, 256
    FR = SR / HOP
    Sdb = librosa.amplitude_to_db(np.abs(librosa.stft(ym.astype(np.float32), n_fft=N, hop_length=HOP)) + 1e-10)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N)
    dS = np.maximum(0.0, np.diff(Sdb, axis=1))
    fh = dS[(freqs >= 2000) & (freqs < 12000)].sum(axis=0)
    fl = dS[(freqs >= 40) & (freqs < 300)].sum(axis=0)
    on_ref = librosa.onset.onset_strength(y=rm.astype(np.float32), sr=SR, hop_length=HOP)
    ref_pk = librosa.onset.onset_detect(onset_envelope=on_ref, sr=SR, hop_length=HOP, units="time", backtrack=False)

    def pick(f, k=4.0):
        med = np.median(f)
        mad = np.median(np.abs(f - med)) + 1e-9
        pk, _ = sig.find_peaks(f, height=med + k * mad, distance=max(1, int(0.06 * FR)))
        return pk

    t_off = N / (2 * SR)
    pk_h = pick(fh)
    t_h = (pk_h + 1) / FR + t_off
    dh = np.array([np.min(np.abs(t - ref_pk)) if len(ref_pk) else 9e9 for t in t_h])
    clicks = t_h[dh > 0.06]
    pk_l = pick(fl)
    t_l = (pk_l + 1) / FR + t_off
    dl = np.array([np.min(np.abs(t - ref_pk)) if len(ref_pk) else 9e9 for t in t_l])
    taiko_ev = t_l[dl > 0.06]
    print(f"golden-window events: clicks={len(clicks)} taiko-ish={len(taiko_ev)}")

    # sub-second music-only pool: exclude +-0.15s before / +0.25s after every event
    pool = np.ones(len(ym), bool)
    for t in np.concatenate([clicks, taiko_ev]):
        a = int(max(0, (t - 0.15) * SR))
        b = int(min(len(ym), (t + 0.25) * SR))
        pool[a:b] = False
    # require local music presence (ref energy) so quiet ambience doesn't dominate
    ref_energy = np.convolve(rm ** 2, np.ones(SR) / SR, mode="same")
    pool &= ref_energy > np.percentile(ref_energy, 20)
    music_only = pool
    n_sec = int(GOLDEN_LEN)
    print(f"music-only pool: {music_only.sum() / SR:.2f} s of {n_sec} s")

    out = {"golden_window": [float(GOLDEN_S0), float(GOLDEN_S0 + GOLDEN_LEN)],
           "events": {"clicks": [float(t) for t in clicks],
                      "taiko": [float(t) for t in taiko_ev][:80]},
           "music_only_pool_seconds": float(music_only.sum() / SR),
           "levels_db": {"y": 10 * np.log10(ssum(y)), "ref_warp": 10 * np.log10(ssum(r_fix[s0:s1]))}}

    # ---------- ERLE per version, per band, overall + music-only ----------
    bands = {"full": None, "bass40-300": (40, 300), "mid300-2k": (300, 2000), "hi2k-12k": (2000, 12000)}
    for tag, v in versions.items():
        m = {}
        for bname, b in bands.items():
            yy = ym if b is None else bp(ym, *b)
            ee = v["e"].mean(axis=1)
            ee = ee if b is None else bp(ee, *b)
            m[f"erle_{bname}_db"] = 10 * np.log10(max(ssum(yy), 1e-12) / max(ssum(ee), 1e-12))
            idx = music_only
            m[f"erle_musiconly_{bname}_db"] = 10 * np.log10(max(ssum(yy[idx]), 1e-12) / max(ssum(ee[idx]), 1e-12))
        v["metrics"] = m
        print(f"{tag}: " + "  ".join(f"{k}={val:5.2f}" for k, val in m.items()))

    # ---------- click preservation & removed leakage ----------
    ev_stats = {}
    for tag, v in versions.items():
        rows = []
        for t in clicks:
            w0, w1 = int(max(0, (t - 0.1) * SR)), int(min(GOLDEN_LEN * SR, (t + 0.1) * SR))
            if w1 - w0 < int(0.05 * SR):
                continue
            yseg = y[w0:w1].mean(axis=1)
            eseg = v["e"][w0:w1].mean(axis=1)
            rseg = v["rem"][w0:w1].mean(axis=1)
            rows.append({"t": float(t),
                         "pres_ratio": float(np.sqrt(ssum(eseg) / max(ssum(yseg), 1e-12))),
                         "removed_ratio": float(np.sqrt(ssum(rseg) / max(ssum(yseg), 1e-12))),
                         "spec_cos": float(np.dot(yseg, eseg) / (np.linalg.norm(yseg) * np.linalg.norm(eseg) + 1e-12))})
        pr = [r["pres_ratio"] for r in rows]
        rr = [r["removed_ratio"] for r in rows]
        ev_stats[tag] = {"n": len(rows),
                         "pres_ratio_median": float(np.median(pr)) if pr else None,
                         "pres_ratio_p10": float(np.percentile(pr, 10)) if pr else None,
                         "removed_ratio_median": float(np.median(rr)) if rr else None,
                         "removed_ratio_p90": float(np.percentile(rr, 90)) if rr else None,
                         "rows": rows}
        print(f"{tag} clicks: n={len(rows)} pres_median={np.median(pr):.3f} (p10 {np.percentile(pr,10):.3f}) "
              f"removed_median={np.median(rr):.3f} (p90 {np.percentile(rr,90):.3f})")
    out["click_preservation"] = ev_stats

    # ---------- per-band single-gain analysis (why A1 fails) ----------
    per_band = []
    for lo, hi in [(40, 300), (300, 1000), (1000, 4000), (4000, 12000)]:
        yb = bp(ym, lo, hi)
        rb = bp(rm, lo, hi)
        g = float(np.dot(yb, rb) / (np.dot(rb, rb) + 1e-12))
        e = yb - g * rb
        red = 10 * np.log10(max(ssum(yb), 1e-12) / max(ssum(e), 1e-12))
        per_band.append({"band": f"{lo}-{hi}", "gain": g, "reduction_db": red})
        print(f"band {lo:5d}-{hi:5d}: single gain {g:+.4f} -> reduction {red:5.2f} dB")
    out["a1_per_band_gain"] = per_band

    # ---------- FIR impulse responses ----------
    H = np.load(os.path.join(WORK_DIR, "fir_H_A2.npy"))
    h = np.fft.irfft(H, 12288, axis=0)
    fig, axs = plt.subplots(2, 2, figsize=(14, 7))
    for jj in range(2):
        for cc in range(2):
            axs[jj, cc].plot(np.arange(12288) / SR * 1000, h[:, jj, cc], lw=0.6)
            axs[jj, cc].set_title(f"h ref{jj} -> mic{cc}")
            axs[jj, cc].set_xlabel("ms")
    fig.suptitle("A2 FIR impulse responses (256 ms)")
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "fir_response_A2.png"), dpi=110)
    # peak delay of each response (coarse path delay sanity)
    pk_delays = {}
    for jj in range(2):
        for cc in range(2):
            pk_delays[f"ref{jj}->mic{cc}"] = float(np.argmax(np.abs(h[:, jj, cc])) / SR * 1000)
    out["fir_peak_delays_ms"] = pk_delays
    print("FIR peak delays (ms):", pk_delays)

    # ---------- train/test path check ----------
    import importlib.util
    spec6 = importlib.util.spec_from_file_location("c06", os.path.join(os.path.dirname(__file__), "06_cancel.py"))
    c06 = importlib.util.module_from_spec(spec6)
    spec6.loader.exec_module(c06)
    n = len(ctx)
    half = n // 2
    W, HOPC = 12288, 3072
    m_hat_tr, H_tr = c06.fir_fit_apply(ctx[:half], r_fix[:half], W, HOPC, 1e-2)
    e_tr = ctx[:half] - m_hat_tr
    red_train = 10 * np.log10(max(ssum(ctx[W:half, 0]) + ssum(ctx[W:half, 1]), 1e-12) / max(ssum(e_tr[W:]), 1e-12))
    ref_part = r_fix[half - W:]
    R, win, nf = c06.stft_frames(ref_part[:, 0], W, HOPC)
    R2, _, _ = c06.stft_frames(ref_part[:, 1], W, HOPC)
    RR = np.stack([R, R2], axis=-1)
    S = np.einsum("tfj,fjc->tfc", RR, H_tr)
    m_te = c06.istft_frames(S[:, :, 0], W, HOPC, nf, len(ref_part))
    m_te2 = c06.istft_frames(S[:, :, 1], W, HOPC, nf, len(ref_part))
    eval_len = min(n - half, len(m_te) - W)
    yy_te = ctx[half:half + eval_len]
    mm_te = 0.5 * (m_te[W:W + eval_len] + m_te2[W:W + eval_len])
    ee_te = 0.5 * (yy_te[:, 0] + yy_te[:, 1]) - mm_te
    red_test = 10 * np.log10(max(ssum(0.5 * (yy_te[:, 0] + yy_te[:, 1])), 1e-12) / max(ssum(ee_te), 1e-12))
    out["path_check"] = {"train_reduction_db": red_train, "test_reduction_db": red_test}
    print(f"path check: train {red_train:.2f} dB vs test {red_test:.2f} dB")

    # ---------- AGC evidence ----------
    yb = bp(ym, 40, 300)
    rb = bp(rm, 40, 300)
    gs, rls = [], []
    for t0 in range(0, len(yb) - SR, SR // 2):
        yy = yb[t0:t0 + SR]
        rr2 = rb[t0:t0 + SR]
        if np.sum(rr2 ** 2) < 1e-9:
            continue
        gs.append(float(np.dot(yy, rr2) / np.dot(rr2, rr2)))
        rls.append(10 * np.log10(ssum(rr2) / SR))
    gs, rls = np.array(gs), np.array(rls)
    corr_g_loud = float(np.corrcoef(gs, rls)[0, 1])
    out["agc_check"] = {"local_gain_vs_ref_loudness_corr": corr_g_loud,
                        "gain_p25_p75": [float(np.percentile(gs, 25)), float(np.percentile(gs, 75))]}
    print(f"AGC check: corr(local gain, ref loudness) = {corr_g_loud:.2f}; gain IQR [{np.percentile(gs, 25):.4f}, {np.percentile(gs, 75):.4f}]")

    # ---------- spectrogram figure ----------
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for ax, (title, s1_) in zip(axs, [("handcam (golden)", ym),
                                      ("A2 residual", versions["A2"]["e"].mean(axis=1)),
                                      ("A2 removed", versions["A2"]["rem"].mean(axis=1))]):
        Sxx = librosa.amplitude_to_db(np.abs(librosa.stft(s1_.astype(np.float32), n_fft=1024, hop_length=256)) + 1e-10)
        ax.imshow(Sxx, origin="lower", aspect="auto", extent=(0, GOLDEN_LEN, 0, SR / 1000), cmap="magma", vmin=-95, vmax=-15)
        ax.set_ylim(0, 20)
        ax.set_ylabel("kHz")
        ax.set_title(title)
    for t in clicks[:60]:
        for ax in axs:
            ax.axvline(t, color="c", lw=0.4, alpha=0.5)
    axs[2].set_xlabel("golden window time [s]")
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "audit_spectrograms_A2.png"), dpi=110)

    save_json(os.path.join(WORK_DIR, "audit_metrics.json"), out)
    print("audit_metrics.json + plots saved")


if __name__ == "__main__":
    main()
