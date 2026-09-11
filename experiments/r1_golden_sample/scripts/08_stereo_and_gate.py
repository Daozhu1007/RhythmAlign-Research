# Decision-gate diagnostics + stereo quick check.
# (1) Magnitude-squared coherence per band between aligned warp and handcam (ceiling test).
# (2) Nonlinearity probes: coherence at different loudness levels; gain-vs-loudness.
# (3) Stereo quick check: player-click vs taiko events, L/R ILD, correlation, IPD, TDOA.
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


def main():
    ctx, _ = load_wav(os.path.join(WORK_DIR, "golden_ctx.wav"))
    r_fix, _ = load_wav(os.path.join(WORK_DIR, "ref_warp_fixed.wav"))
    s0, s1 = int(GOLDEN_S0 * SR), int((GOLDEN_S0 + GOLDEN_LEN) * SR)
    y = ctx[s0:s1]
    r = r_fix[s0:s1]
    ym, rm = y.mean(axis=1), r.mean(axis=1)
    out = {}

    # ---------- MSC per band ----------
    nfft = 8192
    bands = [(40, 150), (150, 300), (300, 600), (600, 1200), (1200, 2400),
             (2400, 4800), (4800, 9600), (9600, 19000)]
    msc = {}
    f_full, Pxy = sig.csd(ym, rm, fs=SR, nperseg=nfft, noverlap=nfft // 2)
    _, Pyy = sig.welch(ym, fs=SR, nperseg=nfft, noverlap=nfft // 2)
    _, Prr = sig.welch(rm, fs=SR, nperseg=nfft, noverlap=nfft // 2)
    coh = np.abs(Pxy) ** 2 / (Pyy * Prr + 1e-20)
    for lo, hi in bands:
        sel = (f_full >= lo) & (f_full < hi)
        msc[f"{lo}-{hi}"] = float(np.mean(coh[sel]))
    out["msc_per_band"] = msc
    print("MSC (coherence) per band, aligned warp vs handcam mid:")
    for k, v in msc.items():
        print(f"    {k:>12s} Hz: {v:.3f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogx(f_full[1:], coh[1:])
    ax.set_xlabel("Hz"); ax.set_ylabel("coherence"); ax.set_ylim(0, 1)
    ax.set_title("MSC: handcam vs aligned reference (golden 15 s)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(WORK_DIR, "msc.png"), dpi=110)

    # ---------- nonlinearity: coherence vs loudness level ----------
    # split the golden window into 1 s blocks, compute MSC in bass and mid band
    # separately for loud vs quiet blocks
    nb = int(1.0 * SR)
    blocks = []
    for t0 in range(0, len(ym) - nb, nb):
        yb_, rb_ = ym[t0:t0 + nb], rm[t0:t0 + nb]
        rl = 10 * np.log10(np.sum(rb_ ** 2) / nb + 1e-12)
        f, Pxy = sig.csd(yb_, rb_, fs=SR, nperseg=4096)
        _, Pyy = sig.welch(yb_, fs=SR, nperseg=4096)
        _, Prr = sig.welch(rb_, fs=SR, nperseg=4096)
        c = np.abs(Pxy) ** 2 / (Pyy * Prr + 1e-20)
        bass = c[(f >= 40) & (f < 300)].mean()
        mid = c[(f >= 300) & (f < 2000)].mean()
        blocks.append((rl, bass, mid))
    blocks = np.array(blocks)
    loud = blocks[:, 0] > np.median(blocks[:, 0])
    out["coherence_by_loudness"] = {
        "loud_bass": float(blocks[loud, 1].mean()), "quiet_bass": float(blocks[~loud, 1].mean()),
        "loud_mid": float(blocks[loud, 2].mean()), "quiet_mid": float(blocks[~loud, 2].mean())}
    print("coherence by block loudness:", out["coherence_by_loudness"])

    # ---------- stereo quick check ----------
    # events: same detection as audit
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
        med = np.median(f); mad = np.median(np.abs(f - med)) + 1e-9
        pk, _ = sig.find_peaks(f, height=med + k * mad, distance=max(1, int(0.06 * FR)))
        return pk

    t_off = N / (2 * SR)
    pk_h = pick(fh); t_h = (pk_h + 1) / FR + t_off
    dh = np.array([np.min(np.abs(t - ref_pk)) if len(ref_pk) else 9e9 for t in t_h])
    clicks = t_h[dh > 0.06]
    pk_l = pick(fl); t_l = (pk_l + 1) / FR + t_off
    dl = np.array([np.min(np.abs(t - ref_pk)) if len(ref_pk) else 9e9 for t in t_l])
    taiko = t_l[dl > 0.06]

    def stereo_features(events, half=0.05):
        rows = []
        for t in events:
            w0, w1 = int(max(0, (t - half) * SR)), int(min(len(y), (t + half) * SR))
            if w1 - w0 < int(0.04 * SR):
                continue
            L, R = y[w0:w1, 0], y[w0:w1, 1]
            ild = 10 * np.log10((np.sum(L ** 2) + 1e-12) / (np.sum(R ** 2) + 1e-12))
            Ln, Rn = L - L.mean(), R - R.mean()
            corr = float(np.dot(Ln, Rn) / (np.linalg.norm(Ln) * np.linalg.norm(Rn) + 1e-12))
            # IPD weighted by magnitude in 1-4 kHz
            FL = np.fft.rfft(sig.windows.hann(len(L)) * L)
            FR_ = np.fft.rfft(sig.windows.hann(len(L)) * R)
            fbin = np.fft.rfftfreq(len(L), 1 / SR)
            bsel = (fbin >= 1000) & (fbin < 4000)
            mag = np.abs(FL)[bsel] * np.abs(FR_)[bsel]
            ipd = np.angle(FL[bsel] * np.conj(FR_[bsel]))
            ipd_w = float(np.sum(ipd * mag) / (np.sum(mag) + 1e-12))
            # TDOA via GCC-PHAT between channels (search +-1 ms)
            n2 = 1 << int(np.ceil(np.log2(2 * len(L))))
            GL = np.fft.rfft(Ln, n2); GR = np.fft.rfft(Rn, n2)
            g = np.fft.irfft(GL * np.conj(GR) / (np.abs(GL * np.conj(GR)) + 1e-12), n2)
            lags = np.concatenate([g[len(g) - int(1e-3 * SR):], g[:int(1e-3 * SR) + 1]])
            lag_axis = np.concatenate([np.arange(-int(1e-3 * SR), 0), np.arange(0, int(1e-3 * SR) + 1)])
            tdoa = float(lag_axis[int(np.argmax(lags))]) / SR
            rows.append({"t": float(t), "ild_db": ild, "lr_corr": corr, "ipd_rad": ipd_w, "tdoa_us": tdoa * 1e6})
        return rows

    ev = {"clicks": stereo_features(clicks), "taiko": stereo_features(taiko)}
    for name, rows in ev.items():
        if rows:
            ild = np.array([r["ild_db"] for r in rows])
            corr = np.array([r["lr_corr"] for r in rows])
            tdoa = np.array([r["tdoa_us"] for r in rows])
            print(f"{name}: n={len(rows)}  ILD median {np.median(ild):+.2f} dB IQR [{np.percentile(ild,25):+.2f},{np.percentile(ild,75):+.2f}]"
                  f"  L/R corr median {np.median(corr):.3f}  TDOA median {np.median(tdoa):+.0f} us IQR [{np.percentile(tdoa,25):+.0f},{np.percentile(tdoa,75):+.0f}]")
    out["stereo"] = ev

    # handcam master L/R correlation (overall + music-only pool)
    pool = np.ones(len(ym), bool)
    for t in np.concatenate([clicks, taiko]):
        pool[int(max(0, (t - 0.15) * SR)):int(min(len(ym), (t + 0.25) * SR))] = False
    Ln, Rn = y[:, 0] - y[:, 0].mean(), y[:, 1] - y[:, 1].mean()
    out["lr_corr_overall"] = float(np.dot(Ln, Rn) / (np.linalg.norm(Ln) * np.linalg.norm(Rn)))
    Lp, Rp = y[pool, 0], y[pool, 1]
    out["lr_corr_music_pool"] = float(np.corrcoef(Lp, Rp)[0, 1]) if len(Lp) > 1000 else None
    print(f"L/R corr overall {out['lr_corr_overall']:.3f}, music pool {out['lr_corr_music_pool']}")

    save_json(os.path.join(WORK_DIR, "gate_diagnostics.json"), out)
    print("gate_diagnostics.json saved")


if __name__ == "__main__":
    main()
