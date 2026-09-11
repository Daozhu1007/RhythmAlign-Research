# R4 - evaluation of C1/C2/C3 vs R2/R3 baselines on the same 84 refined contacts.
# Same onset detector family as R1 audit / R2 11_evaluate.py (STFT flux 2-12 kHz,
# peaks > median + 4*MAD, 60 ms min distance), rescaled to 48 kHz (N=1536 HOP=384
# == 32 ms / 8 ms, identical time-frequency resolution to R2's 1024/256 @ 32 kHz).
# All objective proxies - final judgment requires ears (flagged in REPORT.md).
import json
import os

import numpy as np
import librosa
import soundfile as sf
from scipy import signal as sig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R4 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(R4, "outputs")
LOG = os.path.join(R4, "logs")
os.makedirs(LOG, exist_ok=True)
R1_OUT = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs"
R1_WORK = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\work"
R2_OUT = r"D:\Code\RhythmAlign\experiments\r2_target_separation\outputs"
R3_OUT = r"D:\Code\RhythmAlign\experiments\r3_audio_query\outputs"

SR = 48000
NF, HOP = 1536, 384            # 32 ms / 8 ms - same resolution as R2 @32k
MATCH_TOL = 0.030
GATE_AMP = 1e-6                # |sample| above which a C-series gate counts as open

STEMS = {
    "raw_input": os.path.join(R1_OUT, "golden_raw.wav"),
    "C0_music_only": os.path.join(OUT, "C0_clean_music.wav"),
    "C1_interaction": os.path.join(OUT, "C1_interaction.wav"),
    "C2_interaction": os.path.join(OUT, "C2_interaction.wav"),
    "C3_interaction": os.path.join(OUT, "C3_interaction.wav"),
    "R2_target": os.path.join(R2_OUT, "audiosep_raw_p2_target.wav"),
    "R3_target": os.path.join(R3_OUT, "clapsep_n1_target.wav"),
}
FINAL_MIXES = {
    "C1_final_mix": os.path.join(OUT, "C1_final_mix.wav"),
    "C2_final_mix": os.path.join(OUT, "C2_final_mix.wav"),
    "C3_final_mix": os.path.join(OUT, "C3_final_mix.wav"),
    "R2_final_mix": os.path.join(R2_OUT, "audiosep_raw_p2_final_mix.wav"),
    "R3_final_mix": os.path.join(R3_OUT, "clapsep_n1_final_mix.wav"),
}


def load_mono(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float64)


def onsets(y):
    """Positive spectral flux 2-12 kHz, peaks > median+4*MAD, 60 ms apart."""
    S = np.abs(librosa.stft(y.astype(np.float32), n_fft=NF, hop_length=HOP)) + 1e-10
    Sdb = librosa.amplitude_to_db(S)
    fr = librosa.fft_frequencies(sr=SR, n_fft=NF)
    dS = np.maximum(0.0, np.diff(Sdb, axis=1))
    fh = dS[(fr >= 2000) & (fr < 12000)].sum(axis=0)
    med = np.median(fh)
    mad = np.median(np.abs(fh - med)) + 1e-9
    pk, _ = sig.find_peaks(fh, height=med + 4.0 * mad,
                           distance=max(1, int(0.06 * SR / HOP)))
    return (pk + 1) / (SR / HOP) + NF / (2 * SR)


def match(times, targets, tol=MATCH_TOL):
    out = []
    for t in times:
        d = np.min(np.abs(np.asarray(targets) - t)) if len(targets) else 9e9
        out.append(float(d) if d <= tol else None)
    return out


def local_db(y, t, half=0.050):
    a, b = int(max(0, (t - half) * SR)), int(min(len(y), (t + half) * SR))
    if b - a < 32:
        return None
    return float(20 * np.log10(np.sqrt(np.mean(y[a:b] ** 2)) + 1e-12))


def db_rms(y):
    return float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12))


def gate_spans(stem):
    """Covered spans of a hard-gated stem (exact zeros between events)."""
    k = (np.abs(stem) > GATE_AMP).astype(int)
    d = np.diff(np.concatenate([[0], k, [0]]))
    return [(s / SR, e / SR) for s, e in zip(np.where(d > 0)[0], np.where(d < 0)[0])]


def delayed_corr(x, ref, spans, lo=0.0002, hi=0.020):
    """Per span: max |corr(x, ref shifted +-tau)| for tau in [lo, hi].
    A strong peak at nonzero tau = the stem carries a delayed music copy (comb risk)."""
    best_c, best_d = [], []
    for s, e in spans:
        a, b = int(s * SR), int(e * SR)
        if b - a < int(0.03 * SR):
            continue
        xw, yw = x[a:b], ref[a:b]
        xw = xw - xw.mean()
        yw = yw - yw.mean()
        bc, bd = 0.0, 0.0
        for lag in range(int(lo * SR), int(hi * SR), 2):
            for sgn in (1, -1):
                ys = np.roll(yw, sgn * lag)
                r = abs(float(np.dot(xw, ys) /
                              (np.sqrt(np.dot(xw, xw) * np.dot(ys, ys)) + 1e-12)))
                if r > bc:
                    bc, bd = r, sgn * lag / SR * 1000
        best_c.append(bc)
        best_d.append(bd)
    return best_c, best_d


def comb_ripple(fmix, c0, spans):
    """Per span: p95-p5 of |20log10(fmix/c0)| in 300-4000 Hz.
    Deep regular notches (comb filtering) inflate the ripple."""
    S_mix = np.abs(librosa.stft(fmix.astype(np.float32), n_fft=NF, hop_length=HOP)) + 1e-10
    S_c0 = np.abs(librosa.stft(c0.astype(np.float32), n_fft=NF, hop_length=HOP)) + 1e-10
    fr = librosa.fft_frequencies(sr=SR, n_fft=NF)
    band = (fr >= 300) & (fr <= 4000)
    out = []
    for s, e in spans:
        f0, f1 = int(s * SR / HOP), int(e * SR / HOP)
        if f1 - f0 < 4 or f1 > S_mix.shape[1]:
            continue
        ratio = 20 * np.log10(S_mix[band, f0:f1] / S_c0[band, f0:f1])
        out.append(float(np.percentile(ratio, 95) - np.percentile(ratio, 5)))
    return out


def music_share(stem, ref, spans):
    """LS projection of stem onto ref per span -> share of stem energy explained by music."""
    out = []
    for s, e in spans:
        a, b = int(s * SR), int(e * SR)
        x, y = stem[a:b], ref[a:b]
        if b - a < int(0.05 * SR):
            continue
        g = float(np.dot(x, y) / (np.dot(y, y) + 1e-12))
        out.append(float(10 * np.log10(g * g * np.dot(y, y) / (np.dot(x, x)) + 1e-12)))
    return out


def main():
    rep = {"sr": SR, "detector": f"STFT N={NF} HOP={HOP}, flux 2-12kHz, med+4MAD, 60ms"}

    # ---------------- ground truth ----------------
    R = json.load(open(os.path.join(OUT, "contacts_refined.json")))
    events = [e for e in R["events"]
              if e["refined_confidence"] != "no_transient" and e["type"] != "none"]
    tcts = np.array([e["t_audio_refined"] for e in events])
    conf = [e["refined_confidence"] for e in events]
    gate = json.load(open(os.path.join(R1_WORK, "gate_diagnostics.json")))
    taiko = np.array([c["t"] for c in gate["stereo"]["taiko"]])
    t_near = [t for t in taiko if np.min(np.abs(tcts - t)) <= 0.080]
    rep["ground_truth"] = {"n_contacts": len(tcts),
                           "by_confidence": {c: conf.count(c) for c in sorted(set(conf))},
                           "n_taiko": len(taiko),
                           "taiko_within_80ms_of_contact": len(t_near)}

    ref = load_mono(os.path.join(R1_WORK, "ref_warp_fixed.wav"))[int(3.0 * SR):int(18.0 * SR)]
    raw = load_mono(os.path.join(R1_OUT, "golden_raw.wav"))
    c0 = load_mono(os.path.join(OUT, "C0_clean_music.wav"))
    assert len(ref) == len(raw) == len(c0) == 15 * SR

    ref_on = onsets(ref)
    rep["ref_music_onsets"] = len(ref_on)

    # music-heavy / music-quiet frames (R2/R3 convention), excluding contact +-80 ms
    n = 1024
    fl = len(ref) // n
    frms = np.array([db_rms(ref[i * n:(i + 1) * n]) for i in range(fl)])
    q75, q25 = np.quantile(frms, 0.75), np.quantile(frms, 0.25)
    near = np.zeros(fl, bool)
    for t in tcts:
        i0, i1 = int((t - 0.08) * SR) // n, int((t + 0.08) * SR) // n
        near[max(i0, 0):min(i1 + 1, fl)] = True
    heavy = np.where((frms >= q75) & ~near)[0]
    quiet = np.where((frms <= q25) & ~near)[0]
    rep["frames"] = {"n": fl, "heavy_excl_contacts": len(heavy), "quiet_excl_contacts": len(quiet)}
    print(f"contacts {len(tcts)} {rep['ground_truth']['by_confidence']} | taiko {len(taiko)} "
          f"({len(t_near)} within 80 ms of contact) | ref onsets {len(ref_on)} | "
          f"frames heavy {len(heavy)} quiet {len(quiet)}")

    # window set for in-window metrics: C3 gate spans (main method) on all versions
    spans3 = gate_spans(load_mono(os.path.join(OUT, "C3_interaction.wav")))
    rep["c3_spans"] = {"n": len(spans3),
                       "coverage": float((np.abs(load_mono(os.path.join(OUT, "C3_interaction.wav")))
                                           > GATE_AMP).mean())}

    y_cache = {}
    results = {}
    for name, path in {**STEMS, **FINAL_MIXES}.items():
        y = y_cache.get(name) or load_mono(path)
        y_cache[name] = y
        on = onsets(y)
        m_ct = match(tcts, on)
        m_tk = match(taiko, on)
        m_rf = match(on, ref_on)
        cov = float((np.abs(y) > GATE_AMP).mean())

        hit = [i for i, x in enumerate(m_ct) if x is not None]
        lvl = [local_db(y, tcts[i]) - local_db(raw, tcts[i]) for i in hit]
        pk_r, pk_abs = [], []
        for i in hit:
            a, b = int((tcts[i] - 0.012) * SR), int((tcts[i] + 0.012) * SR)
            pk_abs.append(float(np.max(np.abs(raw[a:b]))))
            pk_r.append(min(float(np.max(np.abs(y[a:b])) / (pk_abs[-1] + 1e-12)), 8.0))
        dyn = float(np.corrcoef(pk_r, pk_abs)[0, 1]) if len(hit) > 2 else None

        def sup(idx):
            v = [local_db(y, (i + 0.5) * n / SR) - local_db(raw, (i + 0.5) * n / SR) for i in idx]
            v = [x for x in v if x is not None]
            return float(np.median(v)) if v else None
        s_heavy, s_quiet = sup(heavy), sup(quiet)

        r = {
            "n_onsets": len(on),
            "contact_retention": {"matched": len(hit), "rate": len(hit) / len(tcts),
                                  "by_confidence": {c: int(np.sum([conf[i] == c and m_ct[i] is not None
                                                                   for i in range(len(tcts))]))
                                                    for c in sorted(set(conf))}},
            "contact_level_vs_raw_db": ({"median": float(np.median(lvl)),
                                         "p5": float(np.percentile(lvl, 5)),
                                         "p95": float(np.percentile(lvl, 95))} if lvl else None),
            "peak_ratio_vs_raw": {"median": float(np.median(pk_r)) if pk_r else None,
                                  "iqr": ([float(np.percentile(pk_r, 25)),
                                           float(np.percentile(pk_r, 75))] if pk_r else None),
                                  "dynamics_corr": dyn},
            "missed_contacts": [float(tcts[i]) for i, x in enumerate(m_ct) if x is None],
            "music_suppression_db": {"heavy": s_heavy, "quiet": s_quiet,
                                     "selectivity": (float(np.median(lvl)) - s_heavy)
                                     if (s_heavy is not None and lvl) else None},
            "taiko_leakage": {"matched": int(np.sum([x is not None for x in m_tk])),
                              "times": [float(t) for t, x in zip(taiko, m_tk) if x is not None]},
            "ref_onsets_present": int(np.sum([x is not None for x in m_rf])),
            "false_onsets": [float(t) for t, x, rr in zip(on, m_ct, m_rf)
                             if x is None and rr is None
                             and not any(abs(t - tk) <= MATCH_TOL for tk in taiko)],
            "corr_with_ref": float(np.corrcoef(y, ref)[0, 1]),
            "gate_coverage": cov if cov < 0.999 else None,
            "music_share_in_c3windows_db": {
                "median": float(np.median(music_share(y, ref, spans3)))},
            "delayed_corr_vs_ref_in_c3windows": {},
        }
        dc, dd = delayed_corr(y if name in STEMS else 2 * (y - c0), ref, spans3)
        r["delayed_corr_vs_ref_in_c3windows"] = {
            "max_corr_median": float(np.median(dc)),
            "max_corr_p90": float(np.percentile(dc, 90)),
            "argmax_delay_ms_median": float(np.median(dd))}
        if name in FINAL_MIXES:
            data, _ = sf.read(path, dtype="float64", always_2d=True)
            r["peak_dbfs"] = float(20 * np.log10(np.max(np.abs(data)) + 1e-12))
            r["final_vs_c0_heavy_level_db"] = float(np.median(
                [local_db(y, (i + .5) * n / SR) - local_db(c0, (i + .5) * n / SR) for i in heavy]))
            rip = comb_ripple(y, c0, spans3)
            r["comb_ripple_final_vs_c0_db"] = {"median": float(np.median(rip)),
                                               "p95": float(np.percentile(rip, 95))}
        results[name] = r
        lv = r["contact_level_vs_raw_db"]
        print(f"\n=== {name} ===")
        print(f"  retention {len(hit)}/{len(tcts)} ({len(hit)/len(tcts):.0%}) "
              f"{r['contact_retention']['by_confidence']}")
        if lv:
            print(f"  contact level vs raw {lv['median']:+.1f} dB "
                  f"[{lv['p5']:+.1f},{lv['p95']:+.1f}] | peak x{r['peak_ratio_vs_raw']['median']:.2f} "
                  f"| dyn corr {dyn:.2f}")
        print(f"  music sup heavy {s_heavy:+.1f} dB quiet {s_quiet:+.1f} dB | "
              f"music share in C3 windows {r['music_share_in_c3windows_db']['median']:+.1f} dB")
        print(f"  taiko {r['taiko_leakage']['matched']}/{len(taiko)} | false onsets "
              f"{len(r['false_onsets'])} {r['false_onsets'][:8]}")
        print(f"  corr(ref) {r['corr_with_ref']:.3f} | delayed corr "
              f"{r['delayed_corr_vs_ref_in_c3windows']['max_corr_median']:.2f} "
              f"@{r['delayed_corr_vs_ref_in_c3windows']['argmax_delay_ms_median']:+.1f} ms "
              f"| coverage {cov:.3f}")
        if "peak_dbfs" in r:
            print(f"  peak {r['peak_dbfs']:.2f} dBFS | vs C0 on music-heavy "
                  f"{r['final_vs_c0_heavy_level_db']:+.1f} dB | comb ripple med "
                  f"{r['comb_ripple_final_vs_c0_db']['median']:.1f} p95 "
                  f"{r['comb_ripple_final_vs_c0_db']['p95']:.1f} dB")

    rep["versions"] = results
    with open(os.path.join(LOG, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print("\nsaved logs/evaluation.json")

    # ---------------- figures ----------------
    def spec_ax(ax, y, title):
        S = librosa.amplitude_to_db(
            np.abs(librosa.stft(y.astype(np.float32), n_fft=NF, hop_length=HOP)) + 1e-10)
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, len(y) / SR, 0, SR // 2], vmax=S.max(), vmin=S.max() - 90)
        for t in tcts:
            ax.axvline(t, color="cyan", lw=0.3, alpha=0.5)
        for t in taiko:
            ax.axvline(t, color="lime", lw=0.3, alpha=0.4)
        ax.set_title(title, fontsize=8)
        ax.set_ylabel("Hz")

    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    for ax, (nm, yy) in zip(axes, [("R2 final (audiosep_raw_p2)", y_cache["R2_final_mix"]),
                                   ("R3 final (clapsep_n1)", y_cache["R3_final_mix"]),
                                   ("C1 final mix", y_cache["C1_final_mix"]),
                                   ("C2 final mix", y_cache["C2_final_mix"]),
                                   ("C3 final mix", y_cache["C3_final_mix"])]):
        spec_ax(ax, yy, nm)
    axes[-1].set_xlabel("s (golden window)")
    fig.suptitle("R4 final mixes vs R2/R3 baselines | cyan=contacts lime=taiko", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cmp_final_mixes_spec.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    t = np.arange(len(raw)) / SR
    ax.fill_between(t, 0, 1.1, where=np.abs(y_cache["C1_interaction"]) > GATE_AMP,
                    step="mid", alpha=0.45, color="tab:orange", label="C1 gate")
    ax.fill_between(t, 0, 1.25, where=np.abs(y_cache["C3_interaction"]) > GATE_AMP,
                    step="mid", alpha=0.45, color="tab:red", label="C3 gate")
    for tk in taiko:
        ax.axvline(tk, color="lime", lw=1, alpha=0.7)
    ax.plot(t, np.abs(raw) / np.max(np.abs(raw)), lw=0.3, color="k", alpha=0.5, label="|raw|")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1.4)
    ax.set_xlabel("s")
    ax.legend(loc="upper right")
    ax.set_title("gate coverage vs raw envelope (lime = taiko)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cmp_coverage.png"), dpi=110)
    plt.close(fig)

    # per-event peak retention: dynamics preservation
    raw_pk = []
    for t in tcts:
        a, b = int((t - 0.012) * SR), int((t + 0.012) * SR)
        raw_pk.append(float(np.max(np.abs(raw[a:b]))))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for nm, col in [("C1_interaction", "tab:orange"), ("C2_interaction", "tab:green"),
                    ("C3_interaction", "tab:red"), ("R2_target", "tab:blue"),
                    ("R3_target", "tab:purple")]:
        yy = y_cache[nm]
        pr = []
        for i in range(len(tcts)):
            a, b = int((tcts[i] - 0.012) * SR), int((tcts[i] + 0.012) * SR)
            pr.append(min(float(np.max(np.abs(yy[a:b])) / (raw_pk[i] + 1e-12)), 8.0))
        ax.scatter(raw_pk, pr, s=12, alpha=0.6, label=nm, color=col)
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("raw contact peak amplitude")
    ax.set_ylabel("stem / raw peak ratio (±12 ms)")
    ax.set_title("per-contact peak retention (dynamics preservation)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cmp_peak_retention.png"), dpi=110)
    plt.close(fig)
    print("saved outputs/cmp_final_mixes_spec.png + cmp_coverage.png + cmp_peak_retention.png")


if __name__ == "__main__":
    main()
