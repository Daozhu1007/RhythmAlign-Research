# R4.5 - evaluation of D0..D3 vs R4 C1/C2/C3 on the same 84 refined contacts.
# Same onset detector family as R1 audit / R4 30_evaluate.py (STFT flux 2-12 kHz,
# peaks > median + 4*MAD, 60 ms min distance), N=1536 HOP=384 @ 48 kHz.
# New: chop metric - candidate-vs-C1 attenuation during C1 active audio.
# All objective proxies - final judgment requires ears.
import json
import os

import numpy as np
import librosa
import soundfile as sf
from scipy import signal as sig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R4 = r"D:\Code\RhythmAlign\experiments\r4_contact_reconstruction"
R45 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(R45, "outputs")
WORK = os.path.join(R45, "work")
LOG = os.path.join(R45, "logs")
R1_OUT = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs"
R1_WORK = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\work"

SR = 48000
NF, HOP = 1536, 384
MATCH_TOL = 0.030
GATE_AMP = 1e-6

STEMS = {f"{v}_interaction": os.path.join(OUT, f"{v}_interaction.wav")
         for v in ["D0", "D1", "D2", "D3"]}
STEMS.update({f"{v}_interaction": os.path.join(R4, "outputs", f"{v}_interaction.wav")
              for v in ["C1", "C2", "C3"]})
FINAL_MIXES = {f"{v}_final_mix": os.path.join(OUT, f"{v}_final_mix.wav")
               for v in ["D0", "D1", "D2", "D3"]}
FINAL_MIXES.update({f"{v}_final_mix": os.path.join(R4, "outputs", f"{v}_final_mix.wav")
                    for v in ["C1", "C2", "C3"]})

CHOP_FRAME = 0.005      # 5 ms RMS frames
CHOP_HOP = 0.002        # 2 ms hop -> episode durations quantized to 2 ms
CHOP_RATIO = 0.8        # >20% attenuation vs C1


def load_mono(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float64)


def onsets(y):
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
    k = (np.abs(stem) > GATE_AMP).astype(int)
    d = np.diff(np.concatenate([[0], k, [0]]))
    return [(s / SR, e / SR) for s, e in zip(np.where(d > 0)[0], np.where(d < 0)[0])]


def chop_metric(y_cand, y_c1, active):
    """Attenuation episodes of y_cand relative to y_c1 during C1-active audio.
    RMS in 5 ms frames / 2 ms hop; attenuation = rms_cand < 0.8 * rms_c1 while
    rms_c1 above a -80 dBFS floor."""
    fl, hp = int(CHOP_FRAME * SR), int(CHOP_HOP * SR)
    n_f = (len(y_c1) - fl) // hp + 1
    idx = np.arange(n_f)[:, None] * hp + np.arange(fl)[None, :]
    r1 = np.sqrt(np.mean(y_c1[idx] ** 2, axis=1))
    r2 = np.sqrt(np.mean(y_cand[idx] ** 2, axis=1))
    act = active[np.minimum((np.arange(n_f) * hp + fl // 2), len(active) - 1)]
    floor = 1e-4
    att = (r2 < CHOP_RATIO * r1) & (r1 > floor) & act
    total = float(att.sum() / max(act.sum(), 1))
    d = np.diff(np.concatenate([[0], att.astype(int), [0]]))
    starts, ends = np.where(d > 0)[0], np.where(d < 0)[0]
    durs = (ends - starts) * CHOP_HOP
    return {"active_time_s": float(act.sum() * CHOP_HOP),
            "attenuated_fraction": total,
            "episodes": int(len(durs)),
            "episodes_ge_20ms": int(np.sum(durs >= 0.020)),
            "episodes_ge_50ms": int(np.sum(durs >= 0.050)),
            "episode_time_total_s": float(durs.sum()),
            "episode_dur_ms_median": (float(np.median(durs) * 1000) if len(durs) else 0.0),
            "episode_dur_ms_p95": (float(np.percentile(durs, 95) * 1000) if len(durs) else 0.0)}


def main():
    rep = {"sr": SR, "detector": f"STFT N={NF} HOP={HOP}, flux 2-12kHz, med+4MAD, 60ms",
           "chop_def": (f"5ms RMS frames/2ms hop, cand < 0.8*C1 while C1 active "
                        f"(env_D0>=0.5) and C1 frame > -80 dBFS")}

    R = json.load(open(os.path.join(R4, "outputs", "contacts_refined.json")))
    events = [e for e in R["events"]
              if e["refined_confidence"] != "no_transient" and e["type"] != "none"]
    tcts = np.array([e["t_audio_refined"] for e in events])
    conf = [e["refined_confidence"] for e in events]
    strong = np.array([c == "strong" for c in conf])
    gate = json.load(open(os.path.join(R1_WORK, "gate_diagnostics.json")))
    taiko = np.array([c["t"] for c in gate["stereo"]["taiko"]])
    near_contact = np.array([np.min(np.abs(tcts - t)) <= 0.080 for t in taiko])

    raw = load_mono(os.path.join(R1_OUT, "golden_raw.wav"))
    ref = load_mono(os.path.join(R1_WORK, "ref_warp_fixed.wav"))[int(3.0 * SR):int(18.0 * SR)]
    c0 = load_mono(os.path.join(R4, "outputs", "C0_clean_music.wav"))
    assert len(ref) == len(raw) == len(c0) == 15 * SR

    ref_on = onsets(ref)
    # music-heavy / quiet frames (R4 convention), excluding contact +-80 ms
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

    env_d0 = np.load(os.path.join(WORK, "env_D0.npy"))
    active = env_d0 >= 0.5
    y1 = load_mono(STEMS["C1_interaction"])

    y_cache = {}
    results = {}
    for name, path in {**STEMS, **FINAL_MIXES}.items():
        y = y_cache.get(name) or load_mono(path)
        y_cache[name] = y
        on = onsets(y)
        m_ct = match(tcts, on)
        m_tk = match(taiko, on)
        hit = [i for i, x in enumerate(m_ct) if x is not None]
        hit_s = [i for i in hit if strong[i]]
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

        cov = float((np.abs(y) > GATE_AMP).mean())
        tk_leak = [float(t) for t, x, nc in zip(taiko, m_tk, near_contact)
                   if x is not None and not nc]
        r = {
            "contact_retention": {"matched": len(hit), "rate": len(hit) / len(tcts),
                                  "strong_matched": len(hit_s),
                                  "strong_rate": len(hit_s) / int(strong.sum())},
            "local_rms_vs_raw_db": ({"median": float(np.median(lvl)),
                                     "p5": float(np.percentile(lvl, 5)),
                                     "p95": float(np.percentile(lvl, 95))} if lvl else None),
            "peak_ratio_vs_raw": {"median": float(np.median(pk_r)) if pk_r else None,
                                  "strong_median": float(np.median(
                                      [pk_r[j] for j, i in enumerate(hit) if strong[i]]))
                                  if any(strong[i] for i in hit) else None,
                                  "dynamics_corr": dyn},
            "music_suppression_db": {"heavy": sup(heavy), "quiet": sup(quiet)},
            "taiko_leakage": {"matched": int(np.sum([x is not None for x in m_tk])),
                              "out_of_window": len(tk_leak), "out_of_window_times": tk_leak},
            "gate_coverage": cov if cov < 0.999 else None,
            "missed_contacts": [float(tcts[i]) for i, x in enumerate(m_ct) if x is None],
        }
        if name in FINAL_MIXES:
            data, _ = sf.read(path, dtype="float64", always_2d=True)
            r["peak_dbfs"] = float(20 * np.log10(np.max(np.abs(data)) + 1e-12))
            r["peak_dbfs_headroom"] = -r["peak_dbfs"]
        if name in STEMS:
            r["chop_vs_C1"] = chop_metric(y, y1, active)
        results[name] = r
        ck = r.get("chop_vs_C1", {})
        print(f"\n=== {name} ===")
        print(f"  retention {len(hit)}/{len(tcts)} (strong {len(hit_s)}/{int(strong.sum())}) "
              f"| peak x{r['peak_ratio_vs_raw']['median']:.2f} (strong "
              f"x{r['peak_ratio_vs_raw']['strong_median']:.2f}) | dyn corr {dyn:.2f}")
        if r["local_rms_vs_raw_db"]:
            lv = r["local_rms_vs_raw_db"]
            print(f"  local RMS vs raw {lv['median']:+.1f} dB [{lv['p5']:+.1f},{lv['p95']:+.1f}]")
        ms = r["music_suppression_db"]
        print(f"  music sup heavy {ms['heavy']:+.1f} dB quiet {ms['quiet']:+.1f} dB "
              f"| taiko {r['taiko_leakage']['matched']}/{len(taiko)} "
              f"(out-of-window {r['taiko_leakage']['out_of_window']}) | coverage {cov:.3f}")
        if ck:
            print(f"  chop vs C1: {ck['attenuated_fraction']*100:.2f}% of C1-active | "
                  f"episodes {ck['episodes']} (>=20ms {ck['episodes_ge_20ms']}, "
                  f">=50ms {ck['episodes_ge_50ms']}) | ep dur med "
                  f"{ck['episode_dur_ms_median']:.0f} ms p95 {ck['episode_dur_ms_p95']:.0f} ms")
        if "peak_dbfs" in r:
            print(f"  final mix peak {r['peak_dbfs']:.2f} dBFS "
                  f"(headroom {r['peak_dbfs_headroom']:.2f} dB)")

    # D0 must equal C1 sample-exactly (both stem and final mix)
    for kind in ["interaction", "final_mix"]:
        a = load_mono(os.path.join(R4, "outputs", f"C1_{kind}.wav"))
        b = load_mono(os.path.join(OUT, f"D0_{kind}.wav"))
        d = float(np.max(np.abs(a - b)))
        results[f"D0_equals_C1_{kind}_max_abs_diff"] = d
        print(f"\nD0 vs C1 {kind}: max abs diff {d:.3e}")

    rep["frames"] = {"n": fl, "heavy_excl_contacts": len(heavy), "quiet_excl_contacts": len(quiet)}
    rep["ref_music_onsets"] = len(ref_on)
    rep["versions"] = results
    with open(os.path.join(LOG, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print("\nsaved logs/evaluation.json")

    # figure: exact gate envelopes in the dense zone where C3 chopping was heard
    def env_from_stem(name, attack, release):
        spans = [(int(s * SR), int(e * SR)) for s, e in gate_spans(y_cache[name])]
        return envelope_from_spans(len(y_cache[name]), spans, attack, release)

    def envelope_from_spans(n, spans, attack, release):
        env = np.zeros(n)
        A, Rn = max(int(attack * SR), 2), max(int(release * SR), 2)
        for s, e in spans:
            s, e = max(s, 0), min(e, n)
            if e <= s:
                continue
            seg = np.ones(e - s)
            a = min(A, len(seg))
            seg[:a] = np.minimum(seg[:a], 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, a)))
            r = min(Rn, len(seg))
            seg[len(seg) - r:] = np.minimum(
                seg[len(seg) - r:], 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, r)))
            env[s:e] = np.maximum(env[s:e], seg)
        return env

    envs = {"C1_interaction": np.load(os.path.join(WORK, "env_D0.npy")),
            "D2_interaction": np.load(os.path.join(WORK, "env_D2.npy")),
            "D3_interaction": np.load(os.path.join(WORK, "env_D3.npy")),
            "C3_interaction": env_from_stem("C3_interaction", 0.008, 0.045)}
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t = np.arange(len(raw)) / SR
    m = (t >= 3.3) & (t <= 7.5)
    for rid, t0, t1 in [("R3", 3.3, 4.75), ("R4", 4.75, 10.42)]:
        axes[0].axvspan(t0, t1, color="tab:gray", alpha=0.10)
        axes[0].text((t0 + t1) / 2, 1.05, rid, ha="center", fontsize=8, color="gray")
    for nm, col in [("C1_interaction", "tab:orange"), ("C3_interaction", "tab:red"),
                    ("D2_interaction", "tab:green"), ("D3_interaction", "tab:blue")]:
        axes[0].plot(t[m], envs[nm][m], lw=0.9, color=col, alpha=0.85, label=nm)
    axes[0].set_ylabel("gate envelope (exact)")
    axes[0].set_ylim(-0.03, 1.12)
    axes[0].legend(fontsize=8, ncol=4, loc="lower right")
    axes[0].set_title("R4.5 gates in dense zone 3.3-7.5 s (C3 chopping reference; "
                      "gray = oracle long regions R3/R4 kept discrete+bridge in D3)")
    durs = []
    for nm, col in [("C3_interaction", "tab:red"), ("D2_interaction", "tab:green"),
                    ("D3_interaction", "tab:blue")]:
        c = results[nm]["chop_vs_C1"]
        axes[1].bar([nm], [c["episodes_ge_20ms"]], color=col, alpha=0.8)
        axes[1].bar([nm], [c["episodes"] - c["episodes_ge_20ms"]],
                    bottom=[c["episodes_ge_20ms"]], color=col, alpha=0.35)
        durs.append(f"{nm}: {c['attenuated_fraction']*100:.2f}% / "
                    f"{c['episodes']} ep / {c['episodes_ge_50ms']} >=50ms")
    axes[1].set_ylabel("chop episodes vs C1")
    axes[1].set_title("chop metric (solid >=20ms) | " + " | ".join(durs), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "r45_gates_and_chop.png"), dpi=110)
    plt.close(fig)
    print("saved outputs/r45_gates_and_chop.png")


if __name__ == "__main__":
    main()
