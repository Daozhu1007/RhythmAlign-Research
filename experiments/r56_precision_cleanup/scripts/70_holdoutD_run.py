# R5.6 Holdout D - step 6: run the FROZEN R5.6 pipeline (E2) on holdout D,
# build AUTO + ORACLE C1 wavs, and compute the full metric report.
# The detector config is outputs/detector_config_r56_frozen.json; E0 = R5.5
# frozen detector recomputed on the same features for reference.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import soundfile as sf
from scipy import signal as sig

import common56 as c
from detector55 import fused_score
from r5_common import SR, audio_onsets, detect_candidates, extract_features
from importlib import import_module

cal = import_module("30_completion_calibration")

DUR = 18.0


def load_mono(path):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float64)


def gate_spans(stem, amp=1e-6):
    k = (np.abs(stem) > amp).astype(int)
    d = np.diff(np.concatenate([[0], k, [0]]))
    return [(s / SR, e / SR) for s, e in zip(np.where(d > 0)[0], np.where(d < 0)[0])]


def main():
    # ---- features (first and only detector-related run on holdout D) ----
    n, series = extract_features(os.path.join(c.R56_OUT, "holdoutD_video.mp4"))
    np.savez(os.path.join(c.R56_WORK, "holdoutD_features.npz"), fps=c.FPS, **series)
    print("holdoutD frames:", n)

    comb = c.get_comb_files(os.path.join(c.R56_WORK, "holdoutD_ctx.wav"),
                            os.path.join(c.R56_WORK, "holdoutD_ref_warp.wav"), DUR)
    bands = c.band_envelopes_files(os.path.join(c.R56_WORK, "holdoutD_ctx.wav"),
                                   os.path.join(c.R56_WORK, "holdoutD_ref_warp.wav"), DUR)
    np.save(os.path.join(c.R56_WORK, "holdoutD_comb.npy"), comb)

    # ---- E0 reference: R5.5 FROZEN detector (prompt definition of E0) ----
    _, _, _, _, e0_cands = c.detect55(series, c.FPS, c.CFG55_FROZEN, rule="A")

    # ---- frozen E2 pipeline ----
    score, ns = fused_score(series, c.CFG55_FROZEN)
    _, mask, ivs, env, cands = c.detect55(series, c.FPS, c.CFG55_FROZEN, rule="A")
    refined = c.refine_candidates([cd["t_video"] for cd in cands], comb,
                                  c.AV_OFF_MS, DUR,
                                  ids=[f"d{i:03d}" for i in range(len(cands))])
    for cd, r in zip(cands, refined):
        r["score"] = cd["score"]
    flags = c.e1_gate_flags(cands, refined, score, ns, ivs, c.E1_CFG)

    e1_cands = [dict(cd, gate=flags[j][2], window=flags[j][1])
                for j, cd in enumerate(cands)]
    e1_t = [refined[j]["t_audio_refined"] for j in range(len(cands))
            if flags[j][1] and refined[j].get("refined_confidence")
            not in (None, "no_transient", "out_of_range")]

    # completion (frozen E2_CFG)
    bursts_s = [(a / c.FPS, b / c.FPS) for a, b in ivs]
    hdn = np.asarray(series["hand_diff"])
    hs = np.percentile(hdn, 99.5) + 1e-9
    added, rejected = [], []
    for p in cal.strong_comb_peaks(comb, c.E2_CFG["min_peak"], c.E2_CFG["min_ratio"],
                                   c.E2_CFG["min_dist_s"]):
        t, k = p["t"], p["k"]
        why = None
        if not any(a <= t <= b for a, b in bursts_s):
            why = "outside_burst"
        elif min([abs(t - x) for x in e1_t], default=10.0) < c.E2_CFG["d_e1_min_ms"] / 1000:
            why = "near_existing_candidate"
        elif min([abs(t - x["t"]) for x in added], default=10.0) < c.E2_CFG["d_e2_min_ms"] / 1000:
            why = "refractory"
        else:
            bs, h1, h2, ba = (float(bands["mid"][k]), float(bands["hi1"][k]),
                              float(bands["hi2"][k]), float(bands["bass"][k]))
            if bs + h1 + h2 < c.E2_CFG["resid_sum_min"] or h1 < c.E2_CFG["hi1_min"]:
                why = f"taiko_guard(bass {ba:.2f} resid {bs + h1 + h2:.2f} hi1 {h1:.2f})"
        if why:
            rejected.append({**p, "reason": why})
            continue
        ki = int(round(t * c.FPS))
        hand = float(min(np.max(hdn[max(ki - 3, 0):ki + 4]) / hs, 2.5))
        added.append({**p, "hand": hand})

    e2_t = e1_t + [a["t"] for a in added]

    # ---- C1 builds: auto (E0 windows, E2 windows) and oracle ----
    ref_full, _ = sf.read(os.path.join(c.R56_WORK, "holdoutD_ref_warp.wav"),
                          dtype="float64", always_2d=True)
    ref = ref_full[int(c.CTX_START * SR):int((c.CTX_START + DUR) * SR)]
    raw_path = os.path.join(c.R56_OUT, "holdoutD_raw.wav")

    e0_used = [r for r in c.refine_candidates([cd["t_video"] for cd in e0_cands],
                                              comb, c.AV_OFF_MS, DUR,
                                              ids=[f"e{i:03d}" for i in range(len(e0_cands))])
               if r.get("refined_confidence") not in (None, "no_transient", "out_of_range")]
    e0_t = [r["t_audio_refined"] for r in e0_used]

    c.build_c1(raw_path, ref, e0_t,
               os.path.join(c.R56_OUT, "holdoutD_e0_C1_interaction.wav"),
               os.path.join(c.R56_OUT, "holdoutD_e0_C1_final_mix.wav"))
    c.build_c1(raw_path, ref, e2_t,
               os.path.join(c.R56_OUT, "holdoutD_auto_C1_interaction.wav"),
               os.path.join(c.R56_OUT, "holdoutD_auto_C1_final_mix.wav"))

    # oracle side
    orc = c.load_json(os.path.join(c.R56_OUT, "holdoutD_oracle_blind.json"))
    contacts = [e for e in orc["events"] if e["status"] == "ok"]
    orc_ref = c.refine_candidates([e["t_video"] for e in contacts], comb,
                                  c.AV_OFF_MS, DUR,
                                  ids=[e["id"] for e in contacts])
    for e, r in zip(contacts, orc_ref):
        r["type"] = e["type"]
        r["oracle_confidence"] = e["confidence"]
    c.save_json(os.path.join(c.R56_OUT, "holdoutD_oracle_refined.json"),
                {"av_off_ms": c.AV_OFF_MS, "n": len(contacts), "events": orc_ref})
    to_all = [r["t_audio_refined"] for r in orc_ref
              if r.get("refined_confidence") not in (None, "no_transient", "out_of_range")]
    c.build_c1(raw_path, ref, to_all,
               os.path.join(c.R56_OUT, "holdoutD_oracle_C1_interaction.wav"),
               os.path.join(c.R56_OUT, "holdoutD_oracle_C1_final_mix.wav"))

    # ---- save visual/refined JSONs ----
    c.save_json(os.path.join(c.R56_OUT, "holdoutD_contacts_auto_visual.json"), {
        "experiment": "R5.6 frozen E2 pipeline on Holdout D",
        "config": "outputs/detector_config_r56_frozen.json",
        "n_bursts": len(ivs),
        "burst_intervals_s": [[round(a / c.FPS, 3), round(b / c.FPS, 3)] for a, b in ivs],
        "n_e0_candidates": len(e0_cands),
        "e1_gate_counts": {
            "strong_present": sum(1 for x in e1_cands if x["gate"].startswith("keep_")),
            "weak_rescue": sum(1 for x in e1_cands if x["gate"].startswith("weak_kept_rescue")),
            "weak_visual": sum(1 for x in e1_cands if "_kept_" in x["gate"]
                               and not x["gate"].startswith("weak_kept_rescue")),
            "weak_nowindow": sum(1 for x in e1_cands if x["gate"].startswith("weak_nowindow")),
            "no_transient": sum(1 for x in e1_cands if x["gate"].startswith("no_transient"))},
        "candidates": e1_cands,
        "completion_added": [{k: round(v, 3) if isinstance(v, float) else v
                              for k, v in a.items() if k != "k"} for a in added],
        "completion_rejected_sample": [{k: (round(v, 2) if isinstance(v, float) else v)
                                        for k, v in a.items() if k != "k"}
                                       for a in rejected[:40]],
    })
    c.save_json(os.path.join(c.R56_OUT, "holdoutD_contacts_auto_refined.json"),
                {"av_off_ms": c.AV_OFF_MS, "events": refined})

    # ---- metrics ----
    rep = {}
    tc = np.array([e["t_video"] for e in contacts])
    strong = [e for e in contacts if e["confidence"] == "high"]
    ts_hi = np.array([e["t_video"] for e in strong])
    for name, cands_t, wins_t, int_p, mix_p in (
            ("E0", [x["t_video"] for x in e0_cands], e0_t,
             os.path.join(c.R56_OUT, "holdoutD_e0_C1_interaction.wav"),
             os.path.join(c.R56_OUT, "holdoutD_e0_C1_final_mix.wav")),
            ("E2", [x["t_video"] for x in e1_cands] + [a["t"] for a in added], e2_t,
             os.path.join(c.R56_OUT, "holdoutD_auto_C1_interaction.wav"),
             os.path.join(c.R56_OUT, "holdoutD_auto_C1_final_mix.wav"))):
        ct = np.array(cands_t)
        m = {"n_candidates": len(cands_t), "n_windows": len(wins_t)}
        for tol in (0.033, 0.050):
            mc, mp, pairs = c.match_one_to_one(ct, tc, tol)
            p = len(mc) / max(len(cands_t), 1)
            r = len(mc) / max(len(tc), 1)
            errs = [abs(e) * 1000 for _, _, e in pairs]
            m[f"tol{int(tol*1000)}"] = {
                "P": round(p, 3), "R": round(r, 3),
                "F1": round(2 * p * r / max(p + r, 1e-9), 3),
                "err_ms_median": round(float(np.median(errs)), 1) if errs else None,
                "err_ms_p95": round(float(np.percentile(errs, 95)), 1) if errs else None}
        hit = [t for t in ts_hi if len(ct) and np.min(np.abs(ct - t)) <= 0.050]
        m["strong_recall"] = round(len(hit) / max(len(ts_hi), 1), 3)
        m["strong_missed"] = [e["id"] for e in strong
                              if not (len(ct) and np.min(np.abs(ct - e["t_video"])) <= 0.050)]
        # audio product metrics
        stem = load_mono(int_p)
        spans = gate_spans(stem)
        on = audio_onsets(stem)
        orc_ts = np.array(to_all)
        covered = lambda t: any(s <= t <= e for s, e in spans)
        m["gate_coverage"] = round(float(np.mean([covered(t) for t in orc_ts])), 3)
        ts_strong = np.array([r["t_audio_refined"] for r in orc_ref
                              if r.get("refined_confidence") == "strong"])
        m["gate_coverage_strong"] = round(float(np.mean([covered(t) for t in ts_strong])), 3) \
            if len(ts_strong) else None
        fw = [(s, e) for s, e in spans
              if len(orc_ts) == 0 or np.min(np.abs(orc_ts - (s + e) / 2)) > 0.080]
        m["false_windows"] = {"count": len(fw), "total_s": round(sum(e - s for s, e in fw), 2)}
        ref_on = audio_onsets(c.ref_slice_mono(
            os.path.join(c.R56_WORK, "holdoutD_ref_warp.wav"), DUR))
        bands_ref = c.ref_bands_profile(
            os.path.join(c.R56_WORK, "holdoutD_ref_warp.wav"), DUR)
        music_hits, taiko_hits = 0, 0
        for t in on:
            k = int(round(t * SR))
            if len(ref_on) and np.min(np.abs(ref_on - t)) <= 0.030:
                music_hits += 1
                if bands_ref["bass"][k] >= 1.5 * (bands_ref["mid"][k] + bands_ref["hi1"][k]
                                                  + bands_ref["hi2"][k]):
                    taiko_hits += 1
        m["music_onset_matched"] = music_hits
        m["taiko_onset_matched"] = taiko_hits
        u_all, tot = c.useful_window_ratio(spans, list(tc))
        hi_med = [e["t_video"] for e in contacts if e["confidence"] in ("high", "med")]
        u_hm, _ = c.useful_window_ratio(spans, hi_med)
        u_hi, _ = c.useful_window_ratio(spans, [e["t_video"] for e in strong])
        m["useful_window_ratio_all"] = round(u_all, 3)
        m["useful_window_ratio_highmed"] = round(u_hm, 3)
        m["useful_window_ratio_high"] = round(u_hi, 3)
        m["interaction_coverage"] = round(float(np.mean(np.abs(stem) > 1e-6)), 3)
        mix, _ = sf.read(mix_p, dtype="float64", always_2d=True)
        om, _ = sf.read(os.path.join(c.R56_OUT, "holdoutD_oracle_C1_final_mix.wav"),
                        dtype="float64", always_2d=True)
        nn = min(len(mix), len(om))
        diff = mix[:nn].mean(axis=1) - om[:nn].mean(axis=1)
        m["auto_vs_oracle_final_rms_db"] = round(float(
            20 * np.log10(np.sqrt(np.mean(diff ** 2)) /
                          (np.sqrt(np.mean(om[:nn].mean(axis=1) ** 2)) + 1e-12))), 1)
        m["final_mix_peak_dbfs"] = round(float(
            20 * np.log10(np.max(np.abs(mix)) + 1e-12)), 2)
        rep[name] = m

    rep["completion"] = {"n_added": len(added),
                         "n_rejected": len(rejected),
                         "added": [{k: round(v, 3) if isinstance(v, float) else v
                                    for k, v in a.items() if k != "k"} for a in added]}
    c.save_json(os.path.join(c.R56_LOG, "holdoutD_metrics.json"), rep)

    print(f"\nE0 (R5.5 frozen): {rep['E0']['n_candidates']} cands")
    t5 = rep["E0"]["tol50"]
    print(f"  P {t5['P']:.3f} R {t5['R']:.3f} F1 {t5['F1']:.3f} | strong-R "
          f"{rep['E0']['strong_recall']:.3f} | gate {rep['E0']['gate_coverage']:.3f} "
          f"(strong {rep['E0']['gate_coverage_strong']}) | FW {rep['E0']['false_windows']['total_s']} s "
          f"| leak {rep['E0']['music_onset_matched']} | UWR {rep['E0']['useful_window_ratio_highmed']:.3f} "
          f"| rms {rep['E0']['auto_vs_oracle_final_rms_db']} dB")
    print(f"\nE2 (R5.6 frozen): {rep['E2']['n_candidates']} cands + {len(added)} completed")
    t5 = rep["E2"]["tol50"]
    print(f"  P {t5['P']:.3f} R {t5['R']:.3f} F1 {t5['F1']:.3f} | strong-R "
          f"{rep['E2']['strong_recall']:.3f} | gate {rep['E2']['gate_coverage']:.3f} "
          f"(strong {rep['E2']['gate_coverage_strong']}) | FW {rep['E2']['false_windows']['total_s']} s "
          f"| leak {rep['E2']['music_onset_matched']} taiko {rep['E2']['taiko_onset_matched']} "
          f"| UWR {rep['E2']['useful_window_ratio_highmed']:.3f} "
          f"| rms {rep['E2']['auto_vs_oracle_final_rms_db']} dB")


if __name__ == "__main__":
    main()
