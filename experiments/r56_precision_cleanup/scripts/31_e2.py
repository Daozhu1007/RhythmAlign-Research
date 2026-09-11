# R5.6 Phase 3b - E2 = E1 precision cleanup + burst-constrained strong-audio
# completion (the ONE recipe). Completion sources ONLY strong comb transients
# (frozen 'strong' tier: peak>=0.35 & >4x local median) inside the E0-frozen
# active bursts, >=90 ms from every E1 candidate and from each other, with a
# taiko guard requiring music-removed residual support (mid+hi1+hi2 >= 1.2
# and hi1 >= 0.15; bass-only taiko onsets fail hi1). Outside bursts: nothing
# is ever added. Hand motion is recorded as a bonus feature, never required.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import soundfile as sf
from scipy import signal as sig

import common56 as c
from detector55 import fused_score
from r5_common import SR, audio_onsets
from importlib import import_module

cal = import_module("30_completion_calibration")

E2_CFG = c.E2_CFG


def complete(w, e1_t):
    comb = c.get_comb(w)
    bands = c.band_envelopes(w)
    _, mask, ivs, env, _ = c.detect55(w.series, c.FPS, c.CFG55_FROZEN, rule="A")
    bursts_s = [(a / c.FPS, b / c.FPS) for a, b in ivs]
    hdn = np.asarray(w.series["hand_diff"])
    hs = np.percentile(hdn, 99.5) + 1e-9

    peaks = cal.strong_comb_peaks(comb, E2_CFG["min_peak"], E2_CFG["min_ratio"],
                                  E2_CFG["min_dist_s"])
    added, rejected = [], []
    for p in peaks:
        t, k = p["t"], p["k"]
        why = None
        if not any(a <= t <= b for a, b in bursts_s):
            why = "outside_burst"
        elif min([abs(t - x) for x in e1_t], default=10.0) < E2_CFG["d_e1_min_ms"] / 1000:
            why = "near_existing_candidate"
        elif min([abs(t - x["t"]) for x in added], default=10.0) < E2_CFG["d_e2_min_ms"] / 1000:
            why = "refractory"
        else:
            bs, h1, h2, ba = (float(bands["mid"][k]), float(bands["hi1"][k]),
                              float(bands["hi2"][k]), float(bands["bass"][k]))
            if bs + h1 + h2 < E2_CFG["resid_sum_min"] or h1 < E2_CFG["hi1_min"]:
                why = f"taiko_guard(bass {ba:.2f} resid {bs + h1 + h2:.2f} hi1 {h1:.2f})"
        if why:
            rejected.append({**p, "reason": why})
            continue
        ki = int(round(t * c.FPS))
        hand = float(min(np.max(hdn[max(ki - 3, 0):ki + 4]) / hs, 2.5))
        edge = min([min(t - a, b - t) for a, b in bursts_s if a <= t <= b]) * 1000
        added.append({**p, "hand": hand, "burst_edge_ms": edge,
                      "bass": float(bands["bass"][k]), "mid": float(bands["mid"][k]),
                      "hi1": float(bands["hi1"][k]), "hi2": float(bands["hi2"][k])})
    return added, rejected, bursts_s


def run_e2(w):
    # ---- E1 stage (identical to 20_e1) ----
    score, ns = fused_score(w.series, c.CFG55_FROZEN)
    _, mask, ivs, env, cands = c.detect55(w.series, c.FPS, c.CFG55_FROZEN, rule="A")
    refined = c.refine(w, [cd["t_video"] for cd in cands],
                       [f"{w.key}{j:03d}" for j in range(len(cands))])
    flags = c.e1_gate_flags(cands, refined, score, ns, ivs, c.E1_CFG)
    e1_t = [refined[j]["t_audio_refined"] for j in range(len(cands))
            if flags[j][1] and refined[j].get("refined_confidence")
            not in (None, "no_transient", "out_of_range")]

    # ---- completion stage ----
    added, rejected, bursts_s = complete(w, e1_t)

    ta = e1_t + [a["t"] for a in added]
    out_int = os.path.join(c.R56_OUT, f"{w.key}_E2_C1_interaction.wav")
    out_mix = os.path.join(c.R56_OUT, f"{w.key}_E2_C1_final_mix.wav")
    ref_full, _ = sf.read(w.ref, dtype="float64", always_2d=True)
    ref = ref_full[int(c.CTX_START * SR):int((c.CTX_START + w.dur) * SR)]
    c.build_c1(w.raw, ref, ta, out_int, out_mix)

    # ---- taiko / music-coincidence audit of ADDED events ----
    ref_full2, _ = sf.read(w.ref, dtype="float64", always_2d=True)
    refm = ref_full2[int(c.CTX_START * SR):int((c.CTX_START + w.dur) * SR)].mean(axis=1)
    ron = audio_onsets(refm)
    audit = []
    for a in added:
        d_music = float(np.min(np.abs(ron - a["t"]))) * 1000 if len(ron) else None
        audit.append({"t": round(a["t"], 3), "d_music_onset_ms": round(d_music, 1),
                      "bass": round(a["bass"], 2), "hi1": round(a["hi1"], 2),
                      "resid_sum": round(a["mid"] + a["hi1"] + a["hi2"], 2),
                      "hand": round(a["hand"], 2),
                      "coincides_music<=30ms": bool(d_music <= 30)})

    doc = {
        "experiment": "R5.6 E2 = E1 + burst-constrained strong-audio completion",
        "window": w.key, "e2_config": E2_CFG,
        "e1_candidates": len(e1_t), "n_added": len(added), "n_rejected": len(rejected),
        "added": [{k: round(v, 3) if isinstance(v, float) else v
                   for k, v in a.items() if k != "k"} for a in added],
        "rejected_sample": [{k: (round(v, 2) if isinstance(v, float) else v)
                             for k, v in a.items() if k != "k"} for a in rejected],
        "taiko_audit": audit,
    }
    c.save_json(os.path.join(c.R56_OUT, f"{w.key}_E2_contacts.json"), doc)
    return {"e1_windows": len(e1_t), "added": len(added),
            "rejected": len(rejected),
            "added_matched_oracle": sum(
                bool(len(w.tc()) and np.min(np.abs(w.tc() - a["t"])) <= 0.080)
                for a in added),
            "added_coincide_music": sum(x["coincides_music<=30ms"] for x in audit)}


def main():
    for w in c.load_dev_windows():
        s = run_e2(w)
        print(f"{w.key}: E1 windows {s['e1_windows']} + added {s['added']} "
              f"(of which matched oracle<=80ms: {s['added_matched_oracle']}; "
              f"coincide music onset<=30ms: {s['added_coincide_music']}) "
              f"| rejected {s['rejected']}")


if __name__ == "__main__":
    main()
