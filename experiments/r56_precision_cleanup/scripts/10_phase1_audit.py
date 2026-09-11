# R5.6 Phase 1 - echo / weak-candidate audit on Dev-A/B/C.
# E0 = R5.5 frozen detector (deterministic recompute, byte-identical config).
# For EVERY candidate collect: visual peak, prominence, prev-dist, echo ratio,
# refinement confidence, comb transient, burst state, burst-edge distance,
# per-channel support. Classify vs oracle (Dev-C only anchored on
# high-confidence oracle events + noncontact regions; unmatched candidates
# stay "unverified" because the Dev-C oracle is known incomplete).
# Answers, with numbers, the four Phase-1 questions BEFORE any rule is written.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import common56 as c
from detector55 import fused_score, _local_prom
import r5_common as r5

OUT = c.R56_LOG
ECHO_WIN_MS = 120.0        # trailing window for the echo ratio (prompt: 80-140)
SILENT_SCORE = 0.15        # fused score below this = visually silent (R5.5: FN median 0.07-0.08, thr 0.40)


def chan_support(ns, i, cols=("jz_text", "jz_diff", "glass_bright", "hand_diff")):
    sup = {}
    for name in cols:
        n = ns[name]
        scale = np.percentile(n, 99.5) + 1e-9
        sup[name] = float(min(n[i] / scale, 2.5))
    sup["n_ge_0.10"] = int(sum(v >= 0.10 for k, v in sup.items() if k != "n_ge_0.10"))
    return sup


def burst_pos(i, ivs):
    for a, b in ivs:
        if a <= i < b:
            return True, float(min(i - a, b - i) / c.FPS * 1000)
    d = min([min(abs(i - a), abs(i - b)) for a, b in ivs], default=10 ** 9)
    return False, float(d / c.FPS * 1000)


def audit_window(w):
    score, ns = fused_score(w.series, c.CFG55_FROZEN)
    _, mask, ivs, env, cands = c.detect55(w.series, c.FPS, c.CFG55_FROZEN, rule="A")
    comb = c.get_comb(w)
    bands = c.band_envelopes(w)

    # oracle refined by id
    oref = {e["id"]: e for e in c.load_json(w.oracle_refined)["events"]}
    tps_times = w.tc()

    tp_t = np.array([cd["t_video"] for cd in cands])
    mc, mp, pairs = c.match_one_to_one(tp_t, tps_times, 0.050)
    cand_oracle = {}
    for j, k, err in pairs:
        cand_oracle[j] = k
    tnc = np.array([e["t_video"] for e in w.noncontacts]) if w.noncontacts else np.array([])

    p995 = np.percentile(score, 99.5) + 1e-9
    rows = []
    for j, cd in enumerate(cands):
        i = cd["i"]
        pr = _local_prom(score, np.array([i]), 9)[0]
        in_b, edge_ms = burst_pos(i, ivs)
        # trailing strongest peak within ECHO_WIN_MS (excluding self)
        lo = max(int(i - ECHO_WIN_MS / 1000 * c.FPS), 0)
        prev_scores = [score[k] for k in range(lo, i)]
        prev_peak = max(prev_scores) if prev_scores else 0.0
        prev_dist = i - (lo + int(np.argmax(prev_scores))) if prev_scores else None
        r = c.refine(w, [cd["t_video"]], [f"{w.key}{j:03d}"])[0]
        k_or = cand_oracle.get(j)
        k_i = int(round(cd["t_video"] * c.FPS))
        near_oc = float(np.min(np.abs(tps_times - cd["t_video"]))) if len(tps_times) else None
        row = {
            "j": j, "id": cd.get("id", f"{w.key}{j:03d}"), "i": i, "t": cd["t_video"],
            "score": float(score[i]), "prominence": float(pr),
            "prom_frac_p995": float(pr / p995),
            "prev_peak": float(prev_peak),
            "prev_peak_dist_ms": float(prev_dist / c.FPS * 1000) if prev_dist is not None else None,
            "echo_ratio": float(score[i] / prev_peak) if prev_peak > 0 else None,
            "in_burst": in_b, "burst_edge_ms": edge_ms,
            "chan": chan_support(ns, i),
            "refined_confidence": r.get("refined_confidence"),
            "comb_peak": r.get("peak"), "comb_ratio": r.get("peak_over_noise"),
            "comb_shift_ms": r.get("refinement_shift_ms"),
            "is_tp": k_or is not None,
            "near_oracle_ms": round(near_oc * 1000, 1) if near_oc is not None else None,
            "near_noncontact": bool(len(tnc) and np.min(np.abs(tnc - cd["t_video"])) <= 0.100),
            "oracle_conf": w.contacts[k_or].get("confidence") if k_or is not None else None,
            "oracle_dense": bool(w.contacts[k_or]["dense"]) if k_or is not None else None,
            "in_gap": any(g["t0"] - 0.15 <= cd["t_video"] <= g["t1"] + 0.15
                          for g in w.regions if g["id"].startswith("G")),
        }
        # band profile at the refined audio time (audit for E2 design)
        ka = int(round(r.get("t_audio_refined", cd["t_video"]) * c.SR))
        if 0 <= ka < len(comb):
            row["bands"] = {b: round(float(v[min(ka, len(v) - 1)]), 3)
                            for b, v in bands.items()}
        rows.append(row)

    # ---- oracle side: per-contact detectability + audio ----
    orc_rows = []
    for e in w.contacts:
        k_i = int(round(e["t_video"] * c.FPS))
        sp = score[max(k_i - 2, 0):k_i + 3]
        s_peak = float(np.max(sp)) if len(sp) else 0.0
        ro = oref.get(e["id"])
        det = bool(len(tp_t) and np.min(np.abs(tp_t - e["t_video"])) <= 0.050)
        orc = {
            "id": e["id"], "t": e["t_video"], "type": e["type"],
            "conf": e.get("confidence"), "dense": bool(e["dense"]),
            "score_at": round(s_peak, 3), "detected": det,
            "silent": s_peak < SILENT_SCORE,
            "refined_confidence": ro.get("refined_confidence") if ro else None,
            "comb_peak": ro.get("peak") if ro else None,
            "comb_ratio": ro.get("peak_over_noise") if ro else None,
        }
        if ro and ro.get("t_audio_refined") is not None:
            ka = int(round(ro["t_audio_refined"] * c.SR))
            orc["bands"] = {b: round(float(v[min(ka, len(v) - 1)]), 3)
                            for b, v in bands.items()}
        orc_rows.append(orc)

    return {"cands": rows, "oracle": orc_rows, "n_bursts": len(ivs),
            "burst_intervals_s": [[round(a / c.FPS, 3), round(b / c.FPS, 3)] for a, b in ivs]}


def summarize(key, aud):
    cands, orc = aud["cands"], aud["oracle"]
    complete = key in ("devA", "devB")
    tp = [x for x in cands if x["is_tp"]]
    fp = [x for x in cands if not x["is_tp"]]
    fp_echo = [x for x in fp if x["near_oracle_ms"] is not None and x["near_oracle_ms"] <= 150]
    fp_conf = [x for x in fp if x["near_noncontact"]]
    fp_other = [x for x in fp if x not in fp_echo and x not in fp_conf]

    def q(vals, names=("p10", "p25", "med", "p75", "p90")):
        if not len(vals):
            return {n: None for n in names}
        a = np.array(vals)
        return dict(zip(names, [round(float(np.percentile(a, p)), 3)
                                for p in (10, 25, 50, 75, 90)]))

    # true adjacent contacts: TP whose previous candidate is 60-140 ms earlier
    def prev_gap(x, rows):
        ts = sorted(r["t"] for r in rows if r["t"] < x["t"])
        return (x["t"] - ts[-1]) * 1000 if ts else None

    adj_tp = []
    for x in tp:
        g = prev_gap(x, cands)
        if g is not None and 60 <= g <= 140:
            r = dict(x)
            r["prev_gap_ms"] = g
            adj_tp.append(r)
    doubles = [x for x in adj_tp if 90 <= x["prev_gap_ms"] <= 120]

    s = {"n_cands": len(cands), "n_tp": len(tp), "n_fp": len(fp),
         "fp_echo_le150": len(fp_echo), "fp_confirmed_noncontact": len(fp_conf),
         "fp_other": len(fp_other) if complete else f"{len(fp_other)} (unverified)"}
    s["echo_ratio_q"] = {
        "echo_fp": q([x["echo_ratio"] for x in fp_echo if x["echo_ratio"] is not None]),
        "true_adjacent_tp": q([x["echo_ratio"] for x in adj_tp if x["echo_ratio"] is not None]),
        "all_tp": q([x["echo_ratio"] for x in tp if x["echo_ratio"] is not None]),
        "fp_confirmed": q([x["echo_ratio"] for x in fp_conf if x["echo_ratio"] is not None]),
    }
    s["real_doubles_90_120ms"] = {"n": len(doubles),
                                  "ratios": sorted(round(x["echo_ratio"], 2) for x in doubles
                                                   if x["echo_ratio"] is not None)}
    # echo-ratio threshold sweep: keep if echo_ratio >= thr OR no prev peak
    sweep = {}
    for thr in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60):
        kept = lambda x: x["echo_ratio"] is None or x["echo_ratio"] >= thr
        sweep[str(thr)] = {
            "echo_fp_removed": f"{sum(not kept(x) for x in fp_echo)}/{len(fp_echo)}",
            "true_adj_killed": f"{sum(not kept(x) for x in adj_tp)}/{len(adj_tp)}",
            "doubles_killed": f"{sum(not kept(x) for x in doubles)}/{len(doubles)}",
            "all_tp_killed": f"{sum(not kept(x) for x in tp)}/{len(tp)}",
        }
    s["echo_sweep"] = sweep
    # refinement confidence vs correctness
    conf_tab = {}
    for lvl in ("strong", "present", "weak", "no_transient"):
        xs = [x for x in cands if x["refined_confidence"] == lvl]
        conf_tab[lvl] = {"n": len(xs),
                         "tp": sum(x["is_tp"] for x in xs),
                         "fp": sum(not x["is_tp"] for x in xs)}
    s["confidence_table"] = conf_tab
    if complete:
        s["conf_precision"] = {k: (v["tp"] / v["n"] if v["n"] else None)
                               for k, v in conf_tab.items()}
    # oracle side
    silent = [x for x in orc if x["silent"]]
    s["oracle"] = {
        "n": len(orc), "n_detected": sum(x["detected"] for x in orc),
        "n_silent": len(silent),
        "silent_detected": sum(x["detected"] for x in silent),
        "silent_refined_conf": {k: sum(1 for x in silent
                                       if x["refined_confidence"] == k)
                                for k in ("strong", "present", "weak", "no_transient", None)},
        "nonsilent_refined_conf": {k: sum(1 for x in orc if not x["silent"]
                                          and x["refined_confidence"] == k)
                                   for k in ("strong", "present", "weak", "no_transient", None)},
        "silent_bands_med": {b: (round(float(np.median([x["bands"][b] for x in silent
                                                        if x.get("bands")])), 3)
                                if any(x.get("bands") for x in silent) else None)
                             for b in ("bass", "mid", "hi1", "hi2")},
        "nonsilent_bands_med": {b: (round(float(np.median([x["bands"][b] for x in orc
                                                           if not x["silent"] and x.get("bands")])), 3)
                                    if any(not x["silent"] and x.get("bands") for x in orc) else None)
                                for b in ("bass", "mid", "hi1", "hi2")},
    }
    hi = [x for x in orc if x["conf"] == "high"]
    s["oracle_high"] = {"n": len(hi), "detected": sum(x["detected"] for x in hi)}
    return s


def main():
    rep = {}
    for w in c.load_dev_windows():
        aud = audit_window(w)
        s = summarize(w.key, aud)
        rep[w.key] = s
        c.save_json(os.path.join(OUT, f"phase1_audit_{w.key}.json"),
                    {"summary": s, **aud})
        print(f"\n===== {w.key} =====")
        print(f"  cands {s['n_cands']} | TP {s['n_tp']} | FP {s['n_fp']} "
              f"(echo<=150ms {s['fp_echo_le150']}, confirmed-noncontact {s['fp_confirmed_noncontact']})")
        print(f"  echo_ratio med: echo_fp {s['echo_ratio_q']['echo_fp']['med']} vs "
              f"true-adjacent {s['echo_ratio_q']['true_adjacent_tp']['med']} "
              f"(all TP {s['echo_ratio_q']['all_tp']['med']})")
        for thr, v in s["echo_sweep"].items():
            print(f"    thr {thr}: echoFP- {v['echo_fp_removed']}  adjTP- {v['true_adj_killed']}  "
                  f"doubles- {v['doubles_killed']}  TP- {v['all_tp_killed']}")
        print(f"  conf table: " + " | ".join(
            f"{k}:{v['n']} (tp {v['tp']})" for k, v in s["confidence_table"].items()))
        o = s["oracle"]
        print(f"  oracle: {o['n_detected']}/{o['n']} detected | silent {o['n_silent']} "
              f"(of which detected {o['silent_detected']}) | silent audio conf {o['silent_refined_conf']}")
        print(f"  silent bands med {o['silent_bands_med']} | nonsilent {o['nonsilent_bands_med']}")
    c.save_json(os.path.join(OUT, "phase1_audit_summary.json"), rep)


if __name__ == "__main__":
    main()
