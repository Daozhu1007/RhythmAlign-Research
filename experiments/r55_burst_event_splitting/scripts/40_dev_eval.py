# R5.5 Phase 4 - full development evaluation of the chosen detector
# (rule A, burst params from 20_burst_and_rules) on Dev-A + Dev-B:
#   event P/R/F1 @+/-33/50 ms overall + SPARSE/DENSE split,
#   strong recall, timing, region coverage, FP/FN;
#   frozen audio: refinement + C1 builds, gate coverage, false windows,
#   auto-vs-oracle final mix RMS diff.
# R5 baseline numbers are recomputed with the SAME eval code for fairness.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import soundfile as sf

from common55 import (R55_OUT, R55_LOG, R55_WORK, R5_OUT, R5_WORK, R1_WORK,
                      R4_OUT, load_windows, save_json, load_json, FPS,
                      match_one_to_one)
from r5_common import SR
R1_OUT = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs"
from detector55 import DEFAULT55, detect55
import r5_common as r5

AV_OFF_MS = 0.0
CTX_START = 3.0
CFG55 = {**DEFAULT55, "burst_enter": 0.40, "burst_stay": 0.22,
         "burst_hold_frames": 30, "a_prom_frac": 0.03, "a_min_dist_frames": 3}

# per-window audio sources (all frozen from R1/R4/R5)
AUDIO = {
    "devA": {"ctx": os.path.join(R1_WORK, "golden_ctx.wav"),
             "ref": os.path.join(R1_WORK, "ref_warp_fixed.wav"), "dur": 15.0,
             "oracle_refined": os.path.join(R4_OUT, "contacts_refined.json")},
    "devB": {"ctx": os.path.join(R5_WORK, "holdout_ctx.wav"),
             "ref": os.path.join(R5_WORK, "holdout_ref_warp.wav"), "dur": 18.0,
             "oracle_refined": os.path.join(R5_OUT, "holdout_oracle_refined.json")},
}
RAW = {"devA": os.path.join(R1_OUT, "golden_raw.wav"),
       "devB": os.path.join(R5_OUT, "holdout_raw.wav")}


def zone_of(w, t, i=None):
    k = int(round(t * FPS)) if i is None else i
    return "dense" if (0 <= k < w.n and w.dense_mask[k]) else "sparse"


def eval_full(cands, w):
    tp_t = np.array([c["t_video"] for c in cands])
    tc = w.tc()
    out = {"n_pred": len(cands), "n_oracle": len(tc)}
    for tol in (0.033, 0.050):
        mc, mp, pairs = match_one_to_one(tp_t, tc, tol)
        p = len(mc) / max(len(cands), 1)
        r = len(mc) / max(len(tc), 1)
        errs = [abs(e) * 1000 for _, _, e in pairs]
        out[f"tol{int(tol*1000)}"] = {
            "P": round(p, 3), "R": round(r, 3),
            "F1": round(2 * p * r / max(p + r, 1e-9), 3),
            "err_ms_median": round(float(np.median(errs)), 1) if errs else None,
            "err_ms_p95": round(float(np.percentile(errs, 95)), 1) if errs else None}
    # sparse / dense split (predictions assigned by zone, contacts by flag)
    for zone in ("sparse", "dense"):
        cs = [c for c in cands if zone_of(w, c["t_video"], c["i"]) == zone]
        es = [e for e in w.contacts if e["dense"] == (zone == "dense")]
        tcs = np.array([e["t_video"] for e in es])
        tps = np.array([c["t_video"] for c in cs])
        m, _, _ = match_one_to_one(tps, tcs, 0.050) if len(tcs) else (set(), set(), [])
        p = len(m) / max(len(cs), 1)
        r = len(m) / max(len(es), 1)
        out[zone] = {"n_pred": len(cs), "n_oracle": len(es),
                     "P": round(p, 3), "R": round(r, 3),
                     "F1": round(2 * p * r / max(p + r, 1e-9), 3)}
    strong = [e for e in w.contacts if e.get("confidence") == "high"]
    ts = np.array([e["t_video"] for e in strong])
    hit = [t for t in ts if len(tp_t) and np.min(np.abs(tp_t - t)) <= 0.050]
    out["strong"] = {"n": len(ts), "recall_50ms": round(len(hit) / max(len(ts), 1), 3)}
    # FN list with zones / types
    mc, mp, pairs = match_one_to_one(tp_t, tc, 0.050)
    matched_o = {k for _, k, _ in pairs}
    out["fn"] = [{"id": w.contacts[k]["id"], "t": w.contacts[k]["t_video"],
                  "type": w.contacts[k]["type"],
                  "zone": "dense" if w.contacts[k]["dense"] else "sparse"}
                 for k in range(len(tc)) if k not in matched_o]
    matched_p = mp
    tnc = np.array([e["t_video"] for e in w.noncontacts]) if w.noncontacts else np.array([])
    out["fp"] = [{"t": round(float(tp_t[j]), 3)} for j in range(len(tp_t))
                 if j not in matched_p]
    for f in out["fp"]:
        f["near_noncontact"] = bool(len(tnc) and np.min(np.abs(tnc - f["t"])) <= 0.100)
    # region coverage (R5 definition: holes > 0.3 s inside regions)
    miss_tot, reg_tot = 0.0, 0.0
    for rg in w.continuous:
        a, b = rg["t0"], rg["t1"]
        holes, t = 0.0, a
        while t < b:
            if len(tp_t) and np.min(np.abs(tp_t - t)) <= 0.3:
                nxt = tp_t[np.argmin(np.abs(tp_t - t))] + 0.3
                t = max(t + 1 / FPS, min(nxt, b))
            else:
                t2 = t
                while t2 < b and not (len(tp_t) and np.min(np.abs(tp_t - t2)) <= 0.3):
                    t2 += 1 / FPS
                holes += min(t2, b) - t
                t = t2
        miss_tot += holes
        reg_tot += b - a
    out["region_coverage"] = round(1 - miss_tot / reg_tot, 4)
    return out


def audio_eval(w, cands, tag):
    """Frozen refinement + C1 on this window; returns audio metrics dict and
    writes interaction/final wavs for AUTO (oracle wavs reused from R5/R4)."""
    src = AUDIO[w.key]
    comb, _ = r5.comb_envelope(src["ctx"], src["ref"], CTX_START, src["dur"])
    dur = src["dur"]
    ids = [f"{tag}{i:03d}" for i in range(len(cands))]
    refined = r5.refine_candidates([c["t_video"] for c in cands], comb,
                                   AV_OFF_MS, dur, ids=ids)
    for c, r in zip(cands, refined):
        r["score"] = c["score"]
    save_json(os.path.join(R55_OUT, f"{w.key}_{tag}_refined.json"),
              {"av_off_ms": AV_OFF_MS, "n_candidates": len(cands), "events": refined})
    auto_used = [r for r in refined
                 if r.get("refined_confidence") not in (None, "no_transient", "out_of_range")]
    # oracle refined (frozen file from R4/R5, same algorithm)
    oref = load_json(src["oracle_refined"])["events"]
    orc = [e for e in oref
           if e.get("refined_confidence") not in (None, "no_transient", "out_of_range")]
    to = np.array([e["t_audio_refined"] for e in orc])
    ta = np.array([r["t_audio_refined"] for r in auto_used])
    # one-to-one match at 80 ms
    pairs = sorted(((abs(a - b), i, j) for i, a in enumerate(ta) for j, b in enumerate(to)),
                   key=lambda x: x[0])
    ua, uo, errs = set(), set(), []
    for d, i, j in pairs:
        if d > 0.080:
            break
        if i in ua or j in uo:
            continue
        ua.add(i)
        uo.add(j)
        errs.append((ta[i] - to[j]) * 1000)
    errs = np.array(errs) if errs else np.array([0.0])
    # gate coverage of oracle by AUTO windows
    spans = []
    for r in auto_used:
        t = r["t_audio_refined"]
        spans.append((max(t - 0.015, 0.0), min(t + 0.150, dur)))
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    def covered(t):
        return any(s <= t <= e for s, e in merged)
    gate = float(np.mean([covered(t) for t in to]))
    strong_ev = [e for e in orc if e.get("refined_confidence") == "strong"]
    gate_s = float(np.mean([covered(e["t_audio_refined"]) for e in strong_ev])) if strong_ev else None
    # false windows: auto spans with no oracle contact within 80 ms of center
    fw = [(round(s, 2), round(e, 2)) for s, e in merged
          if len(to) and np.min(np.abs(to - (s + e) / 2)) > 0.080]
    # C1 builds
    ref_full, _ = sf.read(src["ref"], dtype="float64", always_2d=True)
    ref = ref_full[int(CTX_START * SR):int((CTX_START + dur) * SR)]
    raw_path = RAW[w.key]
    auto_int = os.path.join(R55_OUT, f"{w.key}_{tag}_C1_interaction.wav")
    auto_mix = os.path.join(R55_OUT, f"{w.key}_{tag}_C1_final_mix.wav")
    r5.build_c1(raw_path, ref, ta, auto_int, auto_mix)
    orc_int = os.path.join(R55_OUT, f"{w.key}_oracle_C1_interaction.wav")
    orc_mix = os.path.join(R55_OUT, f"{w.key}_oracle_C1_final_mix.wav")
    r5.build_c1(raw_path, ref, to, orc_int, orc_mix)
    # final mix difference
    a, _ = sf.read(auto_mix, dtype="float64", always_2d=True)
    o, _ = sf.read(orc_mix, dtype="float64", always_2d=True)
    diff = a.mean(axis=1) - o.mean(axis=1)
    rms_db = round(float(20 * np.log10(np.sqrt(np.mean(diff ** 2)) /
                                       (np.sqrt(np.mean(o.mean(axis=1) ** 2)) + 1e-12))), 1)
    return {
        "n_auto_used": len(auto_used), "n_oracle_refined": int(len(to)),
        "matched_80ms": len(uo),
        "err_ms_median": round(float(np.median(errs)), 1),
        "abs_err_ms_p95": round(float(np.percentile(np.abs(errs), 95)), 1),
        "gate_coverage_of_oracle": round(gate, 3),
        "gate_coverage_strong": round(gate_s, 3) if gate_s is not None else None,
        "false_windows": {"count": len(fw), "total_s": round(sum(e - s for s, e in fw), 2)},
        "auto_vs_oracle_final_rms_db": rms_db,
    }


def main():
    windows = load_windows()
    all_metrics = {}
    for w in windows:
        # ---- R5.5 detector ----
        score, mask, ivs, env, cands = detect55(w.series, FPS, CFG55, rule="A")
        save_json(os.path.join(R55_OUT, f"{w.key}_r55_contacts_visual.json"),
                  {"detector": "R5.5 rule A", "config": CFG55,
                   "n_bursts": len(ivs),
                   "burst_intervals": [[round(a / FPS, 3), round(b / FPS, 3)] for a, b in ivs],
                   "n_candidates": len(cands), "candidates": cands})
        m55 = eval_full(cands, w)
        m55["audio"] = audio_eval(w, cands, "r55auto")
        # ---- R5 baseline with identical eval code ----
        score0, cands0 = r5.detect_candidates(w.series, FPS, {
            "smooth_frames": 3, "rise_frames": 2,
            "w_jz_text": 1.8, "w_jz_diff": 1.5, "w_glass_bright": 1.0,
            "w_hand_diff": 0.3, "threshold": 0.40, "slope_min": 0.15,
            "min_dist_frames": 3, "norm_pct": 99.5, "novelty_clip": 2.5})
        m0 = eval_full(cands0, w)
        all_metrics[w.key] = {"r55": m55, "r5": m0}

        print(f"\n===== {w.key} =====")
        for name, m in (("R5.5", m55), ("R5 ", m0)):
            t5, t3 = m["tol50"], m["tol33"]
            print(f"  {name}: pred {m['n_pred']:3d} | @50 P {t5['P']:.3f} R {t5['R']:.3f} "
                  f"F1 {t5['F1']:.3f} | @33 F1 {t3['F1']:.3f} | err med {t5['err_ms_median']} "
                  f"p95 {t5['err_ms_p95']} ms")
            print(f"        sparse: {m['sparse']['n_oracle']:2d} oracle / P {m['sparse']['P']:.3f} "
                  f"R {m['sparse']['R']:.3f} F1 {m['sparse']['F1']:.3f} | "
                  f"dense: {m['dense']['n_oracle']:2d} / P {m['dense']['P']:.3f} "
                  f"R {m['dense']['R']:.3f} F1 {m['dense']['F1']:.3f}")
            print(f"        strong {m['strong']['recall_50ms']:.3f} ({m['strong']['n']}) "
                  f"| regcov {m['region_coverage']:.3f} | FP {len(m['fp'])} FN {len(m['fn'])}")
        a = m55["audio"]
        print(f"  R5.5 audio: gate {a['gate_coverage_of_oracle']:.1%} "
              f"(strong {a['gate_coverage_strong']:.1%}) | false windows {a['false_windows']['count']} "
              f"({a['false_windows']['total_s']} s) | mix RMS diff {a['auto_vs_oracle_final_rms_db']} dB "
              f"| |err| med {a['err_ms_median']} ms")
    save_json(os.path.join(R55_LOG, "phase4_dev_eval.json"), all_metrics)


if __name__ == "__main__":
    main()
