# R5-FIX PART 4 - clean-room evaluator. Historical evaluator files are NOT
# modified; this module re-implements every metric with corrected definitions:
#   4A optimal one-to-one event matching @ 33/50/80 ms (greedy comparison kept
#      only to document the historical loss at 80 ms)
#   4B VISUAL_HIGH (annotation confidence) vs AUDIO_STRONG (refinement
#      contrast) reported separately; neither is interpreted as physical force
#   4C gate support measured on the ACTUAL generated gain envelope
#   4D known-rest exposure reported directly (not hidden in aggregates)
#   4E false exposure = direct envelope overlap with predeclared
#      allowed-contact / known-rest / unlabelled support partitions
#   4F terminology: reference_onset_coincidence (+ low-band variant), with the
#      explicit statement that it does NOT identify neighbouring Taiko
#   4G dynamics: raw-vs-output event amplitude correlations; the invalid
#      corr(peak_ratio, raw_peak) is NOT computed; missed events stay in
#   4H unmatched output onsets use output-onset identities + one-to-one
#      matching; named unmatched_detected_output_onsets (never "hallucinations")
#   4I final-mix RMS = secondary diagnostic only (shared pristine reference
#      dominates it); interaction-stem differences reported separately
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixenv as fx

SR = fx.SR
BIG = 1e3


# ---------------------------------------------------------------- 4A matching
def match_one_to_one_optimal(pred, oracle, tol):
    """Maximum-cardinality, minimum-total-|dt| one-to-one matching."""
    pred = np.asarray(pred, float)
    oracle = np.asarray(oracle, float)
    if len(pred) == 0 or len(oracle) == 0:
        return [], pred, oracle
    dt = np.abs(pred[:, None] - oracle[None, :])
    C = np.where(dt <= tol, dt, BIG)
    ri, ci = linear_sum_assignment(C)
    pairs = [(int(i), int(j), float(pred[i] - oracle[j]))
             for i, j in zip(ri, ci) if C[i, j] < BIG]
    return pairs, pred, oracle


def prf_from_pairs(pairs, n_pred, n_oracle):
    m = len(pairs)
    p = m / max(n_pred, 1)
    r = m / max(n_oracle, 1)
    return {"matches": m, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / max(p + r, 1e-9), 4),
            "unmatched_predictions": int(n_pred - m),
            "unmatched_oracle_events": int(n_oracle - m)}


def match_greedy_historical(pred, oracle, tol):
    """Historical greedy (common55.match_one_to_one policy), only for the
    80-ms comparability note."""
    pred, oracle = np.asarray(pred, float), np.asarray(oracle, float)
    used_o, n = set(), 0
    for j, t in enumerate(pred):
        d = np.abs(oracle - t)
        for k in np.argsort(d)[:4]:
            if d[k] <= tol and k not in used_o:
                used_o.add(k)
                n += 1
                break
    return n


def event_matching_block(pred, oracle, tols=(0.033, 0.050, 0.080)):
    out = {}
    for tol in tols:
        pairs, _, _ = match_one_to_one_optimal(pred, oracle, tol)
        blk = prf_from_pairs(pairs, len(pred), len(oracle))
        blk["abs_err_ms_median"] = (round(float(np.median(np.abs([e for _, _, e in pairs]))) * 1000, 1)
                                    if pairs else None)
        blk["abs_err_ms_p95"] = (round(float(np.percentile(np.abs([e for _, _, e in pairs]), 95)) * 1000, 1)
                                 if pairs else None)
        if tol == 0.080:
            blk["historical_greedy_matches_for_note"] = match_greedy_historical(pred, oracle, tol)
            blk["note"] = ("historical greedy matching is known to lose matches at +/-80 ms "
                           "(this dataset: greedy {g} vs optimal {o})".format(
                               g=blk["historical_greedy_matches_for_note"], o=blk["matches"]))
        out[f"tol{int(tol*1000)}ms"] = blk
    return out


# ------------------------------------------------------------- support spans
def open_spans(env, thr=0.0, closed=False):
    """Contiguous regions where env > thr (closed=True -> >= thr)."""
    k = (env >= thr) if closed else (env > thr)
    k = k.astype(int)
    d = np.diff(np.concatenate([[0], k, [0]]))
    return [(int(s), int(e)) for s, e in zip(np.where(d > 0)[0], np.where(d < 0)[0])]


def span_stats(spans):
    durs = np.array([(e - s) / SR for s, e in spans], float)
    return {"count": int(len(spans)),
            "total_s": round(float(durs.sum()), 4),
            "duration_s_median": round(float(np.median(durs)), 4) if len(durs) else 0.0,
            "duration_s_p90": round(float(np.percentile(durs, 90)), 4) if len(durs) else 0.0,
            "duration_s_max": round(float(durs.max()), 4) if len(durs) else 0.0,
            "durations_s": [round(float(x), 4) for x in durs]}


def gate_support_block(env):
    sp_open = open_spans(env, 0.0)
    sp_half = open_spans(env, 0.5, closed=True)
    return {
        "mean_gain": round(float(env.mean()), 5),
        "nonzero_coverage_frac": round(float(np.mean(env > 0)), 5),
        "coverage_gain_ge_0p5_frac": round(float(np.mean(env >= 0.5)), 5),
        "longest_open_span_s": round(span_stats(sp_open)["duration_s_max"], 4),
        "longest_open_span_ge_0p5_s": round(span_stats(sp_half)["duration_s_max"], 4),
        "open_spans_gain_gt_0": span_stats(sp_open),
        "open_spans_gain_ge_0p5": {k: v for k, v in span_stats(sp_half).items() if k != "durations_s"},
    }


def contact_support_block(env, times, ids=None):
    """gain at each known contact + local maxima (4C)."""
    n = len(env)
    rows, low = [], []
    for i, t in enumerate(times):
        k = int(round(t * SR))
        lo, hi = max(k - int(0.050 * SR), 0), min(k + int(0.050 * SR) + 1, n)
        w20 = env[max(k - int(0.020 * SR), 0):min(k + int(0.020 * SR) + 1, n)]
        g_t = float(env[k]) if 0 <= k < n else 0.0
        row = {"i": i, "t": round(float(t), 4), "gain_at_contact": round(g_t, 4),
               "max_gain_pm20ms": round(float(w20.max()) if len(w20) else 0.0, 4),
               "max_gain_pm50ms": round(float(env[lo:hi].max()) if hi > lo else 0.0, 4)}
        if ids is not None:
            row["id"] = ids[i]
        rows.append(row)
        if row["max_gain_pm50ms"] < 0.5:
            low.append(row)
    gains = np.array([r["gain_at_contact"] for r in rows]) if rows else np.array([0.0])
    return {
        "n_contacts": len(times),
        "gain_at_contact_median": round(float(np.median(gains)), 4),
        "gain_at_contact_min": round(float(gains.min()), 4),
        "n_contacts_gain_lt_0p5": int(np.sum(gains < 0.5)),
        "n_contacts_max50_lt_0p5": len(low),
        "contacts_with_support_below_0p5": low,
        "per_contact": rows,
    }


# ---------------------------------------------------------------- 4D/4E
def exposure_partition_block(env, allowed_spans, rest_spans, stem=None, raw=None):
    n = len(env)
    def merged_union(spans_s):
        return fx.merge_spans([[max(int(s * SR), 0), min(int(e * SR), n)] for s, e in spans_s])
    a_sp = merged_union(allowed_spans)
    r_sp = merged_union(rest_spans)
    a_mask = np.zeros(n, bool)
    for s, e in a_sp:
        a_mask[s:e] = True
    r_mask = np.zeros(n, bool)
    for s, e in r_sp:
        r_mask[s:e] = True
    u_mask = ~(a_mask | r_mask)

    def sec(mask, thr, closed):
        k = (env >= thr) if closed else (env > thr)
        return round(float(np.mean(k & mask) * (n / SR)), 4)

    def mgain(mask):
        return round(float(env[mask].mean()) if np.any(mask) else 0.0, 5)

    out = {
        "allowed_contact_support": {
            "definition": "union over ok oracle contacts of [t-40ms, t+175ms] "
                          "(C1 pre/post + 25 ms A/V mapping uncertainty, R4 metadata)",
            "spans_s": [[round(s / SR, 4), round(e / SR, 4)] for s, e in a_sp],
            "seconds_gain_gt_0": sec(a_mask, 0.0, False),
            "seconds_gain_ge_0p5": sec(a_mask, 0.5, True),
            "mean_gain": mgain(a_mask),
        },
        "known_rest_support": {
            "spans_s": [[round(s / SR, 4), round(e / SR, 4)] for s, e in r_sp],
            "seconds_gain_gt_0": sec(r_mask, 0.0, False),
            "seconds_gain_ge_0p5": sec(r_mask, 0.5, True),
            "mean_gain": mgain(r_mask),
        },
        "unlabelled_support": {
            "seconds_gain_gt_0": sec(u_mask, 0.0, False),
            "seconds_gain_ge_0p5": sec(u_mask, 0.5, True),
            "mean_gain": mgain(u_mask),
            "definition": "all samples outside allowed-contact and known-rest support",
        },
    }
    if r_sp:
        r_slice = np.zeros(n, bool)
        for s, e in r_sp:
            r_slice[s:e] = True
        sp = open_spans(env[r_slice], 0.0)
        out["known_rest_support"]["longest_open_interval_s"] = round(
            max(((e - s) for s, e in sp), default=0.0) / SR, 4)
    if stem is not None and raw is not None and r_sp:
        sl = r_slice
        s_rms = float(np.sqrt(np.mean(stem[sl].astype(np.float64) ** 2)))
        r_rms = float(np.sqrt(np.mean(raw[sl].astype(np.float64) ** 2)))
        out["known_rest_support"]["interaction_rms_dbfs"] = round(20 * np.log10(s_rms + 1e-12), 2)
        out["known_rest_support"]["energy_change_vs_raw_db"] = (
            round(20 * np.log10(s_rms / (r_rms + 1e-12)), 3) if np.any(stem[sl]) else None)
    return out


# ------------------------------------------------------------- onsets (4F/4H)
def audio_onsets(y):
    """Frozen R4-family onset detector (r5_common.audio_onsets), unchanged."""
    R5_SCRIPTS = os.path.join(fx.ROOT, "experiments", "r5_auto_contact_detection", "scripts")
    if R5_SCRIPTS not in sys.path:
        sys.path.insert(0, R5_SCRIPTS)
    import r5_common
    return np.asarray(r5_common.audio_onsets(y))


def coincidence_block(stem_onsets, ref_onsets, ref_bands, tol=0.030):
    """4F: name and framing corrected; this does NOT identify neighbouring Taiko."""
    if len(stem_onsets) == 0 or len(ref_onsets) == 0:
        return {"reference_onset_coincidence": 0,
                "reference_lowband_onset_coincidence": 0,
                "n_output_onsets": int(len(stem_onsets)),
                "caveat": ("reference-onset coincidence counts output onsets within +/-30 ms of "
                           "PRISTINE-MUSIC onsets; it does NOT identify neighbouring Taiko and "
                           "must not be read as source-specific Taiko suppression.")}
    pairs, _, _ = match_one_to_one_optimal(stem_onsets, ref_onsets, tol)
    low = 0
    for _, j, _ in pairs:
        k = int(round(ref_onsets[j] * SR))
        b = {name: float(v[k]) for name, v in ref_bands.items()}
        if b["bass"] >= 1.5 * (b["mid"] + b["hi1"] + b["hi2"]):
            low += 1
    return {"reference_onset_coincidence": int(len(pairs)),
            "reference_lowband_onset_coincidence": int(low),
            "n_output_onsets": int(len(stem_onsets)),
            "caveat": ("reference-onset coincidence counts output onsets within +/-30 ms of "
                       "PRISTINE-MUSIC onsets; it does NOT identify neighbouring Taiko and "
                       "must not be read as source-specific Taiko suppression.")}


def output_onsets_block(stem, oracle_times, label_prefix=""):
    """4H: unmatched output onsets from ACTUAL output-onset identities."""
    mono = stem.mean(axis=1) if stem.ndim == 2 else stem
    on = audio_onsets(mono) if np.any(mono) else np.array([])
    blk = {"n_detected_output_onsets": int(len(on))}
    for tol in (0.033, 0.050, 0.080):
        pairs, _, _ = match_one_to_one_optimal(on, np.asarray(oracle_times, float), tol)
        matched = sorted(i for i, _, _ in pairs)
        unmat = [round(float(on[i]), 4) for i in range(len(on)) if i not in set(matched)]
        blk[f"tol{int(tol*1000)}ms"] = {
            **prf_from_pairs(pairs, len(on), len(oracle_times)),
            "unmatched_detected_output_onsets_s": unmat[:300],
        }
    return blk


# ---------------------------------------------------------------- 4G dynamics
def dynamics_block(stem, raw, times, env, ids=None):
    n = len(raw)
    rows = []
    for i, t in enumerate(times):
        k0, k1 = max(int((t - 0.015) * SR), 0), min(int((t + 0.150) * SR) + 1, n)
        mono = stem.mean(axis=1) if stem.ndim == 2 else stem
        rawn = raw.mean(axis=1) if raw.ndim == 2 else raw
        k = int(round(t * SR))
        w50 = env[max(k - int(0.050 * SR), 0):min(k + int(0.050 * SR) + 1, n)]
        open_ev = bool(len(w50) and w50.max() >= 0.5)
        rp = float(np.max(np.abs(rawn[k0:k1]))) if k1 > k0 else 0.0
        op = float(np.max(np.abs(mono[k0:k1]))) if k1 > k0 else 0.0
        row = {"i": i, "t": round(float(t), 4), "raw_event_peak": round(rp, 5),
               "output_event_peak": round(op, 5),
               "ratio": round(op / rp, 4) if rp > 1e-9 else None,
               "abs_log_level_error_db": round(abs(20 * np.log10(op / rp)), 2) if rp > 1e-9 and op > 1e-12 else None,
               "gate_open": open_ev}
        if ids is not None:
            row["id"] = ids[i]
        rows.append(row)
    all_rows = [r for r in rows if r["raw_event_peak"] > 1e-9]
    open_rows = [r for r in all_rows if r["gate_open"]]
    missed = [r for r in rows if not r["gate_open"]]

    def corr(rows_, kind):
        if len(rows_) < 3:
            return None
        r = np.array([x["raw_event_peak"] for x in rows_])
        o = np.array([x["output_event_peak"] for x in rows_])
        if kind == "pearson":
            if r.std() < 1e-12 or o.std() < 1e-12:
                return None
            v = float(np.corrcoef(r, o)[0, 1])
        else:
            from scipy.stats import spearmanr
            v = float(spearmanr(r, o).statistic)
        return None if not np.isfinite(v) else round(v, 4)

    ratios = [r["ratio"] for r in open_rows if r["ratio"] is not None]
    return {
        "definition": ("per oracle event over [t-15ms, t+150ms]: raw peak vs output peak; "
                       "corr is amplitude-vs-amplitude (NOT the invalid corr(peak_ratio, raw_peak)); "
                       "missed events keep their (near-zero) output peaks and are listed explicitly. "
                       "This measures level behaviour, NOT physical force."),
        "n_events": len(rows),
        "n_open": len(open_rows),
        "n_missed_from_gate": len(missed),
        "missed_events": [{"id": r.get("id"), "t": r["t"], "output_event_peak": r["output_event_peak"]}
                          for r in missed],
        "median_ratio_open": round(float(np.median(ratios)), 4) if ratios else None,
        "pearson_raw_vs_output_all_incl_missed": corr(all_rows, "pearson"),
        "pearson_raw_vs_output_open_only": corr(open_rows, "pearson"),
        "spearman_raw_vs_output_all_incl_missed": corr(all_rows, "spearman"),
        "spearman_raw_vs_output_open_only": corr(open_rows, "spearman"),
    }


# ---------------------------------------------------------------- 4I mix diag
def mix_rms_block(final, raw, ref, stem_fixed=None, stem_old=None, final_old=None):
    def rms(x):
        return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    out = {
        "final_mix_rms_dbfs": round(20 * np.log10(rms(final) + 1e-12), 2),
        "interaction_rms_dbfs": round(20 * np.log10(rms(stem_fixed) + 1e-12), 2) if stem_fixed is not None else None,
        "caveat": ("SECONDARY DIAGNOSTIC ONLY: both final mixes contain the identical 0.5*pristine "
                   "reference, which dominates/cancels in the difference and inflates the "
                   "normaliser; normalised final-mix RMS is NOT perceptual closeness."),
    }
    if final_old is not None:
        n = min(len(final), len(final_old))
        d = rms(final[:n] - final_old[:n])
        out["final_mix_diff_vs_old_rms_db_relative"] = (
            round(20 * np.log10(d / (rms(final_old[:n]) + 1e-12)), 3) if d > 0 else None)
    if stem_old is not None and stem_fixed is not None:
        n = min(len(stem_fixed), len(stem_old))
        d = rms(stem_fixed[:n] - stem_old[:n])
        out["interaction_stem_diff_vs_old_rms_db_relative"] = (
            round(20 * np.log10(d / (rms(stem_old[:n]) + 1e-12)), 3) if d > 0 else None)
    return out


# ---------------------------------------------------------------- confidence 4B
def confidence_block(env, times_video, high_ids, strong_flags, times_audio):
    """VISUAL_HIGH vs AUDIO_STRONG, reported separately. Gate support uses the
    envelope (max +/-50 ms >= 0.5)."""
    n = len(env)

    def support(t):
        k = int(round(t * SR))
        w = env[max(k - int(0.050 * SR), 0):min(k + int(0.050 * SR) + 1, n)]
        return bool(len(w) and w.max() >= 0.5)

    vis = [i for i in range(len(times_video)) if high_ids[i]]
    return {
        "VISUAL_HIGH": {
            "definition": "oracle annotation confidence = high (annotation certainty)",
            "n": len(vis),
            "n_with_gate_support": int(sum(support(times_video[i]) for i in vis)),
            "note": "not loudness/force; reported separately from AUDIO_STRONG",
        },
        "AUDIO_STRONG": {
            "definition": "audio refinement transient confidence = strong (local envelope contrast)",
            "n": int(np.sum(strong_flags)) if strong_flags is not None else None,
            "n_with_gate_support": int(sum(support(times_audio[i]) for i in range(len(times_audio))
                                           if strong_flags is not None and strong_flags[i])),
            "note": "not physical force; reported separately from VISUAL_HIGH",
        },
    }


# ---------------------------------------------------------------- self tests
def run_self_tests():
    """Basic hand calculations; a disagreement is a STOP condition."""
    tests = []
    # T1: simple full match
    pairs, _, _ = match_one_to_one_optimal([0.00, 0.04], [0.00, 0.08], 0.05)
    tests.append({"name": "T1_full_match", "got": len(pairs), "expect": 2,
                  "pass": len(pairs) == 2})
    # T2: greedy loses one, optimal does not (hand-verified):
    #   pred 0.028 nearest oracle 0.029 -> greedy takes it; pred 0.000 then has
    #   no feasible partner; optimal: 0.028->0.031, 0.000->0.029.
    pairs, _, _ = match_one_to_one_optimal([0.028, 0.000], [0.029, 0.031], 0.03)
    g = match_greedy_historical([0.028, 0.000], [0.029, 0.031], 0.03)
    tests.append({"name": "T2_optimal_beats_greedy", "got_optimal": len(pairs),
                  "got_greedy": g, "expect_optimal": 2, "expect_greedy": 1,
                  "pass": len(pairs) == 2 and g == 1})
    # T3: P/R/F1 hand calculation: 1 match, 2 preds, 1 oracle
    pairs, _, _ = match_one_to_one_optimal([0.0, 0.02], [0.019], 0.02)
    b = prf_from_pairs(pairs, 2, 1)
    tests.append({"name": "T3_prf_hand", "got": b, "expect": {
        "matches": 1, "precision": 0.5, "recall": 1.0, "f1": round(2 * 0.5 * 1.0 / 1.5, 4),
        "unmatched_predictions": 1, "unmatched_oracle_events": 0},
        "pass": b["matches"] == 1 and abs(b["precision"] - 0.5) < 1e-9
        and abs(b["recall"] - 1.0) < 1e-9 and abs(b["f1"] - 0.6667) < 1e-3
        and b["unmatched_predictions"] == 1 and b["unmatched_oracle_events"] == 0})
    # T4: P/R/F1 with 4 preds, 6 oracles, 3 matches -> P .75 R .5 F1 .6
    b = prf_from_pairs([(0, 0, 0.0), (1, 1, 0.0), (2, 2, 0.0)], 4, 6)
    tests.append({"name": "T4_prf_hand2", "got": (b["precision"], b["recall"], b["f1"]),
                  "expect": (0.75, 0.5, 0.6),
                  "pass": abs(b["precision"] - 0.75) < 1e-9 and abs(b["recall"] - 0.5) < 1e-9
                  and abs(b["f1"] - 0.6) < 1e-9})
    # T5: gate support on a known envelope
    env = np.zeros(int(1.0 * SR))
    env[int(0.10 * SR):int(0.31 * SR)] = fx.envelope_from_spans_FIXED(
        int(0.21 * SR), [(0, int(0.21 * SR))], 0.010, 0.060)
    blk = gate_support_block(env)
    # span endpoints are exactly 0 (2 samples), hence 0.21 - 2/48000; the block
    # reports 5-decimal rounded values, so compare at 1e-5
    tests.append({"name": "T5_gate_support", "got_nonzero": blk["nonzero_coverage_frac"],
                  "expect_nonzero": 0.21 - 2 / SR, "pass": abs(blk["nonzero_coverage_frac"] - (0.21 - 2 / SR)) < 1e-5
                  and abs(blk["mean_gain"] - (0.21 - 0.035)) < 5e-3})
    # T6: exposure partition: 1 s env open only inside [0.4,0.6] (rest [0.5,1.0))
    env = np.zeros(SR)
    env[int(0.40 * SR):int(0.60 * SR)] = 1.0
    blk = exposure_partition_block(env, allowed_spans=[(0.0, 0.45)], rest_spans=[(0.5, 1.0)])
    tests.append({"name": "T6_partition", "got_allowed": blk["allowed_contact_support"]["seconds_gain_gt_0"],
                  "got_rest": blk["known_rest_support"]["seconds_gain_gt_0"],
                  "expect_allowed": 0.05, "expect_rest": 0.10,
                  "pass": abs(blk["allowed_contact_support"]["seconds_gain_gt_0"] - 0.05) < 1e-6
                  and abs(blk["known_rest_support"]["seconds_gain_gt_0"] - 0.10) < 1e-6})
    ok = all(t["pass"] for t in tests)
    return {"tests": tests, "all_pass": ok}


if __name__ == "__main__":
    res = run_self_tests()
    fx.save_json(os.path.join(fx.FIX_LOG, "evaluator_selftest.json"), res)
    for t in res["tests"]:
        print(t["name"], "PASS" if t["pass"] else f"FAIL {t}")
    print("evaluator self-tests:", "ALL PASS" if res["all_pass"] else "FAIL (STOP condition)")
    sys.exit(0 if res["all_pass"] else 2)
