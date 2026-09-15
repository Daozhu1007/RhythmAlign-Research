# RTA1 step 08 - headroom aggregation + pre-registered decision rules.
#
# Joins per-mixture metrics from the frozen baselines (05) and the oracles
# (06/07), scores the DECISION_RULES.md success requirement on the 16 held-out
# mixtures, and reports which outcome (A/B/C/D) the evidence supports.
#
# --selftest feeds fabricated fixture tables through the aggregation math to
# validate the decision logic itself (software validation only).
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import rta1_lib as rl  # noqa: E402

N_HELDOUT = 16
N_PASS_REQUIRED = 12
SNR_ADV_DB = 6.0
TARGET_NMSE_DB = -15.0
NUIS_ATT_DB = 10.0
EVENT_LEVEL_TOL_DB = 1.0


def score_mixture(row_bounded, rows_baselines, row_complex=None):
    """Per-mixture DECISION_RULES flags for the optimized bounded oracle.

    rows_baselines: {"zeroshot": snr_db, "s1r": snr_db} on the same mixture.
    Event/friction flags come precomputed in the row when scored events exist
    (missing scored events => flag True-but-unverified is NOT allowed; the
    mixture then cannot count toward the pass set, recorded as 'no_events').
    """
    adv_ok = all(row_bounded["recon_snr_db"] - b >= SNR_ADV_DB
                 for b in rows_baselines.values())
    nmse_ok = row_bounded["target_nmse_db"] <= TARGET_NMSE_DB
    nuis_ok = row_bounded["nuisance_attenuation_db"] >= NUIS_ATT_DB
    weak_ok = row_bounded.get("no_omitted_weak_event", None)
    fric_ok = row_bounded.get("no_broken_friction_interval", None)
    lvl_ok = row_bounded.get("event_levels_within_tol", None)
    if None in (weak_ok, fric_ok, lvl_ok):
        return {"pass": False, "reason": "no scored event truth for this stratum",
                "adv_ok": adv_ok, "nmse_ok": nmse_ok, "nuis_ok": nuis_ok}
    ok = bool(adv_ok and nmse_ok and nuis_ok and weak_ok and fric_ok and lvl_ok)
    return {"pass": ok, "adv_ok": adv_ok, "nmse_ok": nmse_ok,
            "nuis_ok": nuis_ok, "weak_ok": weak_ok, "fric_ok": fric_ok,
            "level_ok": lvl_ok}


def aggregate(mixture_rows, meta):
    """mixture_rows: list of {"mixture","panel","session","stratum","scores",
    "bounded_recon_snr","bounded_nmse","bounded_nuis","baseline_snr":{...},
    "complex_recon_snr"}. Returns outcome support per DECISION_RULES."""
    held = [r for r in mixture_rows if r["panel"] == "heldout"]
    scored = [r for r in held if r["scores"].get("pass")]
    n_pass = len(scored)
    sessions = {r["session"] for r in scored}
    strata = {r["stratum"] for r in scored}
    coverage_ok = len(sessions) == 2 and len(strata) >= 4

    outcome_A = bool(n_pass >= N_PASS_REQUIRED and coverage_ok)

    # Outcome B: bounded fails the pass count but the complex oracle beats both
    # baselines by the margin on >= N_PASS_REQUIRED held-out mixtures.
    comp_better = 0
    for r in held:
        if r.get("complex_recon_snr") is None:
            continue
        if all(r["complex_recon_snr"] - b >= SNR_ADV_DB
               for b in r["baseline_snr"].values()):
            comp_better += 1
    outcome_B = bool(not outcome_A and comp_better >= N_PASS_REQUIRED)

    # Outcome C: a frozen baseline is within 3 dB of the best oracle on most
    # held-out mixtures.
    close = 0
    for r in held:
        best_oracle = max(x for x in
                          (r.get("bounded_recon_snr"), r.get("complex_recon_snr"))
                          if x is not None)
        if max(r["baseline_snr"].values()) >= best_oracle - 3.0:
            close += 1
    outcome_C = bool(not outcome_A and not outcome_B
                     and close > len(held) / 2)

    return {
        "n_heldout": len(held),
        "n_pass": n_pass,
        "coverage_sessions": sorted(sessions),
        "coverage_strata": sorted(strata),
        "coverage_ok": coverage_ok,
        "outcome_A_bounded_beats_models": outcome_A,
        "outcome_B_phase_limited": outcome_B,
        "outcome_C_models_approach_oracle": outcome_C,
        "outcome_D_truth_insufficient": meta.get("truth_quality_insufficient", 0)
                                        > len(held) / 2,
        "note": "outcomes are decision SUPPORT; the recorded report decides",
    }


def selftest():
    """Fabricated fixture tables - validates aggregation math ONLY."""
    strata = ("weak", "strong_button", "friction_tail", "dense")

    def mk(i, bounded_snr, base_zs, base_s1r, comp_snr=None, stratum="weak",
           session="session_02", ev=(True, True, True)):
        return {
            "mixture": f"mix_{i}", "panel": "heldout", "session": session,
            "stratum": stratum,
            "bounded_recon_snr": bounded_snr, "bounded_nmse": -20.0,
            "bounded_nuis": 12.0,
            "baseline_snr": {"zeroshot": base_zs, "s1r": base_s1r},
            "complex_recon_snr": comp_snr,
            "scores": {"pass": bounded_snr - max(base_zs, base_s1r) >= SNR_ADV_DB
                       and ev == (True, True, True)},
        }

    # Outcome A case: bounded oracle dominates on 14/16
    rows_a = ([mk(i, 25.0, 10.0, 11.0, stratum=strata[i % 4],
                  session="session_02" if i % 2 == 0 else "session_03")
              for i in range(14)]
              + [mk(14, 12.0, 10.0, 11.0), mk(15, 12.0, 10.0, 11.0,
                                              session="session_03")])
    agg = aggregate(rows_a, {"truth_quality_insufficient": 0})
    assert agg["outcome_A_bounded_beats_models"] \
        and not agg["outcome_B_phase_limited"], agg

    # Outcome B case: bounded fails, complex dominates
    rows_b = [mk(i, 12.0, 10.0, 11.0, comp_snr=30.0,
                 stratum=strata[i % 4],
                 session="session_02" if i % 2 else "session_03")
              for i in range(16)]
    agg_b = aggregate(rows_b, {"truth_quality_insufficient": 0})
    assert agg_b["outcome_B_phase_limited"] and \
        not agg_b["outcome_A_bounded_beats_models"], agg_b

    # Outcome C case: everything close
    rows_c = [mk(i, 12.5, 10.0, 11.0, comp_snr=12.6) for i in range(16)]
    agg_c = aggregate(rows_c, {"truth_quality_insufficient": 0})
    assert agg_c["outcome_C_models_approach_oracle"], agg_c  # noqa

    # Outcome D case: too little trustworthy truth
    agg_d = aggregate(rows_a, {"truth_quality_insufficient": 10})
    assert agg_d["outcome_D_truth_insufficient"], agg_d

    print("SELFTEST OK: decision aggregation A/B/C/D all reproduce (fixture data)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baselines", default=os.path.join(
        rl.ORACLE, "baselines", "baseline_metrics.json"))
    ap.add_argument("--oracles", default=os.path.join(
        rl.ORACLE, "oracle_metrics.json"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    for p in (args.baselines, args.oracles):
        if not os.path.exists(p):
            print(f"missing input: {p} - run 05/06/07 first")
            return 1
    print("real aggregation runs only after capture + 05/06/07")
    return 1


if __name__ == "__main__":
    sys.exit(main())
