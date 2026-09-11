# R5-FIX PART 1 - deterministic unit / numerical tests for the corrected C1
# envelope, run on synthetic span configurations AND on the real golden and
# Holdout-D E0 span sets. The OLD implementation is tested identically to
# document the defect (it must FAIL the discontinuity tests).
# Output: corrected_C1_envelope_tests.json (phase root)
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixenv as fx

A = max(int(0.010 * fx.SR), 2)   # 480 samples attack
R = max(int(0.060 * fx.SR), 2)   # 2880 samples release
STEP_BOUND = 2.0 / min(A, R)     # max |d(gain)/d(sample)| allowed (no discontinuity)


def check_envelope(env, spans, name):
    """Generic property battery. spans: list of (s,e) sample spans that were
    used to build env (post-merge)."""
    n = len(env)
    res = {"name": name, "n_samples": int(n), "n_spans": len(spans)}
    support = np.zeros(n, bool)
    for s, e in spans:
        support[s:e] = True
    res["gain_in_unit_range"] = bool(np.all(env >= 0.0) and np.all(env <= 1.0))
    res["outside_support_zero"] = bool(np.all(env[~support] == 0.0))
    # attack/release monotonicity + continuity, per span on THIS env where the
    # span is isolated (single-span configs); for merged configs these are
    # checked on the isolated sub-envelope by the caller.
    d = np.diff(env)
    res["max_abs_gain_step"] = float(np.max(np.abs(d))) if n > 1 else 0.0
    res["no_discontinuity"] = bool(res["max_abs_gain_step"] <= STEP_BOUND)
    # interior zero notch: any zero strictly inside a span with nonzero neighbours
    notches = 0
    for s, e in spans:
        seg = env[s:e]
        if len(seg) >= 3:
            inner = seg[1:-1]
            notches += int(np.sum((inner == 0.0)))
    res["interior_zero_notches"] = int(notches)
    res["no_interior_zero_notch"] = bool(notches == 0)
    return res


def check_single_span():
    s, e = 5000, 5000 + int(0.300 * fx.SR)
    spans = [(s, e)]
    env = fx.envelope_from_spans_FIXED(90000, spans, 0.010, 0.060)
    res = check_envelope(env, spans, "single_isolated_span")
    a_seg = env[s:s + A]
    r_seg = env[e - R:e]
    res["attack_monotonic_nondecreasing"] = bool(np.all(np.diff(a_seg) >= -1e-12))
    res["release_monotonic_nonincreasing"] = bool(np.all(np.diff(r_seg) <= 1e-12))
    res["attack_starts_at_zero"] = bool(env[s] == 0.0)
    res["attack_reaches_interior_continuously"] = bool(
        abs(env[s + A - 1] - 1.0) < 1e-12 and abs(env[s + A] - 1.0) < 1e-12
        and abs(env[s + A - 1] - env[s + A - 2]) <= STEP_BOUND)
    res["release_leaves_interior_continuously"] = bool(
        abs(env[e - R] - 1.0) < 1e-12 and abs(env[e - R - 1] - 1.0) < 1e-12
        and abs(env[e - R] - env[e - R + 1]) <= STEP_BOUND)
    res["release_ends_at_zero"] = bool(env[e - 1] == 0.0)
    res["pass"] = bool(res["gain_in_unit_range"] and res["outside_support_zero"]
                       and res["no_discontinuity"] and res["no_interior_zero_notch"]
                       and res["attack_monotonic_nondecreasing"]
                       and res["release_monotonic_nonincreasing"]
                       and res["attack_starts_at_zero"]
                       and res["attack_reaches_interior_continuously"]
                       and res["release_leaves_interior_continuously"]
                       and res["release_ends_at_zero"])
    return res, env


def check_two_overlapping():
    n = 96000
    spans = [(5000, 5000 + int(0.300 * fx.SR)), (5000 + int(0.200 * fx.SR), 5000 + int(0.480 * fx.SR))]
    merged = fx.merge_spans([list(sp) for sp in spans])
    env = fx.envelope_from_spans_FIXED(n, merged, 0.010, 0.060)
    e1 = fx.envelope_from_spans_FIXED(n, [list(spans[0])], 0.010, 0.060)
    e2 = fx.envelope_from_spans_FIXED(n, [list(spans[1])], 0.010, 0.060)
    res = check_envelope(env, merged, "two_overlapping_spans")
    res["overlap_never_reduces_gain"] = bool(
        np.all(env >= e1 - 1e-15) and np.all(env >= e2 - 1e-15)
        and np.all(env >= np.maximum(e1, e2) - 1e-15))
    # no dip between the two constituent openings: inside the merged span the
    # combined env never falls below either individual envelope (covered above);
    # additionally check no local minimum below 1.0 in the shared interior
    core = env[spans[1][0]:spans[0][1]]
    res["shared_region_full_gain"] = bool(np.all(core >= 1.0 - 1e-12))
    res["pass"] = bool(res["gain_in_unit_range"] and res["outside_support_zero"]
                       and res["no_discontinuity"] and res["no_interior_zero_notch"]
                       and res["overlap_never_reduces_gain"] and res["shared_region_full_gain"])
    return res, env, e1, e2


def check_nearly_touching():
    n = 96000
    gap = 5  # samples
    s0 = 20000
    e0 = s0 + int(0.220 * fx.SR)
    s1 = e0 + gap
    e1 = s1 + int(0.180 * fx.SR)
    spans = [(s0, e0), (s1, e1)]
    env = fx.envelope_from_spans_FIXED(n, spans, 0.010, 0.060)
    res = check_envelope(env, spans, "two_nearly_touching_spans")
    gapseg = env[e0:s1]
    res["gap_stays_open_low"] = bool(np.all(gapseg > 0.0))          # release tail overlaps entry ramp
    res["gap_max_gain"] = float(np.max(gapseg))
    res["no_0_to_1_jump_in_gap"] = bool(np.max(np.abs(np.diff(env[e0 - R:s1 + A]))) <= STEP_BOUND)
    res["pass"] = bool(res["gain_in_unit_range"] and res["outside_support_zero"]
                       and res["no_discontinuity"] and res["no_interior_zero_notch"]
                       and res["no_0_to_1_jump_in_gap"])
    return res, env


def check_dense_merged():
    n = 240000
    t = 30000
    spans = []
    rng = np.random.default_rng(20260909)   # fixed, documented
    for _ in range(12):
        dur = int((0.140 + 0.010 * rng.random()) * fx.SR)
        spans.append((t, t + dur))
        t += int((0.010 + 0.012 * rng.random()) * fx.SR)   # heavily overlapping
    merged = fx.merge_spans([list(sp) for sp in spans])
    env = fx.envelope_from_spans_FIXED(n, merged, 0.010, 0.060)
    res = check_envelope(env, merged, "dense_merged_spans")
    indiv = [fx.envelope_from_spans_FIXED(n, [list(sp)], 0.010, 0.060) for sp in spans]
    res["overlap_never_reduces_gain"] = bool(
        all(np.all(env >= ei - 1e-15) for ei in indiv))
    res["pass"] = bool(res["gain_in_unit_range"] and res["outside_support_zero"]
                       and res["no_discontinuity"] and res["no_interior_zero_notch"]
                       and res["overlap_never_reduces_gain"])
    return res, env


def check_exact_boundaries():
    s, e = 7000, 7000 + int(0.300 * fx.SR)
    env = fx.envelope_from_spans_FIXED(80000, [(s, e)], 0.010, 0.060)
    res = {"name": "exact_boundary_samples",
           "env_at_span_start": float(env[s]),
           "env_at_span_start_plus_1": float(env[s + 1]),
           "env_at_attack_end_minus_1": float(env[s + A - 1]),
           "env_at_attack_end": float(env[s + A]),
           "env_at_release_start_minus_1": float(env[e - R - 1]),
           "env_at_release_start": float(env[e - R]),
           "env_at_span_end_minus_1": float(env[e - 1]),
           "env_at_span_end": float(env[e]),
           "env_before_span": float(env[s - 1]),
           "env_after_span": float(env[e + 1])}
    res["pass"] = bool(res["env_at_span_start"] == 0.0 and res["env_before_span"] == 0.0
                       and res["env_at_span_start_plus_1"] > 0.0
                       and res["env_at_attack_end_minus_1"] == 1.0
                       and res["env_at_attack_end"] == 1.0
                       and res["env_at_release_start_minus_1"] == 1.0
                       and res["env_at_release_start"] == 1.0
                       and res["env_at_span_end_minus_1"] == 0.0
                       and res["env_at_span_end"] == 0.0 and res["env_after_span"] == 0.0)
    return res, env


def real_spans_checks():
    out = {}
    # golden oracle spans (84 events, same filter as R4 build)
    ev = [e for e in fx.load_json(os.path.join(fx.R4_OUT, "contacts_refined.json"))["events"]
          if e["refined_confidence"] != "no_transient" and e["type"] != "none"]
    g_spans = fx.spans_from_times([e["t_audio_refined"] for e in ev], 15.0)
    g_env = fx.envelope_from_spans_FIXED(15 * fx.SR, g_spans, 0.010, 0.060)
    res = check_envelope(g_env, g_spans, "real_golden_oracle_corrected")
    res["mean_gain"] = float(g_env.mean())
    out["real_golden_oracle_corrected"] = res
    # holdout-D E0 spans
    Ajs = fx.load_json(os.path.join(fx.R56_OUT, "holdoutD_contacts_auto_refined.json"))
    e0_t = [e["t_audio_refined"] for e in Ajs["events"]
            if e.get("refined_confidence") in ("strong", "present", "weak")]
    h_spans = fx.spans_from_times(e0_t, 18.0)
    h_env = fx.envelope_from_spans_FIXED(18 * fx.SR, h_spans, 0.010, 0.060)
    res_h = check_envelope(h_env, h_spans, "real_holdoutD_E0_corrected")
    res_h["mean_gain"] = float(h_env.mean())
    out["real_holdoutD_E0_corrected"] = res_h
    # old envelopes on the same real spans for contrast
    g_old = fx.envelope_from_spans_OLD(15 * fx.SR, g_spans, 0.010, 0.060)
    h_old = fx.envelope_from_spans_OLD(18 * fx.SR, h_spans, 0.010, 0.060)
    out["real_golden_oracle_OLD_contrast"] = {
        "name": "real_golden_oracle_OLD_contrast",
        "max_abs_gain_step": float(np.max(np.abs(np.diff(g_old)))),
        "interior_zero_notches": int(sum(
            np.sum(fx.envelope_from_spans_OLD(15 * fx.SR, [sp], 0.010, 0.060)[sp[0] + 1:sp[1] - 1] == 0.0)
            for sp in g_spans)),
        "mean_gain": float(g_old.mean()),
    }
    out["real_holdoutD_E0_OLD_contrast"] = {
        "name": "real_holdoutD_E0_OLD_contrast",
        "max_abs_gain_step": float(np.max(np.abs(np.diff(h_old)))),
        "mean_gain": float(h_old.mean()),
    }
    ok = bool(res["no_discontinuity"] and res["no_interior_zero_notch"]
              and res["gain_in_unit_range"] and res["outside_support_zero"]
              and res_h["no_discontinuity"] and res_h["no_interior_zero_notch"]
              and res_h["gain_in_unit_range"] and res_h["outside_support_zero"])
    out["real_spans_pass"] = ok
    return out, g_env, g_spans, h_env, h_spans


def old_single_span_contrast():
    s, e = 5000, 5000 + int(0.300 * fx.SR)
    env = fx.envelope_from_spans_OLD(90000, [(s, e)], 0.010, 0.060)
    return {"name": "OLD_single_span_contrast",
            "max_abs_gain_step": float(np.max(np.abs(np.diff(env)))),
            "env_at_span_start": float(env[s]),
            "env_at_attack_end_minus_1": float(env[s + A - 1]),
            "env_at_attack_end": float(env[s + A]),
            "env_at_release_start_minus_1": float(env[e - R - 1]),
            "env_at_release_start": float(env[e - R]),
            "env_at_span_end_minus_1": float(env[e - 1]),
            "documented_defect": "entry starts at 1 after a 0->1 jump; gain falls to 0 at attack end then jumps back to 1; gain drops 1->0 at release start, rises 0->1, then drops to 0 at span end"}


def main():
    r1, env1 = check_single_span()
    r2, env2, e2a, e2b = check_two_overlapping()
    r3, env3 = check_nearly_touching()
    r4, env4 = check_dense_merged()
    r5, env5 = check_exact_boundaries()
    r6, g_env, g_spans, h_env, h_spans = real_spans_checks()
    r_old = old_single_span_contrast()

    out = {
        "phase": "R5-FIX PART 1 corrected C1 envelope tests",
        "envelope_definition": {
            "outside": "gain = 0",
            "entry": "raised-cosine rise 0 -> 1 over attack = 10 ms",
            "interior": "gain = 1",
            "exit": "raised-cosine fall 1 -> 0 over release = 60 ms",
            "overlap_merge": "pointwise max across spans",
            "window_policy_unchanged": "pre 15 ms / post 150 ms (frozen C1 policy)",
            "max_abs_gain_step_bound": STEP_BOUND,
        },
        "tests": {"1_single_isolated_span": r1, "2_two_overlapping_spans": r2,
                  "3_two_nearly_touching_spans": r3, "4_dense_merged_spans": r4,
                  "5_exact_boundary_samples": r5, "6_derivative_discontinuity_sanity_real_spans": r6,
                  "OLD_implementation_contrast": r_old},
        "all_corrected_tests_pass": bool(r1["pass"] and r2["pass"] and r3["pass"]
                                         and r4["pass"] and r5["pass"] and r6["real_spans_pass"]),
    }
    fx.save_json(os.path.join(fx.FIX, "corrected_C1_envelope_tests.json"), out)

    np.save(os.path.join(fx.FIX_WORK, "env_test_single_fixed.npy"), env1)
    np.save(os.path.join(fx.FIX_WORK, "env_test_overlap_fixed.npy"), env2)
    np.save(os.path.join(fx.FIX_WORK, "env_test_overlap_a.npy"), e2a)
    np.save(os.path.join(fx.FIX_WORK, "env_test_overlap_b.npy"), e2b)
    np.save(os.path.join(fx.FIX_WORK, "env_test_near_fixed.npy"), env3)
    np.save(os.path.join(fx.FIX_WORK, "env_golden_fixed.npy"), g_env)
    np.save(os.path.join(fx.FIX_WORK, "env_golden_old.npy"),
            fx.envelope_from_spans_OLD(15 * fx.SR, g_spans, 0.010, 0.060))
    np.save(os.path.join(fx.FIX_WORK, "env_holdoutD_e0_fixed.npy"), h_env)
    np.save(os.path.join(fx.FIX_WORK, "env_holdoutD_e0_old.npy"),
            fx.envelope_from_spans_OLD(18 * fx.SR, h_spans, 0.010, 0.060))

    for k, t in out["tests"].items():
        print(k, "->", "PASS" if t.get("pass", True) else f"FAIL {t}")
    print("all_corrected_tests_pass:", out["all_corrected_tests_pass"])
    assert out["all_corrected_tests_pass"], "corrected envelope failed deterministic tests (STOP condition)"
    assert r_old["max_abs_gain_step"] > 0.5, "OLD envelope unexpectedly smooth - check test setup"


if __name__ == "__main__":
    main()
