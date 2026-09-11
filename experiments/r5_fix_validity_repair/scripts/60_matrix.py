# R5-FIX PART 6 - corrected comparison matrix (report-ready extraction from
# logs/evaluation_corrected.json) + answers to the five primary questions.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixenv as fx

E = fx.load_json(os.path.join(fx.FIX_LOG, "evaluation_corrected.json"))


def row(name):
    c = E[name]
    gs, cs = c["gate_support"], c["contact_support"]
    st, fm = c["stem_onsets_vs_oracle"], c["final_mix_onsets_vs_oracle"]
    ep = c["exposure_partition"]
    dyn = c["dynamics"]
    r = {
        "case": name,
        "mean_gain": gs["mean_gain"],
        "coverage_gt0": gs["nonzero_coverage_frac"],
        "coverage_ge05": gs["coverage_gain_ge_0p5_frac"],
        "longest_open_s": gs["longest_open_span_s"],
        "contact_gain_median": cs["gain_at_contact_median"],
        "contacts_no_support": cs["n_contacts_max50_lt_0p5"],
        "stem_onsets": st["n_detected_output_onsets"],
        "stem_onsets_matched_50": st["tol50ms"]["matches"],
        "stem_onsets_matched_33": st["tol33ms"]["matches"],
        "stem_unmatched_output_onsets_50": st["tol50ms"]["unmatched_predictions"],
        "stem_unmatched_oracle_50": st["tol50ms"]["unmatched_oracle_events"],
        "final_mix_onsets_matched_50": fm["tol50ms"]["matches"],
        "rest_gt0_s": ep["known_rest_support"]["seconds_gain_gt_0"],
        "rest_ge05_s": ep["known_rest_support"]["seconds_gain_ge_0p5"],
        "rest_mean_gain": ep["known_rest_support"]["mean_gain"],
        "rest_energy_vs_raw_db": ep["known_rest_support"].get("energy_change_vs_raw_db"),
        "rest_longest_open_s": ep["known_rest_support"].get("longest_open_interval_s"),
        "allowed_gt0_s": ep["allowed_contact_support"]["seconds_gain_gt_0"],
        "unlabelled_gt0_s": ep["unlabelled_support"]["seconds_gain_gt_0"],
        "ref_onset_coincidence": c["reference_onset_coincidence"]["reference_onset_coincidence"],
        "ref_lowband_coincidence": c["reference_onset_coincidence"]["reference_lowband_onset_coincidence"],
        "spearman_amp_all": dyn["spearman_raw_vs_output_all_incl_missed"],
        "spearman_amp_open": dyn["spearman_raw_vs_output_open_only"],
        "median_event_peak_ratio": dyn["median_ratio_open"],
        "dynamics_missed": dyn["n_missed_from_gate"],
    }
    if "detector_matching_vs_oracle_video" in c:
        dm = c["detector_matching_vs_oracle_video"]
        r["detector_F1_33"] = dm["tol33ms"]["f1"]
        r["detector_F1_50"] = dm["tol50ms"]["f1"]
        r["detector_matches_80_optimal"] = dm["tol80ms"]["matches"]
        r["detector_matches_80_greedy"] = dm["tol80ms"]["historical_greedy_matches_for_note"]
        r["VISUAL_HIGH_recall_1to1_50"] = dm["VISUAL_HIGH_recall_1to1_50ms"]["recall"]
    return r


def main():
    golden = [row(n) for n in ["golden_old_C1", "golden_C1_fixed", "golden_old_C2",
                               "golden_C2_fixed", "golden_music_only", "golden_C1_fixed_shift"]]
    holdout = [row(n) for n in ["holdoutD_old_E0_C1", "holdoutD_E0_C1_fixed",
                                "holdoutD_old_oracle_C1", "holdoutD_oracle_C1_fixed",
                                "holdoutD_music_only", "holdoutD_E0_fixed_shift"]]

    def g(n, k):
        c = E[n]
        return {"golden_C1_fixed": c, "golden_C1_fixed_shift": E["golden_C1_fixed_shift"],
                "golden_old_C1": c}.get(n)

    old_g, fix_g, old_o, fix_o = E["golden_old_C1"], E["golden_C1_fixed"], E["golden_old_C2"], E["golden_C2_fixed"]
    old_h, fix_h = E["holdoutD_old_E0_C1"], E["holdoutD_E0_C1_fixed"]
    orc_h, sh_h = E["holdoutD_oracle_C1_fixed"], E["holdoutD_E0_fixed_shift"]
    mo_g, mo_h = E["golden_music_only"], E["holdoutD_music_only"]
    sh_g = E["golden_C1_fixed_shift"]

    answers = {
        "A_does_fixing_C1_change_contact_preservation": {
            "golden": {"matched_50_old": old_g["stem_onsets_vs_oracle"]["tol50ms"]["matches"],
                       "matched_50_fixed": fix_g["stem_onsets_vs_oracle"]["tol50ms"]["matches"],
                       "median_ratio_old": old_g["dynamics"]["median_ratio_open"],
                       "median_ratio_fixed": fix_g["dynamics"]["median_ratio_open"]},
            "holdoutD_E0": {"matched_50_old": old_h["stem_onsets_vs_oracle"]["tol50ms"]["matches"],
                            "matched_50_fixed": fix_h["stem_onsets_vs_oracle"]["tol50ms"]["matches"],
                            "note": ("part of the historical 'preserved' count was produced by the "
                                     "0->1 entry CLICK of the defective envelope, not by retained "
                                     "interaction audio: fixed C1 detects fewer output onsets but "
                                     "with far fewer unmatched ones")},
            "answer": ("No material change in genuine retention (golden 64/84 both); the old build "
                       "inflated onset counts with boundary clicks (24 unmatched output onsets vs 3)."),
        },
        "B_does_fixing_C1_change_boundary_chopping": {
            "golden_unmatched_output_onsets": {"old": old_g["stem_onsets_vs_oracle"]["tol50ms"]["unmatched_predictions"],
                                               "fixed": fix_g["stem_onsets_vs_oracle"]["tol50ms"]["unmatched_predictions"]},
            "holdout_unmatched_output_onsets": {"old": old_h["stem_onsets_vs_oracle"]["tol50ms"]["unmatched_predictions"],
                                                "fixed": fix_h["stem_onsets_vs_oracle"]["tol50ms"]["unmatched_predictions"]},
            "answer": ("Yes - decisively. The reversed-ramp envelope generated one spurious onset per "
                       "open span (0->1 jump at entry, 1->0 jump at release start, final drop). "
                       "Corrected C1 removes almost all of them (golden 88->67 detected onsets, "
                       "unmatched 24->3)."),
        },
        "C_does_fixing_C1_change_rest_exposure": {
            "rest_gt0_s": {"old": old_h["exposure_partition"]["known_rest_support"]["seconds_gain_gt_0"],
                           "fixed": fix_h["exposure_partition"]["known_rest_support"]["seconds_gain_gt_0"],
                           "oracle": orc_h["exposure_partition"]["known_rest_support"]["seconds_gain_gt_0"],
                           "shifted": sh_h["exposure_partition"]["known_rest_support"]["seconds_gain_gt_0"]},
            "answer": ("No - rest exposure is UNCHANGED (1.425 s gain>0, 1.285 s >=0.5, mean gain "
                       "0.64, only -1.8 dB vs raw). It is caused by E0 PREDICTIONS inside the "
                       "annotated rest [6,8], not by the envelope defect. Oracle timing gives exact 0; "
                       "the shifted control makes it worse (1.73 s)."),
        },
        "D_is_corrected_C2_still_inferior": {
            "golden": {"matched_50": {"C1_fixed": fix_g["stem_onsets_vs_oracle"]["tol50ms"]["matches"],
                                      "C2_fixed": fix_o["stem_onsets_vs_oracle"]["tol50ms"]["matches"]},
                       "median_event_ratio": {"C1_fixed": fix_g["dynamics"]["median_ratio_open"],
                                              "C2_fixed": fix_o["dynamics"]["median_ratio_open"]},
                       "spearman_amp": {"C1_fixed": fix_g["dynamics"]["spearman_raw_vs_output_open_only"],
                                        "C2_fixed": fix_o["dynamics"]["spearman_raw_vs_output_open_only"]}},
            "answer": ("Corrected percussive HPSS is no longer destructive (median event peak ratio "
                       "0.57->0.86, onset retention now equals C1 at 50 ms on golden), but it remains "
                       "inferior: lower amplitude fidelity, lower final-mix retention, and it adds an "
                       "STFT round trip for no measured benefit. The historical REJECTION of the old "
                       "C2 was directionally right even though the control was mis-implemented."),
        },
        "E_does_placement_beat_equal_coverage_shift": {
            "golden": {"matched_50": {"in_place": fix_g["stem_onsets_vs_oracle"]["tol50ms"]["matches"],
                                      "shifted": sh_g["stem_onsets_vs_oracle"]["tol50ms"]["matches"]},
                       "spearman_amp": {"in_place": fix_g["dynamics"]["spearman_raw_vs_output_open_only"],
                                        "shifted": sh_g["dynamics"]["spearman_raw_vs_output_open_only"]}},
            "holdoutD": {"matched_50": {"in_place": fix_h["stem_onsets_vs_oracle"]["tol50ms"]["matches"],
                                        "shifted": sh_h["stem_onsets_vs_oracle"]["tol50ms"]["matches"]},
                         "rest_gt0_s": {"in_place": fix_h["exposure_partition"]["known_rest_support"]["seconds_gain_gt_0"],
                                        "shifted": sh_h["exposure_partition"]["known_rest_support"]["seconds_gain_gt_0"]}},
            "answer": ("Yes - equal-coverage shifted gating loses retention (golden 64->50, holdout "
                       "54->49), degrades amplitude correlation (golden 0.96->0.68), and increases "
                       "rest exposure (holdout 1.42->1.73 s). Correct placement carries the benefit, "
                       "not merely exposing the same amount of audio. Single shift per window; not a "
                       "statistical test."),
        },
        "control_A_music_only_confound": {
            "final_mix_onsets_matched_50": {"golden": mo_g["final_mix_onsets_vs_oracle"]["tol50ms"]["matches"],
                                            "holdoutD": mo_h["final_mix_onsets_vs_oracle"]["tol50ms"]["matches"]},
            "note": ("with NO interaction stem at all, music alone still 'matches' 27/84 golden and "
                     "37/86 holdout oracle events through the final mix - final-mix onset matching "
                     "is a weak preservation metric"),
        },
    }

    out = {
        "phase": "R5-FIX PART 6 corrected comparison matrix",
        "metric_terminology": {
            "stem_onsets_matched": "output-onset identities of the INTERACTION STEM one-to-one matched to oracle contact audio times (tolerance in key)",
            "unmatched_output_onsets": "named unmatched_detected_output_onsets; never 'hallucinations'",
            "ref_onset_coincidence": ("stem onsets within +/-30 ms of pristine-reference onsets; does NOT "
                                      "identify neighbouring Taiko; no source-specific suppression claim"),
            "rest columns": "direct gain-envelope support inside annotated G1 rest [6,8] s (Holdout-D); golden has no annotated rest (0 by definition of the partition)",
            "dynamics": "raw-vs-output EVENT AMPLITUDE correlations; corr(peak_ratio, raw_peak) retired",
            "VISUAL_HIGH/AUDIO_STRONG": "reported separately in evaluation_corrected.json confidence_4B; neither measures physical force",
        },
        "golden": golden,
        "holdoutD": holdout,
        "primary_questions": answers,
    }
    fx.save_json(os.path.join(fx.FIX, "corrected_comparison_matrix.json"), out)
    print("matrix -> corrected_comparison_matrix.json")
    for q, a in answers.items():
        if "answer" in a:
            print("-", q, ":", a["answer"][:110], "...")


if __name__ == "__main__":
    main()
