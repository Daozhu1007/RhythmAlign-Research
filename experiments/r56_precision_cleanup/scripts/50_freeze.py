# R5.6 Freeze - write detector_config_r56_frozen.json AFTER dev decision,
# BEFORE any holdout-D work. Chosen pipeline: E2 (E1 precision cleanup +
# burst-constrained strong-audio completion).
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common56 as c

CFG = {
    "experiment": "R5.6 precision cleanup + burst-constrained audio completion",
    "selected": "E2",
    "baseline": "E0 = R5.5 frozen detector (detector_config_r55_frozen.json), unchanged",
    "frozen_components": [
        "R5.5 visual feature extraction + fusion (r5_common)",
        "R5.5 burst-state detector level-1 + rule A level-2 (detector55)",
        "C1 reconstruction (pre 15 ms / post 150 ms, attack 10 ms / release 60 ms)",
        "audio refinement +/-40 ms on R4 comb",
        "clean music alignment + final gain convention (0.5*ref + 0.5*interaction)",
    ],
    "e1_precision_cleanup": {
        "description": "candidate post-filter ONLY; changes which R5.5 candidates open windows",
        "strong_present": "open window (unchanged)",
        "no_transient": "candidate stays in list, never opens a window (E0 convention)",
        "weak": ("open window only with extra support: "
                 "(a) visual: prom_frac >= {weak_prom_frac} OR >= {weak_chan_n} channels above "
                 "{weak_chan_floor} of p99.5 OR (in burst within {weak_burst_edge_ms} ms of a burst "
                 "edge AND within {weak_near_sp_ms} ms of a strong/present candidate); "
                 "(b) audio-rescue: comb peak >= {rescue_comb_peak} AND peak_over_noise >= "
                 "{rescue_comb_ratio} (visually-silent true contacts)"),
        "echo_ratio_suppression": "REJECTED by Phase-1 audit: echo FPs are not separable from "
                                  "true adjacent contacts by visual peak ratio (median 1.09 vs "
                                  "0.75-0.84; every useful threshold kills more true 90-120 ms "
                                  "doubles than echoes). Refinement confidence already covers "
                                  "87/99 echo FPs via weak/no_transient.",
        **c.E1_CFG,
    },
    "e2_audio_completion": {
        "description": ("adds audio-only candidates ONLY inside E0-frozen active bursts, "
                        "for visually-silent contacts with strong audio evidence"),
        "source": "frozen R4 comb envelope local peaks",
        "outside_burst": "never adds anything (hard rule)",
        "strong_tier": "peak >= {min_peak} AND > {min_ratio}x local median of +/-0.25 s "
                       "(excl. +/-60 ms), min distance {min_dist_s} s - mirrors the frozen "
                       "'strong' refinement level",
        "taiko_guard": ("music-removed residual support required: mid+hi1+hi2 >= {resid_sum_min} "
                        "AND hi1 >= {hi1_min}; bass-only (taiko-like) onsets fail hi1"),
        "dedup": ">= {d_e1_min_ms} ms from every E1 window, >= {d_e2_min_ms} ms between added events",
        "hand_motion": "recorded as bonus evidence only, never required",
        **c.E2_CFG,
    },
    "dev_evidence": {
        "window_convention": "candidates = R5.5 visual candidate list (incl. no_transient); "
                             "windows = gates actually opened",
        "devA": {"strong_R": [0.947, 0.947, 0.947], "strong_gate": [0.833, 0.817, 0.900],
                 "false_win_s": [2.72, 1.96, 2.14], "music_leak": [27, 21, 24],
                 "UWR_himed": [0.774, 0.818, 0.806], "rms_db": [-15.7, -15.9, -16.9],
                 "F1_50": [0.619, 0.619, 0.630]},
        "devB": {"strong_R": [0.818, 0.818, 0.818], "strong_gate": [0.894, 0.872, 0.936],
                 "false_win_s": [3.44, 2.53, 2.65], "music_leak": [17, 11, 10],
                 "UWR_himed": [0.755, 0.822, 0.833], "rms_db": [-10.8, -11.6, -12.3],
                 "F1_50": [0.586, 0.586, 0.609]},
        "devC_oracle_incomplete": {
            "strong_R": [0.810, 0.810, 0.905], "strong_gate": [0.794, 0.735, 0.853],
            "false_win_s": [6.18, 5.03, 5.23], "music_leak": [33, 15, 19],
            "taiko_leak": [3, 2, 2],
            "UWR_himed": [0.569, 0.631, 0.654], "rms_db": [-8.8, -9.6, -10.2],
            "F1_50": [0.417, 0.417, 0.474]},
        "order": "[E0, E1, E2]",
        "completion_quality": "20 added events on devA/B/C: 18 match oracle within 80 ms; "
                              "0 bass-dominated; 5 coincide with music onsets but are "
                              "residual-supported on-beat contacts (2 recovered strong oracle)",
    },
    "decision": ("E2 kept: strong recall does not drop on any window (devC +0.095), strong gate "
                 "coverage rises on all three windows (+6.7/+4.2/+5.9 pt), false windows / music "
                 "leakage drop on all three, UWR rises on all three, final-mix RMS improves on "
                 "all three. E1 alone trades gate coverage down; E2's completion buys it back."),
    "frozen": True,
    "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "freeze_note": ("No parameter may be changed after this point based on holdout-D results. "
                    "Holdout-D oracle annotation (video-first) must complete before the frozen "
                    "pipeline runs on it."),
}

if __name__ == "__main__":
    p = os.path.join(c.R56_OUT, "detector_config_r56_frozen.json")
    c.save_json(p, CFG)
    print("frozen:", p, "at", CFG["frozen_at"])
