# R5.5 Phase 5 - freeze the R5.5 detector configuration after dev-only tuning.
# After this file is written, NO parameter may be changed based on Holdout-C.
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common55 import R55_OUT, save_json
from detector55 import DEFAULT55

CFG = {**DEFAULT55, "burst_enter": 0.40, "burst_stay": 0.22,
       "burst_hold_frames": 30, "a_prom_frac": 0.03, "a_min_dist_frames": 3}

doc = {
    "experiment": "R5.5 burst-aware contact detector (level1 burst + level2 atomic)",
    "rule": "A",
    "ruleA": "every prominent local peak of the R5-frozen fused novelty score "
             "inside a burst is an atomic event candidate (platform re-trigger); "
             "no rising-edge requirement inside bursts",
    "ruleB_rejected": "jz_diff atomic peaks + glass/hand agreement, burst-local "
                      "adaptive threshold (worse on both dev windows: dense F1 "
                      "0.523/0.545 vs A 0.609/0.611; strong recall lower)",
    "outside_bursts": "R5 frozen rising-edge rule, unchanged",
    "visual_features": "R5-frozen (jz_text, jz_diff, glass_bright, hand_diff) + "
                       "R5-frozen fusion weights/normalization",
    "config": CFG,
    "level1_burst": {
        "activity": "fused score smoothed by burst_smooth_frames",
        "enter": "env >= burst_enter",
        "stay": "hysteresis: stay while env >= burst_stay",
        "exit": "burst_hold_frames consecutive frames below stay",
        "dev_region_match": {"mean_coverage": 0.947, "false_burst_s_per_window": 0.77,
                             "missed_burst_s_per_window": 0.89},
    },
    "dev_results": {
        "devA": {"F1_50": 0.619, "dense_F1": 0.609, "dense_R": 0.768,
                 "strong_recall": 0.947, "region_coverage": 1.0,
                 "gate_coverage": 0.869, "mix_rms_db": -15.7},
        "devB": {"F1_50": 0.586, "dense_F1": 0.611, "dense_R": 0.689,
                 "strong_recall": 0.818, "region_coverage": 0.978,
                 "gate_coverage": 0.837, "mix_rms_db": -10.8},
        "r5_baseline": {"devA": {"F1_50": 0.605, "strong_recall": 0.842,
                                  "gate_coverage": 0.638},
                         "devB": {"F1_50": 0.488, "strong_recall": 0.455,
                                   "gate_coverage": 0.663}},
    },
    "tuning_provenance": "dev only (golden 22.5-37.5 + old holdout 84-102, both "
                         "declared DEVELOPMENT for R5.5): burst 3x3x3 grid on "
                         "region match; rule A vs B comparison; a_prom/a_dist "
                         "2x2 refinement. No holdout-C used.",
    "frozen": True,
    "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "freeze_note": "No parameter may be changed after this point based on "
                   "Holdout-C results.",
}
save_json(os.path.join(R55_OUT, "detector_config_r55_frozen.json"), doc)
print("frozen at", doc["frozen_at"])
