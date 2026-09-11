# R5 - finalize dev detector configuration and dev visual candidates.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from r5_common import (DEFAULT_CONFIG, OUT, LOG, WORK, R4_OUT, load_json, save_json,
                       detect_candidates)
from importlib import import_module
d20 = import_module("20_detector_dev")
FPS = 60.04

CHOSEN = {"threshold": 0.40, "slope_min": 0.15, "w_jz_text": 1.8, "w_jz_diff": 1.5,
          "w_glass_bright": 1.0, "w_hand_diff": 0.3}

series = d20.load_dev()
oracle, contacts, noncontacts = d20.oracle_events()
regions = [rg for rg in oracle["regions"] if not rg["id"].startswith("G")]
cfg = {**DEFAULT_CONFIG, **CHOSEN}

score, cands = detect_candidates(series, FPS, cfg)
tc = np.array([e["t_video"] for e in contacts])

def in_region(t):
    return any(rg["t0"] - 0.05 <= t <= rg["t1"] + 0.05 for rg in regions)

# region-interior FP: distance to nearest oracle event (association fuzz check)
fp_in = [c["t_video"] for c in cands
         if np.min(np.abs(tc - c["t_video"])) > 0.050 and in_region(c["t_video"])]
d_fp = [float(np.min(np.abs(tc - t))) * 1000 for t in fp_in]
print(f"region-interior FPs: {len(fp_in)}, nearest-oracle distance ms: "
      f"median {np.median(d_fp):.0f} p25 {np.percentile(d_fp, 25):.0f} "
      f"p75 {np.percentile(d_fp, 75):.0f} max {np.max(d_fp):.0f}")
save_json(os.path.join(LOG, "dev_fp_distance.json"),
          {"fp_times": [round(float(t), 3) for t in fp_in],
           "nearest_oracle_ms": [round(x, 1) for x in d_fp]})

config = {
    "experiment": "R5 contact candidate detector",
    "video_decode": "ffmpeg transpose=2 (upright), scale 960x540, rgb24",
    "fps": FPS,
    "geometry_px_960x540": {
        "jz_band_upper": list(__import__("r5_common").JZ_BAND),
        "jz_low": list(__import__("r5_common").JZ_LOW),
        "glass_ellipse": {"c": [__import__("r5_common").GLASS_CX, __import__("r5_common").GLASS_CY],
                          "ab": [__import__("r5_common").GLASS_A, __import__("r5_common").GLASS_B],
                          "scale": 0.92},
        "hand_band_y0": __import__("r5_common").HAND_Y0,
    },
    "masks": {
        "warm_text": "r>150 & g>140 & b<0.6*min(r,g) & |r-g|<45  (skin-resistant)",
        "white_text": "r>200 & g>200 & b>180",
        "glass_bright": "gray>200 inside glass ellipse",
    },
    "features": {
        "jz_text": "warm_count + white_count in judgment zones (text evidence)",
        "jz_diff": "mean |gray_t - gray_t-1| in judgment zones",
        "glass_bright": "count gray>200 in glass ellipse (hit-effect flash)",
        "hand_diff": "mean |gray diff| in y>=330 band (hand/arm motion)",
    },
    "config": cfg,
    "tuning_provenance": "dev only: 5-threshold sweep -> thr x slope sweep -> one weight revision; no holdout used",
}
save_json(os.path.join(OUT, "detector_config.json"), config)

out = {
    "experiment": "R5 dev automatic visual candidates (golden sample)",
    "source_video": "D:/Code/RhythmAlign/experiments/r1_golden_sample/outputs/golden_video.mp4",
    "detector_config": "outputs/detector_config.json",
    "n_candidates": len(cands),
    "candidates": cands,
}
save_json(os.path.join(OUT, "dev_contacts_auto_visual.json"), out)

# final dev metrics for the record
r = d20.eval_candidates(cands, contacts, oracle["regions"], noncontacts)
save_json(os.path.join(LOG, "dev_metrics_chosen.json"), r)
print("dev metrics:", {k: r[k] for k in ("n_pred", "tol33", "tol50", "strong",
                                         "region_coverage_overall", "false_open_s")})
print("saved outputs/detector_config.json + outputs/dev_contacts_auto_visual.json")
