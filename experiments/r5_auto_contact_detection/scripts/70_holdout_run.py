# R5 Phase 9 - run the FROZEN detector on the holdout and evaluate vs the blind oracle.
# No parameter may be changed here (detector_config_frozen.json).
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from r5_common import (OUT, LOG, WORK, load_json, save_json, extract_features,
                       detect_candidates)
from importlib import import_module
d20 = import_module("20_detector_dev")
FPS = 60.04


def main():
    frozen = load_json(os.path.join(OUT, "detector_config_frozen.json"))
    cfg = frozen["config"]
    n, series = extract_features(os.path.join(OUT, "holdout_video.mp4"))
    save_path = os.path.join(WORK, "holdout_features.npz")
    np.savez(save_path, fps=FPS, **series)
    print("holdout frames:", n)

    score, cands = detect_candidates(series, FPS, cfg)
    out = {
        "experiment": "R5 holdout automatic visual candidates (FROZEN detector)",
        "source_video": "outputs/holdout_video.mp4",
        "detector_config": "outputs/detector_config_frozen.json",
        "n_candidates": len(cands),
        "candidates": cands,
    }
    save_json(os.path.join(OUT, "holdout_contacts_auto_visual.json"), out)
    print("candidates:", len(cands))

    oracle = load_json(os.path.join(OUT, "holdout_oracle_blind.json"))
    contacts = [e for e in oracle["events"] if e["status"] == "ok"]
    noncontacts = [e for e in oracle["events"] if e["status"] != "ok"]
    r = d20.eval_candidates(cands, contacts, oracle["regions"], noncontacts,
                            dur_s=18.0)
    save_json(os.path.join(LOG, "holdout_metrics.json"), r)
    t5, t3 = r["tol50"], r["tol33"]
    print(f"\nHOLDOUT (blind oracle: {len(contacts)} contacts, {len(noncontacts)} rejected/rest)")
    print(f"  +/-50ms: P {t5['precision']:.3f} R {t5['recall']:.3f} F1 {t5['f1']:.3f} "
          f"| err med {t5['err_ms_median']} p95 {t5['err_ms_p95']} ms")
    print(f"  +/-33ms: P {t3['precision']:.3f} R {t3['recall']:.3f} F1 {t3['f1']:.3f}")
    print(f"  strong (oracle conf=high) recall: {r['strong']['recall_50ms']:.3f} "
          f"({int(round(r['strong']['recall_50ms']*r['strong']['n']))}/{r['strong']['n']})")
    print(f"  region coverage: {r['region_coverage_overall']:.3f} | false-open {r['false_open_s']} s")
    print(f"  FP: {len(r['false_positives'])} | FN: {len(r['false_negatives'])}")
    for fp in r["false_positives"]:
        print(f"    FP t={fp['t']:.3f} score={fp['score']} near_noncontact={fp['near_noncontact']}")
    print(f"  FN ids: {r['false_negatives']}")
    print("  regions:", {k: v["coverage"] for k, v in r["regions"].items()})


if __name__ == "__main__":
    main()
