# R5 diagnostic - FP/FN composition vs continuous regions (dev).
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from r5_common import detect_candidates
from importlib import import_module
d20 = import_module("20_detector_dev")
FPS = 60.04

series = d20.load_dev()
oracle, contacts, noncontacts = d20.oracle_events()
regions = [rg for rg in oracle["regions"] if not rg["id"].startswith("G")]


def in_region(t):
    return any(rg["t0"] - 0.05 <= t <= rg["t1"] + 0.05 for rg in regions)


score, cands = detect_candidates(series, FPS, {"threshold": 0.40, "slope_min": 0.15})
tp = np.array([c["t_video"] for c in cands])
tc = np.array([e["t_video"] for e in contacts])

# FN breakdown
print("=== FN (missed oracle contacts) ===")
fn_in, fn_out = 0, 0
for e in contacts:
    t = e["t_video"]
    if len(tp) and np.min(np.abs(tp - t)) <= 0.050:
        continue
    if in_region(t):
        fn_in += 1
    else:
        fn_out += 1
        print(f"  OUT {e['id']} t={t:.3f} {e['type']}/{e['confidence']} {e['area']}")
print(f"FN inside regions: {fn_in}, outside regions: {fn_out}")

# FP breakdown
print("=== FP (predictions not near any oracle contact) ===")
fp_in, fp_out = [], []
for c in cands:
    t = c["t_video"]
    if np.min(np.abs(tc - t)) <= 0.050:
        continue
    (fp_in if in_region(t) else fp_out).append(c)
for c in fp_out:
    near = [e["id"] for e in noncontacts if abs(e["t_video"] - c["t_video"]) <= 0.15]
    print(f"  OUT t={c['t_video']:.3f} score={c['score']} noncontact:{near}")
print(f"FP inside regions: {len(fp_in)} {[round(c['t_video'],2) for c in fp_in]}")
print(f"FP outside regions: {len(fp_out)}")

# oracle events inside vs outside regions
n_in = sum(1 for e in contacts if in_region(e["t_video"]))
print(f"\noracle contacts inside regions: {n_in}/{len(contacts)}")
