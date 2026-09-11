# R5 diagnostic - signed matching errors + FP location map (dev, wide band).
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

used_p, errs = set(), []
for e in contacts:
    t = e["t_video"]
    d = np.abs(tp - t)
    order = np.argsort(d)
    for j in order[:5]:
        if d[j] <= 0.100 and j not in used_p:
            used_p.add(j)
            errs.append((tp[j] - t, in_region(t)))
            break
e_in = [x[0] for x in errs if x[1]]
e_out = [x[0] for x in errs if not x[1]]
print("signed err frames (in-region):", np.round(np.array(e_in) * FPS, 1))
print("in-region signed err: mean %.1f med %.1f frames" %
      (np.mean(e_in) * FPS, np.median(e_in) * FPS))
if e_out:
    print("out-region signed err frames:", np.round(np.array(e_out) * FPS, 1),
          "med %.1f" % (np.median(e_out) * FPS))
# FP map
print("\nFP (no oracle within 100ms):")
for c in cands:
    t = c["t_video"]
    if np.min(np.abs(tc - t)) > 0.100:
        print(f"  t={t:.3f} score={c['score']} region={in_region(t)}")
