# R5 diagnostic - candidate vs oracle lag structure (dev).
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
score, cands = detect_candidates(series, FPS, {"threshold": 0.35})
tp = np.array([c["t_video"] for c in cands])
tc = np.array([e["t_video"] for e in contacts])
lag = np.array([tp[np.argmin(np.abs(tp - t))] - t for t in tc])
fr = np.round(lag * FPS).astype(int)
print("hist delta frames:", dict(sorted(__import__("collections").Counter(fr).items())))
for lim in (0.033, 0.05, 0.083, 0.1, 0.15):
    print(f"recall @{int(lim*1000)}ms: {np.mean(np.abs(lag) <= lim):.3f}")
far = [(round(float(t), 2), int(l)) for t, l in zip(tc, fr) if abs(l) > 5]
print("events with |delta|>5 frames:", far)
# where do unmatched predictions sit?
un = [c["t_video"] for c in cands if np.min(np.abs(tc - c["t_video"])) > 0.083]
print("unmatched preds >83ms from any oracle event:", [round(float(t), 2) for t in un])
