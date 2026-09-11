# R5.5 common: window loading, dense/sparse split, evaluation helpers.
# Feature extraction / C1 / refinement primitives are imported UNCHANGED from
# R5 (frozen visual feature definitions + frozen audio pipeline).
import importlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
R55 = os.path.abspath(os.path.join(_HERE, ".."))
R55_WORK = os.path.join(R55, "work")
R55_OUT = os.path.join(R55, "outputs")
R55_LOG = os.path.join(R55, "logs")
R5_SCRIPTS = r"D:\Code\RhythmAlign\experiments\r5_auto_contact_detection\scripts"
R5_OUT = os.path.join(R5_SCRIPTS, "..", "outputs")
R5_WORK = os.path.join(R5_SCRIPTS, "..", "work")
R4_OUT = r"D:\Code\RhythmAlign\experiments\r4_contact_reconstruction\outputs"
R1_WORK = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\work"
HANCAM_MP4 = r"D:\Daozh\Videos\舞萌手元\13.2\共感觉\AP\共感怪物AP.mp4"

sys.path.insert(0, R5_SCRIPTS)
_r5 = importlib.import_module("r5_common")

FPS = 60.04
MATCH_TOLS = (0.033, 0.050)
DENSE_PAD_S = 0.15   # a contact is DENSE if inside a continuous region +/- this pad

# R5 FROZEN detector config (outputs/detector_config_frozen.json) - baseline only
R5_FROZEN_CFG = {
    "smooth_frames": 3, "rise_frames": 2,
    "w_jz_text": 1.8, "w_jz_diff": 1.5, "w_glass_bright": 1.0, "w_hand_diff": 0.3,
    "threshold": 0.40, "slope_min": 0.15, "min_dist_frames": 3,
    "norm_pct": 99.5, "novelty_clip": 2.5,
}


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_json(p, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=float)


def r5_module(name):
    return importlib.import_module(name)


# re-export the frozen primitives so every script uses identical code
extract_features = _r5.extract_features
detect_candidates = _r5.detect_candidates
detector_score = _r5.detector_score
comb_envelope = _r5.comb_envelope
refine_candidates = _r5.refine_candidates
build_c1 = _r5.build_c1
audio_onsets = _r5.audio_onsets
ffmpeg_run = _r5.ffmpeg_run
iter_frames = _r5.iter_frames


class Window:
    """A labeled dev/holdout window: feature series + oracle + dense mask."""

    def __init__(self, key, features_npz, oracle_json, t0_s, t1_s, dur_s):
        self.key = key
        d = np.load(features_npz)
        self.series = {k: d[k] for k in d.files if k != "fps"}
        self.n = len(self.series["jz_diff"])
        self.dur = dur_s
        self.t0_native = t0_s
        o = load_json(oracle_json)
        self.oracle = o
        ev = o["events"]
        self.contacts = [e for e in ev
                         if e["type"] in ("press", "touch", "slide")
                         and e.get("status") != "rejected_noncontact"]
        self.noncontacts = [e for e in ev
                            if e["type"] in ("release", "rest", "none")
                            or e.get("status") == "rejected_noncontact"]
        self.regions = o["regions"]
        self.continuous = [r for r in self.regions if not r["id"].startswith("G")]
        # dense mask over frames: inside any continuous region (no pad: regions
        # are declared continuous spans; pad handled at contact level)
        m = np.zeros(self.n, bool)
        for r in self.continuous:
            a = max(int(r["t0"] * FPS), 0)
            b = min(int(np.ceil(r["t1"] * FPS)), self.n)
            m[a:b] = True
        self.dense_mask = m
        for e in self.contacts:
            e["dense"] = bool(self._in_continuous(e["t_video"], DENSE_PAD_S))

    def _in_continuous(self, t, pad):
        return any(r["t0"] - pad <= t <= r["t1"] + pad for r in self.continuous)

    def tc(self):
        return np.array([e["t_video"] for e in self.contacts])


def load_windows():
    return [
        Window("devA",
               os.path.join(R5_WORK, "dev_features.npz"),
               os.path.join(R4_OUT, "contacts_oracle.json"),
               22.5, 37.5, 15.0),
        Window("devB",
               os.path.join(R5_WORK, "holdout_features.npz"),
               os.path.join(R5_OUT, "holdout_oracle_blind.json"),
               84.0, 102.0, 18.0),
    ]


def match_one_to_one(tp_times, oracle_times, tol):
    """Greedy nearest one-to-one matching (same policy as R5 eval)."""
    tp_times = np.asarray(tp_times)
    oracle_times = np.asarray(oracle_times)
    matched_c, matched_p, pairs = set(), set(), []
    for j, t in enumerate(tp_times):
        d = np.abs(oracle_times - t)
        order = np.argsort(d)
        for k in order[:4]:
            if d[k] <= tol and k not in matched_c:
                matched_c.add(k)
                matched_p.add(j)
                pairs.append((j, k, float(tp_times[j] - oracle_times[k])))
                break
    return matched_c, matched_p, pairs


def prf(match, n_pred, n_oracle):
    p = match / max(n_pred, 1)
    r = match / max(n_oracle, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def split_sparse_dense(contacts):
    return ([e for e in contacts if not e["dense"]],
            [e for e in contacts if e["dense"]])
