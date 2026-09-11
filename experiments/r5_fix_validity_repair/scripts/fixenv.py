# R5-FIX shared environment/paths and the two envelope implementations.
# The OLD implementation is copied VERBATIM from
# experiments/r5_auto_contact_detection/scripts/r5_common.py (== R4 20_build.py)
# and is used ONLY to reconstruct/verify historical artifacts and to produce
# old-vs-corrected diagnostics. The CORRECTED implementation is the intended
# C1 policy: pre 15 ms / post 150 ms, attack 10 ms rise 0->1, release 60 ms
# fall 1->0, overlap merge by pointwise max, gain exactly 0 outside support.
import json
import os

import numpy as np

FIX = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIX_WORK = os.path.join(FIX, "work")
FIX_OUT = os.path.join(FIX, "outputs")
FIX_LOG = os.path.join(FIX, "logs")
FIX_LISTEN = os.path.join(FIX, "listening_pack")
for d in (FIX_WORK, FIX_OUT, FIX_LOG, FIX_LISTEN):
    os.makedirs(d, exist_ok=True)

ROOT = os.path.abspath(os.path.join(FIX, "..", ".."))
R1_OUT = os.path.join(ROOT, "experiments", "r1_golden_sample", "outputs")
R1_WORK = os.path.join(ROOT, "experiments", "r1_golden_sample", "work")
R4_OUT = os.path.join(ROOT, "experiments", "r4_contact_reconstruction", "outputs")
R4_WORK = os.path.join(ROOT, "experiments", "r4_contact_reconstruction", "work")
R45_OUT = os.path.join(ROOT, "experiments", "r45_conservative_windows", "outputs")
R5_OUT = os.path.join(ROOT, "experiments", "r5_auto_contact_detection", "outputs")
R55_OUT = os.path.join(ROOT, "experiments", "r55_burst_event_splitting", "outputs")
R56_OUT = os.path.join(ROOT, "experiments", "r56_precision_cleanup", "outputs")
R56_WORK = os.path.join(ROOT, "experiments", "r56_precision_cleanup", "work")

SR = 48000
C1_PRE_S, C1_POST_S = 0.015, 0.150
C1_ATTACK_S, C1_RELEASE_S = 0.010, 0.060


def save_json(p, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=float)


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def merge_spans(spans):
    """Verbatim historical merge (R4 20_build.py): merge when next start <=
    current end, extend to the max end."""
    spans = sorted(spans)
    out = []
    for s, e in spans:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def spans_from_times(t_refined, dur, pre=C1_PRE_S, post=C1_POST_S):
    spans = []
    for t in t_refined:
        spans.append((max(t - pre, 0.0), min(t + post, dur)))
    return merge_spans([(int(s * SR), int(e * SR)) for s, e in spans])


def _ramp_up(n):
    """0 -> 1 raised cosine (verbatim historical helper)."""
    return 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, max(n, 2)))


def _ramp_dn(n):
    """1 -> 0 raised cosine (verbatim historical helper)."""
    return 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, max(n, 2)))


# ------------------------------------------------------------------
# OLD (defective, frozen as historical evidence) - verbatim from
# r5_common.envelope_from_spans / R4 20_build.envelope_from_spans.
# Defect: attack and release ramps are swapped.
# ------------------------------------------------------------------
def envelope_from_spans_OLD(n, spans, attack, release):
    env = np.zeros(n)
    A, Rn = max(int(attack * SR), 2), max(int(release * SR), 2)
    for s, e in spans:
        s, e = max(s, 0), min(e, n)
        if e <= s:
            continue
        seg = np.ones(e - s)
        a = min(A, len(seg))
        seg[:a] = np.minimum(seg[:a], _ramp_dn(a))          # defect: falls 1->0 at entry
        r = min(Rn, len(seg))
        seg[len(seg) - r:] = np.minimum(seg[len(seg) - r:], _ramp_up(r))  # defect: rises 0->1 at exit
        env[s:e] = np.maximum(env[s:e], seg)
    return env


# ------------------------------------------------------------------
# CORRECTED implementation of the SAME window policy.
# outside: 0; entry: 0->1 over attack; interior: 1; exit: 1->0 over release.
# ------------------------------------------------------------------
def envelope_from_spans_FIXED(n, spans, attack, release):
    env = np.zeros(n)
    A, Rn = max(int(attack * SR), 2), max(int(release * SR), 2)
    for s, e in spans:
        s, e = max(s, 0), min(e, n)
        if e <= s:
            continue
        seg = np.ones(e - s)
        a = min(A, len(seg))
        seg[:a] = np.minimum(seg[:a], _ramp_up(a))          # entry rises 0 -> 1
        r = min(Rn, len(seg))
        seg[len(seg) - r:] = np.minimum(seg[len(seg) - r:], _ramp_dn(r))  # exit falls 1 -> 0
        env[s:e] = np.maximum(env[s:e], seg)
    return env


def build_interaction(raw, env):
    return env[:, None] * raw


def final_mix(ref, interaction):
    """Fixed gain convention, no limiter/compressor/normalization."""
    return 0.5 * ref + 0.5 * interaction
