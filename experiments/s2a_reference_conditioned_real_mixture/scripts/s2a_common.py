# S2A shared library: reference-conditioned real-mixture extraction stage.
#
# Reuses (read-only):
#   - S1W frozen split + provenance (work/private/DATA_SPLIT.private.json, seed 20260912)
#   - S1W raw32k mono extractions of ORIGINAL_RAW_HANDCAM recordings
#   - S1R phase-0 anchor windows + window bank conventions
#   - production RhythmAlign alignment logic (auto_sync, read-only import)
#
# S2A adds ONE new input: the time-aligned pristine song reference that
# RhythmAlign natively operates with (product input, not an oracle). Training
# remains on authentic raw handcam mixtures (protocol sections 8/17).
import os
import json

S2A = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
S1R = os.path.abspath(os.path.join(S2A, "..", "s1r_real_mixture_adaptation"))
S1W = os.path.abspath(os.path.join(S2A, "..", "s1w_existing_corpus_adaptation"))

import sys  # noqa: E402
sys.path.insert(0, os.path.join(S1R, "scripts"))

S2A_PRIVATE = os.path.join(S2A, "work", "private")
S2A_REF22K = os.path.join(S2A, "work", "audio", "ref22k")
S2A_REF32K = os.path.join(S2A, "work", "audio", "ref32k")
S1W_PRIVATE = os.path.join(S1W, "work", "private")
RAW32K_DIR = os.path.join(S1W, "work", "audio", "raw32k")

S1R_CKPT = os.path.join(S1R, "checkpoints", "s1r_selected.ckpt")
S1R_CKPT_SHA256 = "e1fade5ebeb84e75527d41bb5aba48dc197243a49aa34cb2fee673897a1cc303"

SR = 32000           # training/inference rate (S1R convention)
ALIGN_SR = 22050     # production alignment rate
ALIGN_HOP = 512      # production alignment hop

# Protocol section 10/26: alignment acceptance (conservative)
ALIGN_Z_MIN = 2.0            # production confidence threshold (_CONFIDENCE_THRESHOLD)
ALIGN_ONSET_MAX_DIFF_S = 0.25   # |offset_hybrid - offset_onset|
ALIGN_DRIFT_RESID_MAX_S = 0.05  # max |window offset - global offset|
ALIGN_OVERLAP_MIN_S = 60.0      # minimum ref-in-handcam overlap
SPECTRAL_AGREE_MIN = 0.35        # aligned log-mel agreement (song-identity check)

# Protocol section 11 gate thresholds
GATE_PREF_TRAIN_RECS, GATE_PREF_TRAIN_SONGS = 12, 8
GATE_PREF_DEV_RECS, GATE_PREF_TEST_RECS = 3, 4
GATE_MIN_TRAIN_RECS, GATE_MIN_TRAIN_SONGS = 6, 4
GATE_MIN_DEV_RECS, GATE_MIN_TEST_RECS = 2, 2


def load_split():
    with open(os.path.join(S1W_PRIVATE, "DATA_SPLIT.private.json"), encoding="utf-8") as f:
        raw = json.load(f)
    recs = []
    for key in ("sealed_primary", "test_generalization", "dev", "train"):
        for r in raw[key]:
            r["split"] = r.get("split", key.upper())
            recs.append(r)
    return {"seed": raw["seed"], "recordings": recs}


def load_reference_map():
    with open(os.path.join(S2A_PRIVATE, "REFERENCE_MAP.private.json"), encoding="utf-8") as f:
        return json.load(f)


def load_alignment():
    with open(os.path.join(S2A_PRIVATE, "ALIGNMENT.private.json"), encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def ref_audio32k(rec_entry):
    """Aligned pristine reference as 32 kHz mono float32 (decoded via ffmpeg)."""
    import soundfile as sf

    path = os.path.join(S2A_REF32K, rec_entry["ref_id"] + ".wav")
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == SR, (path, sr)
    return x


def ref_position_s(entry, t_hand_s):
    """Reference time (s) for handcam time t_hand_s.

    Production convention (R1/S1R sealed test): offset = t_handcam - t_ref, so
    t_ref = t_hand - offset (drift-aware; all accepted pairs have drift 0).
    """
    al = entry.get("alignment", entry)
    off = al["offset_s"]
    drift = al.get("drift_rate_s_per_s", 0.0)
    return t_hand_s - (off + drift * t_hand_s)


def ref_segment(entry, ref32k, t0_hand, t1_hand):
    """Aligned reference segment for handcam [t0, t1) seconds; zero if out of range.

    Returned segment has the same length as the requested span; regions outside
    the reference's time support are zero (handled by masks upstream).
    """
    n = int(round((t1_hand - t0_hand) * SR))
    t0_ref = ref_position_s(entry, t0_hand)
    i0 = int(round(t0_ref * SR))
    out = np_zero(n)
    if i0 + n <= 0 or i0 >= len(ref32k):
        return out, 0.0
    lo, hi = max(0, i0), min(len(ref32k), i0 + n)
    out[lo - i0:hi - i0] = ref32k[lo:hi]
    cov = (hi - lo) / max(1, n)
    return out, cov


import numpy as np  # noqa: E402


def np_zero(n):
    return np.zeros(n, dtype=np.float32)
