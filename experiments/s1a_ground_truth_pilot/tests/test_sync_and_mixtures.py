"""Sync map fitting + mixture construction tests."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import sync_fit as SF


def test_clock_map_recovers_known_offset_and_drift():
    rng = np.random.default_rng(0)
    t_aux = np.array([0.5, 10.0, 20.0, 30.0])
    a_true, b_true = 0.013, 1.0 + 70e-6
    t_tgt = a_true + b_true * t_aux + rng.standard_normal(4) * 1e-5
    m = SF.fit_clock_map(list(zip(t_aux, t_tgt)))
    assert abs(m["a_s"] - a_true) < 1e-4
    assert abs(m["drift_ppm"] - 70.0) < 1.0
    assert m["leave_one_out_max_abs_err_s"] < 1e-3
    assert SF.sync_verdict(m, 0.002)["sync_ok"]


def test_clock_map_flags_poor_sync():
    t_aux = np.array([0.5, 10.0, 20.0, 30.0])
    t_tgt = 0.013 + 1.00007 * t_aux + np.array([0, 0.01, 0, 0.01])  # 10 ms outliers
    m = SF.fit_clock_map(list(zip(t_aux, t_tgt)))
    assert not SF.sync_verdict(m, 0.002)["sync_ok"]


def test_onset_picker_finds_clap():
    sr = 32000
    x = np.zeros(int(5 * sr))
    i0 = int(2.345 * sr)
    rng = np.random.default_rng(1)
    x[i0:i0 + int(0.05 * sr)] = 0.8 * rng.standard_normal(int(0.05 * sr)) * \
        np.exp(-np.arange(int(0.05 * sr)) / (0.01 * sr))
    hit = SF.find_onset(x, sr, 2.3)
    assert hit is not None and abs(hit["t"] - 2.345) < 0.02
    assert hit["peakiness"] > 5


def test_onset_picker_none_on_silence():
    hit = SF.find_onset(np.zeros(int(3 * 32000)), 32000, 1.5)
    assert hit is None


def test_mixture_level_and_sum_identity(tmp_path):
    import common as C
    sr = 8000
    rng = np.random.default_rng(2)
    dur = 2.0
    t = np.arange(int(dur * sr)) / sr
    s = 0.3 * np.sin(2 * np.pi * 500 * t) * (t > 0.5)   # active only after 0.5 s
    n_src = rng.standard_normal(len(t)) * 0.5
    p_s = tmp_path / "s_src.wav"; p_n = tmp_path / "n_src.wav"
    C.write_wav(str(p_s), s, sr); C.write_wav(str(p_n), n_src, sr)
    recipe = {"sr": sr, "cases": [{
        "id": "t1",
        "target": {"path": str(p_s), "offset_s": 0.0, "duration_s": dur,
                   "events": [{"t0": 0.5, "t1": 2.0, "label": "tone"}]},
        "nuisances": [{"path": str(p_n), "offset_s": 0.0, "level_db": -10.0,
                       "family": "test"}],
    }]}
    import json
    rp = tmp_path / "recipe.json"
    rp.write_text(json.dumps(recipe))
    # module filename starts with a digit; load via importlib machinery
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_mixtures", os.path.join(os.path.dirname(__file__), "..", "scripts",
                                       "05_build_mixtures.py"))
    bm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bm)
    rec = bm.build_case(recipe["cases"][0], sr, str(tmp_path / "out"))
    assert rec["sum_identity_ok"]
    meas = rec["measured_levels_db"][0]
    assert meas == pytest.approx(-10.0, abs=0.1)
    # level defined on the active region only
    y = C.to_mono(C.read_wav(os.path.join(str(tmp_path / "out"), "y_t1.wav"))[0])
    assert len(y) == int(dur * sr)
