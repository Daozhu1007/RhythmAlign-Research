# R5-FIX PART 0 - historical freeze.
# 1) Record SHA-256 hashes of every historical artifact this phase depends on.
# 2) Reproduce the independent review's decisive numerical checks BEFORE any
#    corrected implementation is used. Nothing downstream runs if these fail.
# No historical file is modified anywhere in this phase.
import glob
import hashlib
import os
import sys

import numpy as np
import soundfile as sf
from scipy import ndimage as ndi
from scipy import signal as sig
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixenv as fx

MANIFEST_PATH = os.path.join(fx.FIX, "HISTORICAL_BASELINE_MANIFEST.json")
CHECKS_PATH = os.path.join(fx.FIX_LOG, "historical_reproduction_checks.json")

MANIFEST_FILES = {
    "R4_golden_outputs": [
        "C0_clean_music.wav", "C1_interaction.wav", "C1_final_mix.wav",
        "C2_interaction.wav", "C2_final_mix.wav",
        "C3_interaction.wav", "C3_final_mix.wav",
        "contacts_oracle.json", "contacts_refined.json"],
    "R45_conservative_windows": sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(fx.R45_OUT, "D*.wav"))),
    "R55_frozen_config": ["detector_config_r55_frozen.json"],
    "R56_frozen_config": ["detector_config_r56_frozen.json"],
    "HoldoutD_E0_predictions": [
        "holdoutD_contacts_auto_refined.json",   # 150 refined candidates; E0 windows = strong/present/weak (123)
        "holdoutD_contacts_auto_visual.json",    # E1/E2 visual candidates + completion provenance
        "holdoutD_e0_C1_interaction.wav",        # OLD defective E0+C1 outputs
        "holdoutD_e0_C1_final_mix.wav"],
    "HoldoutD_oracle": [
        "holdoutD_oracle_blind.json", "holdoutD_oracle_refined.json",
        "holdoutD_oracle_C1_interaction.wav", "holdoutD_oracle_C1_final_mix.wav"],
    "HoldoutD_old_E2_outputs": [
        "holdoutD_auto_C1_interaction.wav", "holdoutD_auto_C1_final_mix.wav"],
    "HoldoutD_sources_and_metrics": [
        "holdoutD_raw.wav", os.path.join("..", "work", "holdoutD_ref_warp.wav"),
        os.path.join("..", "work", "holdoutD_comb.npy"),
        os.path.join("..", "logs", "holdoutD_metrics.json")],
    "Golden_sources": [
        os.path.join("experiments", "r1_golden_sample", "outputs", "golden_raw.wav"),
        os.path.join("experiments", "r1_golden_sample", "work", "ref_warp_fixed.wav")],
}
BASE_DIRS = {
    "R4_golden_outputs": fx.R4_OUT,
    "R45_conservative_windows": fx.R45_OUT,
    "R55_frozen_config": fx.R55_OUT,
    "R56_frozen_config": fx.R56_OUT,
    "HoldoutD_E0_predictions": fx.R56_OUT,
    "HoldoutD_oracle": fx.R56_OUT,
    "HoldoutD_old_E2_outputs": fx.R56_OUT,
    "HoldoutD_sources_and_metrics": fx.R56_OUT,
    "Golden_sources": fx.ROOT,
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest():
    entries = {}
    for group, names in MANIFEST_FILES.items():
        for name in names:
            p = os.path.normpath(os.path.join(BASE_DIRS[group], name))
            assert os.path.exists(p), f"missing historical artifact: {p}"
            entries[p] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
    return entries


def verify_manifest(entries):
    """Second hash pass: confirm nothing changed while checks ran."""
    for p, meta in entries.items():
        cur = sha256(p)
        assert cur == meta["sha256"], f"HISTORICAL FILE CHANGED DURING PHASE: {p}"
    return True


def load_wav(path):
    return sf.read(path, dtype="float64", always_2d=True)[0]


def check_r4_c1():
    raw = load_wav(os.path.join(fx.R1_OUT, "golden_raw.wav"))
    ref = load_wav(os.path.join(fx.R1_WORK, "ref_warp_fixed.wav"))[3 * fx.SR:18 * fx.SR]
    ev = [e for e in fx.load_json(os.path.join(fx.R4_OUT, "contacts_refined.json"))["events"]
          if e["refined_confidence"] != "no_transient" and e["type"] != "none"]
    spans = fx.spans_from_times([e["t_audio_refined"] for e in ev], 15.0)
    env = fx.envelope_from_spans_OLD(len(raw), spans, 0.010, 0.060)
    c1 = env[:, None] * raw
    old32 = sf.read(os.path.join(fx.R4_OUT, "C1_interaction.wav"), dtype="float32", always_2d=True)[0]
    mix32 = sf.read(os.path.join(fx.R4_OUT, "C1_final_mix.wav"), dtype="float32", always_2d=True)[0]
    return {
        "n_events": len(ev), "n_spans_after_merge": len(spans),
        "max_abs_err_interaction_vs_saved_wav": float(np.max(np.abs(c1.astype(np.float32) - old32))),
        "max_abs_err_final_mix_vs_saved_wav": float(np.max(np.abs((0.5 * ref + 0.5 * c1).astype(np.float32) - mix32))),
        "mean_env": float(env.mean()),
    }


def check_r4_c2():
    raw = load_wav(os.path.join(fx.R1_OUT, "golden_raw.wav"))
    ev = [e for e in fx.load_json(os.path.join(fx.R4_OUT, "contacts_refined.json"))["events"]
          if e["refined_confidence"] != "no_transient" and e["type"] != "none"]
    spans = fx.spans_from_times([e["t_audio_refined"] for e in ev], 15.0)
    env = fx.envelope_from_spans_OLD(len(raw), spans, 0.010, 0.060)
    # verbatim historical percussive_blend (R4 20_build.py) - REVERSED orientation
    def percussive_blend_OLD(raw, n_fft=2048, hop=512, floor=0.35, mask_pow=0.8, emph=0.65):
        mono = raw.mean(axis=1)
        f, t, Z = sig.stft(mono, fs=fx.SR, nperseg=n_fft, noverlap=n_fft - hop)
        Zh = ndi.median_filter(np.abs(Z), size=(31, 1), mode="reflect")
        Zp = ndi.median_filter(np.abs(Z), size=(1, 31), mode="reflect")
        M = Zp / (Zh + Zp + 1e-9)   # numerator = TIME-axis median = harmonic estimate
        g = np.clip(floor + emph * np.power(M, mask_pow), floor, 1.0)
        out = np.zeros_like(raw)
        for c in range(raw.shape[1]):
            fc, tc, Zc = sig.stft(raw[:, c], fs=fx.SR, nperseg=n_fft, noverlap=n_fft - hop)
            _, xr = sig.istft(Zc * g, fs=fx.SR, nperseg=n_fft, noverlap=n_fft - hop)
            out[: min(len(xr), len(raw)), c] = xr[: len(raw)]
        return out
    c2 = env[:, None] * percussive_blend_OLD(raw)
    old32 = sf.read(os.path.join(fx.R4_OUT, "C2_interaction.wav"), dtype="float32", always_2d=True)[0]
    return {"max_abs_err_interaction_vs_saved_wav": float(np.max(np.abs(c2.astype(np.float32) - old32)))}


def rest_stats(stem_path, raw, t0=6.0, t1=8.0):
    stem = load_wav(stem_path)
    sl = slice(int(t0 * fx.SR), int(t1 * fx.SR))
    s, r = stem[sl].ravel(), raw[sl].ravel()   # review convention: both channels pooled
    nz = float(np.mean(np.abs(stem[sl].mean(axis=1)) > 1e-6)) * (t1 - t0)
    if not np.any(s):
        e_chg = None   # digital zero (oracle): log-energy undefined, stored as null
    else:
        e_chg = float(20 * np.log10(np.sqrt(np.mean(s ** 2)) / (np.sqrt(np.mean(r ** 2)) + 1e-12)))
    return {"nonzero_mono_seconds": round(nz, 6), "energy_change_vs_raw_db": e_chg}


def check_holdoutD_rest_and_E0():
    raw = load_wav(os.path.join(fx.R56_OUT, "holdoutD_raw.wav"))
    out = {"rest_6_8": {}}
    for key, wav in (("E0", "holdoutD_e0_C1_interaction.wav"),
                     ("E2", "holdoutD_auto_C1_interaction.wav"),
                     ("oracle", "holdoutD_oracle_C1_interaction.wav")):
        out["rest_6_8"][key] = rest_stats(os.path.join(fx.R56_OUT, wav), raw)
    # E0 envelope provenance: rebuild OLD envelope from SAVED prediction JSON only
    A = fx.load_json(os.path.join(fx.R56_OUT, "holdoutD_contacts_auto_refined.json"))
    e0_t = [e["t_audio_refined"] for e in A["events"]
            if e.get("refined_confidence") in ("strong", "present", "weak")]
    spans = fx.spans_from_times(e0_t, 18.0)
    env = fx.envelope_from_spans_OLD(len(raw), spans, 0.010, 0.060)
    e0_32 = sf.read(os.path.join(fx.R56_OUT, "holdoutD_e0_C1_interaction.wav"),
                    dtype="float32", always_2d=True)[0]
    out["E0_envelope_provenance"] = {
        "n_window_times": len(e0_t), "n_spans_after_merge": len(spans),
        "max_abs_err_vs_saved_wav": float(np.max(np.abs((env[:, None] * raw).astype(np.float32) - e0_32))),
        "mean_env": float(env.mean()),
    }
    return out


def match_greedy(pred, orc, tol):
    pred, orc = np.asarray(pred), np.asarray(orc)
    mp, mc, n = set(), set(), 0
    for j, t in enumerate(pred):
        d = np.abs(orc - t)
        for k in np.argsort(d)[:4]:
            if d[k] <= tol and k not in mc:
                mc.add(k); mp.add(j); n += 1
                break
    return n


def match_max_cardinality(pred, orc, tol):
    pred, orc = np.asarray(pred), np.asarray(orc)
    C = np.full((len(pred), len(orc)), 1e3)
    dt = np.abs(pred[:, None] - orc[None, :])
    C[dt <= tol] = dt[dt <= tol]
    ri, ci = linear_sum_assignment(C)
    return int(np.sum(C[ri, ci] < 1e3))


def check_matching():
    A = fx.load_json(os.path.join(fx.R56_OUT, "holdoutD_contacts_auto_visual.json"))
    V = fx.load_json(os.path.join(fx.R56_OUT, "holdoutD_contacts_auto_visual.json"))
    pred_e2 = [c["t_video"] for c in V["candidates"]] + [a["t"] for a in V["completion_added"]]
    # E0 candidate t_video times are not stored verbatim; the historical metric
    # used the same 150 visual candidates as E2 (before E2's 4 completions).
    orc = fx.load_json(os.path.join(fx.R56_OUT, "holdoutD_oracle_blind.json"))
    tc = [e["t_video"] for e in orc["events"] if e["status"] == "ok"]
    out = {}
    for key, pred in (("E0", [c["t_video"] for c in V["candidates"]]), ("E2", pred_e2)):
        out[key] = {}
        for tol in (0.033, 0.050, 0.080):
            out[key][f"{int(tol*1000)}ms"] = {
                "greedy_matches": match_greedy(pred, tc, tol),
                "max_cardinality_matches": match_max_cardinality(pred, tc, tol),
                "n_pred": len(pred), "n_oracle": len(tc),
            }
    return out


def main():
    print("building manifest hashes ...")
    entries = build_manifest()
    manifest = {
        "phase": "R5-FIX historical freeze",
        "created": "2026-09-09",
        "purpose": ("Immutable evidence record. Historical R4-R5.6 outputs are NOT "
                    "modified by this phase; hashes must still match at any later audit."),
        "hash_algorithm": "sha256",
        "artifacts": entries,
    }
    fx.save_json(MANIFEST_PATH, manifest)
    print(f"  {len(entries)} artifacts hashed -> {MANIFEST_PATH}")

    print("reproducing review checks ...")
    checks = {
        "r4_C1": check_r4_c1(),
        "r4_C2": check_r4_c2(),
        "holdoutD": check_holdoutD_rest_and_E0(),
        "matching": check_matching(),
        "expected_from_review": {
            "r4_C1_max_abs_err": 0.0,
            "r4_C2_max_abs_err": 2.78e-17,
            "rest_nonzero_s": {"E0": 1.423604, "E2": 1.745604, "oracle": 0.0},
            "E0_envelope_provenance_max_abs_err": 0.0,
            "greedy_matches": {"E0": {"33": 40, "50": 63, "80": 78},
                               "E2": {"33": 41, "50": 64, "80": 78}},
            "max_card_matches_80ms": 80,
        },
    }
    ok = True
    c1 = checks["r4_C1"]
    ok &= c1["max_abs_err_interaction_vs_saved_wav"] == 0.0 and c1["max_abs_err_final_mix_vs_saved_wav"] < 1e-12
    ok &= checks["r4_C2"]["max_abs_err_interaction_vs_saved_wav"] < 1e-15
    r = checks["holdoutD"]["rest_6_8"]
    ok &= abs(r["E0"]["nonzero_mono_seconds"] - 1.4236041666666666) < 1e-4
    ok &= abs(r["E2"]["nonzero_mono_seconds"] - 1.7456041666666666) < 1e-4
    ok &= r["oracle"]["nonzero_mono_seconds"] == 0.0
    ok &= checks["holdoutD"]["E0_envelope_provenance"]["max_abs_err_vs_saved_wav"] == 0.0
    m = checks["matching"]
    ok &= m["E0"]["33ms"]["greedy_matches"] == 40 and m["E0"]["50ms"]["greedy_matches"] == 63
    ok &= m["E0"]["80ms"]["greedy_matches"] == 78 and m["E0"]["80ms"]["max_cardinality_matches"] == 80
    ok &= m["E2"]["50ms"]["greedy_matches"] == 64 and m["E2"]["80ms"]["max_cardinality_matches"] == 80
    checks["all_checks_pass"] = bool(ok)
    fx.save_json(CHECKS_PATH, checks)
    print(f"  checks -> {CHECKS_PATH}")
    assert verify_manifest(entries), "historical files changed during checks"
    if not ok:
        print("FAILED CHECKS:", checks)
        sys.exit(2)
    print("ALL HISTORICAL REPRODUCTION CHECKS PASS")


if __name__ == "__main__":
    main()
