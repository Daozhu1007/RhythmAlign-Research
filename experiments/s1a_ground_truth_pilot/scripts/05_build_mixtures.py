"""S1a 05 — exact T2 mixture construction from a recipe file.

Recipe JSON schema (list under "cases"):
{
  "cases": [
    {
      "id": "case01",
      "target": {"path": "...wav", "offset_s": 0.0, "duration_s": 5.0,
                 "events": [{"t0":..,"t1":..,"label":"tap"}, ...],
                 "friction_intervals": [...]},           # events optional
      "nuisances": [
         {"path": "...wav", "offset_s": 0.0, "level_db": -10.0,
          "family": "music|speech|ambience|impact|composite",
          "path_transform": {"rir_wav": "...wav", "gain_db": 0.0} | null}
      ],
      "level_convention": "target active-region RMS vs nuisance RMS in same region"
    }
  ]
}

Construction (fixed gain, no normalization):
  y = s + sum_j g_j * n_j   with g_j set so that
  rms(s | active target) / rms(n_j | active target) = 10^(level_db_j / 20).

Outputs: <out>/s_<id>.wav, n_<id>.wav (sum of scaled nuisances), y_<id>.wav (float32),
MIXTURE_RECIPES.json with per-case components, gains, hashes, measured levels, and a
numerical sum identity check on the STORED float32 files.

Usage: python 05_build_mixtures.py --recipe <recipes.json> --out <dir>
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ID_RMS_FLOOR = 1e-8  # active-region RMS floor (documented; never tuned post hoc)


def _load_segment(path: str, offset_s: float, dur_s: float, sr: int) -> np.ndarray:
    x, x_sr = C.read_wav(path)
    x = C.to_mono(x)
    if x_sr != sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(x_sr, sr)
        x = resample_poly(x, sr // g, x_sr // g, window=("kaiser", 8.0))
    i0 = int(round(offset_s * sr))
    i1 = i0 + int(round(dur_s * sr))
    seg = x[i0:i1]
    if len(seg) < i1 - i0:
        seg = np.pad(seg, (0, i1 - i0 - len(seg)))
    return seg


def _rms_active(x: np.ndarray, active_mask: np.ndarray) -> float:
    if active_mask is None or not np.any(active_mask):
        return float(np.sqrt(np.mean(x ** 2) + 0.0))
    return float(np.sqrt(np.mean(x[active_mask] ** 2)))


def build_case(case: dict, sr: int, out_dir: str) -> dict:
    tgt = case["target"]
    s = _load_segment(tgt["path"], tgt.get("offset_s", 0.0), tgt["duration_s"], sr)
    n_tot = len(s)
    ev = tgt.get("events") or []
    active = np.zeros(n_tot, dtype=bool)
    for e in ev:
        active[int(e["t0"] * sr):int(np.ceil(e["t1"] * sr))] = True
    if not np.any(active):  # whole clip active (documented convention)
        active = np.ones(n_tot, dtype=bool)

    rms_s = _rms_active(s, active)
    nuis_parts, nuis_meta = [], []
    for j, nz in enumerate(case["nuisances"]):
        n_j = _load_segment(nz["path"], nz.get("offset_s", 0.0), tgt["duration_s"], sr)
        if nz.get("path_transform"):
            pt = nz["path_transform"]
            n_j = apply_path_transform(n_j, sr, pt)
        rms_n = _rms_active(n_j, active)
        level = float(nz["level_db"])
        gain = (rms_s / max(rms_n, ID_RMS_FLOOR)) * 10.0 ** (-level / 20.0)
        nuis_parts.append(gain * n_j)
        nuis_meta.append({
            "path": C.relpath_safe(nz["path"], C.S1A_DIR),
            "sha256": C.sha256_file(nz["path"]),
            "offset_s": nz.get("offset_s", 0.0), "family": nz.get("family"),
            "requested_level_db": level, "applied_gain": float(gain),
            "path_transform": nz.get("path_transform"),
            "rms_active_nuisance": rms_n,
        })

    n_sum = np.sum(nuis_parts, axis=0) if nuis_parts else np.zeros_like(s)
    y = s + n_sum
    C.write_wav(os.path.join(out_dir, f"s_{case['id']}.wav"), s, sr)
    C.write_wav(os.path.join(out_dir, f"n_{case['id']}.wav"), n_sum, sr)
    C.write_wav(os.path.join(out_dir, f"y_{case['id']}.wav"), y, sr)

    # read back and verify the sum identity on STORED float32 values
    s_r = C.to_mono(C.read_wav(os.path.join(out_dir, f"s_{case['id']}.wav"))[0])
    n_r = C.to_mono(C.read_wav(os.path.join(out_dir, f"n_{case['id']}.wav"))[0])
    y_r = C.to_mono(C.read_wav(os.path.join(out_dir, f"y_{case['id']}.wav"))[0])
    sum_err = float(np.max(np.abs(y_r - (s_r + n_r))))
    measured = []
    for j, nz in enumerate(case["nuisances"]):
        meas = 20.0 * np.log10(rms_s / max(_rms_active(nuis_parts[j], active),
                                           ID_RMS_FLOOR))
        measured.append(float(meas))
    return {
        "id": case["id"],
        "truth_tier": "T2",
        "duration_s": tgt["duration_s"], "sr": sr,
        "target": {"path": C.relpath_safe(tgt["path"], C.S1A_DIR),
                   "sha256": C.sha256_file(tgt["path"]),
                   "offset_s": tgt.get("offset_s", 0.0),
                   "rms_active": rms_s,
                   "events": ev,
                   "friction_intervals": tgt.get("friction_intervals", [])},
        "nuisances": nuis_meta,
        "requested_levels_db": [float(nz["level_db"]) for nz in case["nuisances"]],
        "measured_levels_db": measured,
        "active_convention": "events" if np.any(ev) else "whole_clip",
        "rms_floor": ID_RMS_FLOOR,
        "files": {k: C.relpath_safe(os.path.join(out_dir, f"{k}_{case['id']}.wav"), C.S1A_DIR)
                  for k in ("s", "n", "y")},
        "file_sha256": {k: C.sha256_file(os.path.join(out_dir, f"{k}_{case['id']}.wav"))
                        for k in ("s", "n", "y")},
        "stored_float32_sum_max_abs_err": sum_err,
        "sum_identity_ok": bool(sum_err <= 1e-6),
    }


def apply_path_transform(x: np.ndarray, sr: int, pt: dict) -> np.ndarray:
    """Documented acoustic path transform: RIR convolution + static gain (no dynamics)."""
    if pt.get("rir_wav"):
        rir, rir_sr = C.read_wav(pt["rir_wav"])
        rir = C.to_mono(rir)
        if rir_sr != sr:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(rir_sr, sr)
            rir = resample_poly(rir, sr // g, rir_sr // g, window=("kaiser", 8.0))
        if pt.get("rir_norm_peak"):
            rir = rir / (np.max(np.abs(rir)) + 1e-12)
        x = np.convolve(x, rir)[: len(x)]
    if pt.get("gain_db"):
        x = x * 10.0 ** (float(pt["gain_db"]) / 20.0)
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recipe = C.load_json(args.recipe)
    sr = int(recipe.get("sr", C.WORK_SR))
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    records = [build_case(c, sr, out_dir) for c in recipe["cases"]]
    out = {
        "stage": "s1a", "sr": sr,
        "data_class": recipe.get("data_class", "REAL_CAPTURE"),
        "gain_convention": "fixed; target stored at recorded gain; no normalization",
        "level_definition": "rms(s|active)/rms(n_j|active) = 10^(level/20); "
                            "active = annotated target events (else whole clip)",
        "n_cases": len(records),
        "cases": records,
    }
    C.save_json(out, os.path.join(C.S1A_DIR, "MIXTURE_RECIPES.json"))
    C.log(f"built {len(records)} T2 mixtures -> {out_dir}")
    for r in records:
        C.log(f"  {r['id']}: sum_ok={r['sum_identity_ok']} "
              f"(err {r['stored_float32_sum_max_abs_err']:.2e})")
    assert all(r["sum_identity_ok"] for r in records), "STOP: mixture sum identity failed"


if __name__ == "__main__":
    main()
