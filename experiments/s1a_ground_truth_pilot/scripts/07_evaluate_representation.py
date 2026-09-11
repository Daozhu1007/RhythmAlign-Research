"""S1a 07 — representation evaluation on exact T2 cases (two axes, never one score).

For each case: RAW_MIXTURE, BOUNDED_REAL_MASK_ORACLE, OPTIMIZED_BOUNDED_MASK_ORACLE,
COMPLEX_RATIO_ORACLE, TRUE_TARGET_STFT_ROUNDTRIP vs TRUE_TARGET.
Writes REPRESENTATION_RESULTS.json (per-case + aggregates by level and family).

Usage: python 07_evaluate_representation.py [--recipes MIXTURE_RECIPES.json]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import evaluator as EV
from stft_clapsep import ClapSepSTFT

MASK_VARIANTS = ("BOUNDED_REAL_MASK_ORACLE", "OPTIMIZED_BOUNDED_MASK_ORACLE")


def evaluate_case(case: dict, st: ClapSepSTFT) -> dict:
    sr = case["sr"]
    L = int(round(case["duration_s"] * sr))
    s = C.to_mono(C.read_wav(os.path.join(C.S1A_DIR, case["files"]["s"]))[0])
    n = C.to_mono(C.read_wav(os.path.join(C.S1A_DIR, case["files"]["n"]))[0])
    y = C.to_mono(C.read_wav(os.path.join(C.S1A_DIR, case["files"]["y"]))[0])
    events = case["target"].get("events") or []
    friction = case["target"].get("friction_intervals") or []

    Y, S, N = st.stft(y), st.stft(s), st.stft(n)
    out = {"id": case["id"], "levels_db": case["requested_levels_db"],
           "families": [nz["family"] for nz in case["nuisances"]], "variants": {}}
    for v, path in case_target_paths(case).items():
        x = C.to_mono(C.read_wav(os.path.join(C.S1A_DIR, path))[0]).astype(np.float64)
        rec = EV.evaluate_variant(x, s, sr, events=events, friction=friction)
        rec["output_to_mixture_energy_db"] = C.db(
            (np.dot(x, x) + EV.EPS ** 2) / (np.dot(y, y) + EV.EPS ** 2))
        out["variants"][v] = rec

    # exact linear path decomposition for mask oracles (linearity of the mask path)
    for v, mask_file in (("BOUNDED_REAL_MASK_ORACLE", "mask_bounded.npy"),
                         ("OPTIMIZED_BOUNDED_MASK_ORACLE", "mask_optimized.npy")):
        if v not in out["variants"]:
            continue
        mp = os.path.join(C.S1A_DIR, "work", "representation", case["id"], mask_file)
        assert os.path.exists(mp), f"missing {mask_file}; run 06 first"
        M = np.load(mp)
        out["variants"][v]["exact_path_decomposition"] = EV.exact_path_decomposition(
            st, M, Y, S, N, n, L)

    # raw-mixture suppression reference = 0 dB attenuation by definition
    out["variants"]["RAW_MIXTURE"]["nuisance_projection"] = EV.nuisance_projection(y, s, [n])
    for v in MASK_VARIANTS + ("COMPLEX_RATIO_ORACLE",):
        if v in out["variants"]:
            x = C.to_mono(C.read_wav(os.path.join(
                C.S1A_DIR, case_target_paths(case)[v]))[0]).astype(np.float64)
            out["variants"][v]["nuisance_projection"] = EV.nuisance_projection(x, s, [n])
    return out


def case_target_paths(case: dict) -> dict:
    base = os.path.join(C.S1A_DIR, "work", "representation", case["id"])
    return {
        "RAW_MIXTURE": case["files"]["y"],
        "BOUNDED_REAL_MASK_ORACLE":
            os.path.relpath(os.path.join(base, "bounded_real_mask_oracle.wav"),
                            C.S1A_DIR).replace("\\", "/"),
        "OPTIMIZED_BOUNDED_MASK_ORACLE":
            os.path.relpath(os.path.join(base, "optimized_bounded_mask_oracle.wav"),
                            C.S1A_DIR).replace("\\", "/"),
        "COMPLEX_RATIO_ORACLE":
            os.path.relpath(os.path.join(base, "complex_ratio_oracle.wav"),
                            C.S1A_DIR).replace("\\", "/"),
        "TRUE_TARGET_STFT_ROUNDTRIP":
            os.path.relpath(os.path.join(base, "true_target_stft_roundtrip.wav"),
                            C.S1A_DIR).replace("\\", "/"),
        "TRUE_TARGET": case["files"]["s"],
    }


def aggregate(results: list) -> dict:
    """Medians by variant + by requested target level (fixture/real agnostic)."""
    variants = sorted({v for r in results for v in r["variants"]})
    agg = {}
    for v in variants:
        rows = [r["variants"][v] for r in results if v in r["variants"]]
        agg[v] = {
            "n": len(rows),
            "recon_snr_db_median": _med([r.get("recon_snr_db") for r in rows]),
            "si_sdr_db_median": _med([r.get("si_sdr_db") for r in rows]),
            "projection_gain_db_median": _med([r.get("projection_gain_db") for r in rows]),
            "log_mag_l1_median": _med([r.get("mrstft", {}).get("log_mag_l1_mean")
                                       for r in rows]),
            "hf_ratio_median": _med([r.get("hf_retention", {}).get("8000_16000_ratio")
                                     for r in rows]),
            "nuisance_attenuation_db_median": _med(
                [r.get("exact_path_decomposition", {}).get("nuisance_attenuation_db")
                 for r in rows]),
            "nuisance_proj_coeff_median": _med(
                [abs(r.get("nuisance_projection", {}).get("coefficients", [0, 1])[1])
                 if r.get("nuisance_projection") else None for r in rows]),
        }
    by_level = {}
    levels = sorted({lv for r in results for lv in r["levels_db"]})
    for lv in levels:
        sub = [r for r in results if lv in r["levels_db"]]
        by_level[str(lv)] = {
            v: {
                "recon_snr_db_median": _med([r["variants"][v].get("recon_snr_db")
                                             for r in sub if v in r["variants"]]),
                "projection_gain_db_median": _med(
                    [r["variants"][v].get("projection_gain_db")
                     for r in sub if v in r["variants"]]),
            } for v in variants if any(v in r["variants"] for r in sub)
        }
    return {"by_variant": agg, "by_level": by_level}


def _med(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", default=os.path.join(C.S1A_DIR, "MIXTURE_RECIPES.json"))
    args = ap.parse_args()
    recipes = C.load_json(args.recipes)
    st = ClapSepSTFT()
    results = [evaluate_case(c, st) for c in recipes["cases"]]
    out = {
        "stage": "s1a",
        "data_class": recipes.get("data_class", "REAL_CAPTURE"),
        "metric_eps": EV.EPS,
        "gain_convention": "fixed; no per-output normalization; no SI-SDR-only claims",
        "results": results,
        "aggregates": aggregate(results),
    }
    C.save_json(out, os.path.join(C.S1A_DIR, "REPRESENTATION_RESULTS.json"))
    C.log(f"evaluated {len(results)} cases -> REPRESENTATION_RESULTS.json")
    for v, rec in out["aggregates"]["by_variant"].items():
        C.log(f"  {v}: SNR {rec['recon_snr_db_median']}, "
              f"gain {rec['projection_gain_db_median']} dB")


if __name__ == "__main__":
    main()
