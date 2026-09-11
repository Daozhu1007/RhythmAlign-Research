"""S1a 08 — small blind listening pack (representative T2 cases, anonymous files).

Selects a few representative cases, writes anonymous WAVs for
raw mixture / bounded oracle / optimized oracle / true-target anchor (+ complex oracle
selectively), randomizes order with a FIXED seed, stores the mapping ONLY in
listening_key.json. No loudness normalization; fixed gain for all variants.

Usage: python 08_listening_pack.py [--recipes MIXTURE_RECIPES.json] [--n-cases 4]
"""
import argparse
import glob
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

SEED = 20260910  # fixed; do not change after any pack has been distributed
VARIANT_LABELS = {
    "RAW_MIXTURE": "raw mixture",
    "BOUNDED_REAL_MASK_ORACLE": "bounded-mask oracle",
    "OPTIMIZED_BOUNDED_MASK_ORACLE": "optimized bounded-mask oracle",
    "COMPLEX_RATIO_ORACLE": "complex-ratio oracle",
    "TRUE_TARGET": "true target anchor",
}


def pick_cases(recipes: dict, n_cases: int) -> list:
    """Deterministic representative pick: worst/median/best optimized-oracle SNR + weakest."""
    if recipes.get("data_class") != "REAL_CAPTURE":
        # fixtures: still pick deterministically from the panel
        cases = recipes["cases"]
        return cases[:n_cases]
    res = C.load_json(os.path.join(C.S1A_DIR, "REPRESENTATION_RESULTS.json"))
    by_id = {r["id"]: r for r in res["results"]}
    def opt_snr(c):
        v = by_id[c["id"]]["variants"].get("OPTIMIZED_BOUNDED_MASK_ORACLE")
        return v["recon_snr_db"] if v else 0.0
    ranked = sorted(recipes["cases"], key=opt_snr)
    picks = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]  # worst/median/best
    weakest = min(ranked, key=lambda c: min(c["requested_levels_db"]))
    for c in (weakest,):
        if c not in picks:
            picks.append(c)
    return picks[:n_cases]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", default=os.path.join(C.S1A_DIR, "MIXTURE_RECIPES.json"))
    ap.add_argument("--n-cases", type=int, default=4)
    args = ap.parse_args()
    recipes = C.load_json(args.recipes)
    cases = pick_cases(recipes, args.n_cases)
    base = os.path.join(C.S1A_DIR, "work", "representation")

    rng = random.Random(SEED)
    os.makedirs(C.LISTENING_DIR, exist_ok=True)
    plan = []
    for case in cases:
        variants = ["RAW_MIXTURE", "BOUNDED_REAL_MASK_ORACLE",
                    "OPTIMIZED_BOUNDED_MASK_ORACLE", "TRUE_TARGET"]
        if any(nz["family"] == "music" and nz["requested_level_db"] <= -10
               for nz in case["nuisances"]):
            variants.append("COMPLEX_RATIO_ORACLE")  # selective inclusion
        for v in variants:
            src = (case["files"][{"RAW_MIXTURE": "y", "TRUE_TARGET": "s"}.get(v, "")]
                   if v in ("RAW_MIXTURE", "TRUE_TARGET")
                   else os.path.join(base, case["id"], f"{v.lower()}.wav"))
            plan.append({"case_id": case["id"], "variant": v, "src": src})
    rng.shuffle(plan)  # anonymous order AND anonymous names
    key, entries = [], []
    for i, item in enumerate(plan):
        anon = f"X{i + 1:02d}.wav"  # neutral name; no variant hint
        x, sr = C.read_wav(os.path.join(C.S1A_DIR, item["src"]))
        C.write_wav(os.path.join(C.LISTENING_DIR, anon),
                    C.to_mono(x).astype(np.float32), sr)
        key.append({"file": anon, "case_id": item["case_id"],
                    "variant": item["variant"]})
        entries.append(anon)
    C.save_json({"seed": SEED, "note": "DO NOT OPEN before rating. Mapping of anonymous "
                                      "files to variants.", "key": key},
                os.path.join(C.S1A_DIR, "listening_key.json"))
    C.log(f"listening pack: {len(entries)} anonymous files -> listening_pack/ "
          f"(key in listening_key.json, seed {SEED})")


if __name__ == "__main__":
    main()
