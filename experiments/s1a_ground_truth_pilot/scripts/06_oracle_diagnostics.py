"""S1a 06 — oracle diagnostics for every T2 case in MIXTURE_RECIPES.json.

Runs BOUNDED_REAL_MASK_ORACLE, OPTIMIZED_BOUNDED_MASK_ORACLE (fixed L-BFGS-B budget,
deterministic init) and COMPLEX_RATIO_ORACLE with the exact CLAPSep STFT/ISTFT
convention; writes per-case outputs (fixed gain, no normalization) and a run log.

Usage: python 06_oracle_diagnostics.py [--recipes MIXTURE_RECIPES.json]
                                       [--opt-iters 150]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from stft_clapsep import ClapSepSTFT, run_oracles

VARIANTS = ["RAW_MIXTURE", "BOUNDED_REAL_MASK_ORACLE",
            "OPTIMIZED_BOUNDED_MASK_ORACLE", "COMPLEX_RATIO_ORACLE",
            "TRUE_TARGET_STFT_ROUNDTRIP", "TRUE_TARGET"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", default=os.path.join(C.S1A_DIR, "MIXTURE_RECIPES.json"))
    ap.add_argument("--opt-iters", type=int, default=150)
    args = ap.parse_args()
    recipes = C.load_json(args.recipes)
    st = ClapSepSTFT()
    out_dir = os.path.join(C.S1A_DIR, "work", "representation")
    os.makedirs(out_dir, exist_ok=True)
    runs = []
    for case in recipes["cases"]:
        cid = case["id"]
        y, _ = C.read_wav(os.path.join(C.S1A_DIR, case["files"]["y"]))
        s, _ = C.read_wav(os.path.join(C.S1A_DIR, case["files"]["s"]))
        y, s = C.to_mono(y), C.to_mono(s)
        t0 = time.time()
        res = run_oracles(st, y.astype(np.float64), s.astype(np.float64),
                          opt_max_iter=args.opt_iters)
        cdir = os.path.join(out_dir, cid)
        os.makedirs(cdir, exist_ok=True)
        paths = {}
        for v in VARIANTS:
            p = os.path.join(cdir, f"{v.lower()}.wav")
            C.write_wav(p, res["outputs"][v], case["sr"])
            paths[v] = os.path.relpath(p, C.S1A_DIR).replace("\\", "/")
        np.save(os.path.join(cdir, "mask_optimized.npy"),
                res["masks"]["optimized"].astype(np.float64))
        np.save(os.path.join(cdir, "mask_bounded.npy"),
                res["masks"]["bounded"].astype(np.float64))
        runs.append({"id": cid, "outputs": paths,
                     "optimizer": res["optimizer"],
                     "runtime_s": round(time.time() - t0, 2),
                     "loss_reduction_db": C.db((res["optimizer"]["loss_final"]
                                               + 1e-300)
                                              / (res["optimizer"]["loss_init"] + 1e-300))})
        C.log(f"{cid}: opt loss {res['optimizer']['loss_init']:.4g} -> "
              f"{res['optimizer']['loss_final']:.4g} "
              f"({res['optimizer']['n_iter']} iters, {runs[-1]['runtime_s']}s)")
    C.save_json({"opt_iters": args.opt_iters, "runs": runs},
                os.path.join(C.LOG_DIR, "oracle_runs.json"))


if __name__ == "__main__":
    main()
