"""S1a 91 — end-to-end fixture pipeline (tooling validation; runs all steps in order).

Marks every produced artifact as SYNTHETIC_FIXTURE. Enforces stop conditions:
mixture sum identity, sync verdicts, and Part-15 oracle validation must all pass
or the pipeline aborts loudly.

Usage: python 91_run_synthetic_pipeline.py [--skip-opt]  (opt oracles are the slow part)
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
FIX_ROOT = os.path.join(C.FIXTURE_DIR, "session0001")


def run_py(script, *args):
    cmd = [sys.executable, os.path.join(SCRIPTS, script), *args]
    C.log("$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=SCRIPTS)
    if r.returncode != 0:
        raise SystemExit(f"STOP: {script} failed (exit {r.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt-iters", type=int, default=60)
    args = ap.parse_args()

    run_py("01_media_inventory.py", "--root", C.FIXTURE_DIR,
           "--out", os.path.join(C.MANIFEST_DIR, "media_inventory_fixtures.json"))
    run_py("02_extract_pcm.py", "--root", FIX_ROOT)  # default workdir matches 04
    run_py("03_qc_audio.py", "--pcm-json",
           os.path.join(C.MANIFEST_DIR, "pcm_extraction_session0001.json"),
           "--tag", "session0001")
    run_py("04_sync_fit.py", "--session", "session0001",
           "--root", FIX_ROOT)
    run_py("05_build_mixtures.py", "--recipe",
           os.path.join(C.FIXTURE_DIR, "fixture_mixtures_recipe.json"),
           "--out", os.path.join(C.FIXTURE_DIR, "mixtures"))
    # 05 writes the BUILT recipes (with component paths + hashes) to the
    # top-level MIXTURE_RECIPES.json; 06/07/08 consume that file.
    built = os.path.join(C.S1A_DIR, "MIXTURE_RECIPES.json")
    run_py("06_oracle_diagnostics.py", "--recipes", built,
           "--opt-iters", str(args.opt_iters))
    run_py("07_evaluate_representation.py", "--recipes", built)
    run_py("08_listening_pack.py", "--recipes", built, "--n-cases", "4")

    # Part-15 synthetic oracle validation receipt
    import oracle_validation as OV
    val = OV.run_all()
    C.save_json(val, os.path.join(C.LOG_DIR, "synthetic_validation.json"))
    C.log(f"Part-15 synthetic validation: all_pass={val['all_pass']}")
    assert val["all_pass"], "STOP: oracle synthetic validation failed"

    # mixture sum identity stop condition
    recipes = C.load_json(os.path.join(C.FIXTURE_DIR, "fixture_mixtures_recipe.json"))
    mix = C.load_json(os.path.join(C.S1A_DIR, "MIXTURE_RECIPES.json"))
    ok = all(c["sum_identity_ok"] for c in mix["cases"])
    C.log(f"mixture sum identity ok={ok}")
    assert ok, "STOP: exact mixture components do not sum numerically"

    C.log("fixture pipeline complete; artifacts are SYNTHETIC_FIXTURE class only")


if __name__ == "__main__":
    main()
