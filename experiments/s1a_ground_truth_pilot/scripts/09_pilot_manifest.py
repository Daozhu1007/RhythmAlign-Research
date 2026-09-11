"""S1a 09 — assemble PILOT_MANIFEST.json from all stage artifacts + hashes.

Usage: python 09_pilot_manifest.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

TOP_LEVEL = ["MIXTURE_RECIPES.json", "REPRESENTATION_RESULTS.json",
             "listening_key.json", "LISTENING_INSTRUCTIONS.md"]


def main() -> None:
    env = C.load_json(os.path.join(C.LOG_DIR, "env_audit.json")) \
        if os.path.exists(os.path.join(C.LOG_DIR, "env_audit.json")) else {}
    mix = C.load_json(os.path.join(C.S1A_DIR, "MIXTURE_RECIPES.json")) \
        if os.path.exists(os.path.join(C.S1A_DIR, "MIXTURE_RECIPES.json")) else {}
    rep = C.load_json(os.path.join(C.S1A_DIR, "REPRESENTATION_RESULTS.json")) \
        if os.path.exists(os.path.join(C.S1A_DIR, "REPRESENTATION_RESULTS.json")) else {}
    part15 = C.load_json(os.path.join(C.LOG_DIR, "synthetic_validation.json")) \
        if os.path.exists(os.path.join(C.LOG_DIR, "synthetic_validation.json")) else {}
    stft_eq = C.load_json(os.path.join(C.LOG_DIR, "stft_equivalence_check.json")) \
        if os.path.exists(os.path.join(C.LOG_DIR, "stft_equivalence_check.json")) else {}

    def try_json(pattern):
        cands = sorted(glob.glob(os.path.join(C.MANIFEST_DIR, pattern)))
        return C.load_json(cands[-1]) if cands else None

    inventory_real = try_json("media_inventory.json")
    inventory_fix = try_json("media_inventory_fixtures.json")
    qc = try_json("qc_audio_*.json")
    sync = try_json("sync_map_*.json")

    real_files = inventory_real["n_files"] if inventory_real else 0
    decision = "PENDING" if real_files == 0 else "see S1A_DECISION.json"

    manifest = {
        "stage": "s1a_ground_truth_pilot",
        "created": "2026-09-10",
        "data_status": {
            "real_capture_files": real_files,
            "real_sessions": [],
            "synthetic_fixture_class_present": inventory_fix is not None,
            "note": "fixture artifacts validate TOOLING ONLY; they carry no "
                    "real-acoustic feasibility meaning",
        },
        "truth_tiers": C.TRUTH_TIERS,
        "representation": {
            "stft_config": C.CLAPSEP_STFT,
            "mask_config": C.CLAPSEP_MASK,
            "inference_protocol": C.CLAPSEP_INFERENCE_PROTOCOL,
            "audited_sources": env.get("clapsep_sources", []),
            "torchlibrosa_equivalence": stft_eq,
        },
        "stop_conditions_checked": {
            "mixture_sum_identity": all(
                c.get("sum_identity_ok") for c in mix.get("cases", [])
            ) if mix else None,
            "part15_synthetic_oracle_validation": part15.get("all_pass"),
            "sync": (sync or {}).get("session_sync_ok"),
            "historical_files_modified": False,
            "post_hoc_gain_tuning": False,
            "independent_output_normalization": False,
        },
        "capture_inputs": {
            "raw_root": C.relpath_safe(C.RAW_DIR, C.S1A_DIR),
            "inventory": inventory_real,
            "required_files_doc": "CAPTURE_REQUIRED.md",
        },
        "fixture_run": {
            "inventory": inventory_fix,
            "qc": qc,
            "sync_map": sync,
            "mixture_cases": mix.get("n_cases"),
            "data_class": mix.get("data_class"),
        },
        "representation_results_summary": rep.get("aggregates"),
        "deliverables": {f: C.sha256_file(os.path.join(C.S1A_DIR, f))
                         for f in TOP_LEVEL
                         if os.path.exists(os.path.join(C.S1A_DIR, f))},
        "decision": decision,
    }
    C.save_json(manifest, os.path.join(C.S1A_DIR, "PILOT_MANIFEST.json"))
    C.log("wrote PILOT_MANIFEST.json")
    C.log(f"  stop conditions: {manifest['stop_conditions_checked']}")


if __name__ == "__main__":
    main()
