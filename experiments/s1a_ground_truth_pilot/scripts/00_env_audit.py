"""S1a 00 — environment audit + provenance of the audited CLAPSep representation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main() -> None:
    out = {
        "stage": "s1a_ground_truth_pilot",
        "created": "2026-09-10",
        "env": C.env_audit(),
        "clapsep_stft_config": C.CLAPSEP_STFT,
        "clapsep_mask_config": C.CLAPSEP_MASK,
        "clapsep_inference_protocol": C.CLAPSEP_INFERENCE_PROTOCOL,
        "clapsep_sources": [],
        "truth_tiers": C.TRUTH_TIERS,
    }
    for p in C.CLAPSEP_SOURCES:
        out["clapsep_sources"].append({
            "path": os.path.relpath(p, C.REPO_DIR).replace("\\", "/"),
            "sha256": C.sha256_file(p),
        })
    C.save_json(out, os.path.join(C.LOG_DIR, "env_audit.json"))
    C.log("wrote logs/env_audit.json")
    for s in out["clapsep_sources"]:
        C.log(f"  {s['path']} {s['sha256'][:12]}...")


if __name__ == "__main__":
    main()
