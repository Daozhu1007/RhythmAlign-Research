# S2A step 7: emit PUBLIC reference manifest (anonymized, publication-safe).
# Private paths, song titles, and filesystem names never leave work/private/.
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
import s2a_common as sc  # noqa: E402


def main():
    split = json.load(open(os.path.join(sc.S2A_PRIVATE, "S2A_SPLIT.private.json"),
                           encoding="utf-8"))
    rows = []
    for p in split["pairs"]:
        a = p["alignment"]
        rows.append({
            "recording_id": p["recording_id"],
            "s2a_split": p["s2a_split"],
            "song_id": p["song_id"],
            "ref_id": p["ref_id"],
            "duration_s": p["duration_s"],
            "ref_provenance_class": "in-tree owner-collected pristine song reference "
                                    "(corpus provenance class C; never a product output)",
            "offset_s": a["offset_s"],
            "z_hybrid": a["z_hybrid"],
            "stability_inliers": a["stability_inliers"],
            "drift_resid_s": a["drift_resid_s"],
            "overlap_s": a["overlap_s"],
            "spectral_agreement": a["spectral_agreement"],
            "spectral_agreement_baseline": a["spectral_agreement_baseline"],
            "split_note": p.get("split_note", ""),
        })
    pub = {
        "stage": "S2A reference-conditioned real-mixture extraction",
        "privacy_note": "publication-safe metadata only: stable anonymized IDs; "
                        "no paths, no titles, no media, no device metadata",
        "alignment": "production RhythmAlign hybrid path (read-only), production "
                     "confidence threshold + onset/segment-stability/identity checks",
        "n_pairs": len(rows),
        "recordings": rows,
        "rejected_candidates": [
            {"recording_id": r["recording_id"], "s2a_split_provenance": r["s2a_split_provenance"],
             "reason": r["reason"]}
            for r in split["rejected"]],
        "gate": {k: (v if k != "preferred_met" else v)
                 for k, v in split["gate"].items()},
        "promotions": split["promotions"],
    }
    sc.save_json(os.path.join(sc.S2A, "REFERENCE_MANIFEST.public.json"), pub)
    print("wrote REFERENCE_MANIFEST.public.json with", len(rows), "recordings")


if __name__ == "__main__":
    main()
