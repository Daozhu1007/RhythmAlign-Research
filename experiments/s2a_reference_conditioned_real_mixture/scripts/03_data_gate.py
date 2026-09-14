# S2A step 3: reference data gate + S2A split (protocol sections 11-12).
#
# - Uses ONLY alignment-accepted pairs (conservative; ambiguous = rejected).
# - The S1W/S1R frozen split (seed 20260912) is the provenance base; SEALED is
#   excluded entirely (its only candidate reference failed song-identity, and it
#   may never contribute to training/selection).
# - Documented split adaptation: two TRAIN recordings with fully-accepted
#   alignments are PROMOTED to S2A-DEV because reference availability leaves
#   only one native-DEV recording after conservative rejection. Their
#   songs remain represented in TRAIN via sibling recordings. This meets the
#   preferred DEV minimum (3) honestly instead of stopping with DEV=1; the
#   song-disjointness PREFERENCE for DEV is documented as impractical.
# - TEST keeps ONLY song identities absent from TRAIN (song-disjoint holdout).
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
import s2a_common as sc  # noqa: E402
import soundfile as sf  # noqa: E402
import subprocess  # noqa: E402

PROMOTE_TO_DEV = {
    "recording_0027": "TRAIN->DEV: alignment fully accepted (stable, identity OK); native DEV reduced to 1 by conservative rejection; song stays in TRAIN via sibling recordings",
    "recording_0052": "TRAIN->DEV: alignment fully accepted; native DEV reduced to 1 by conservative rejection; song stays in TRAIN via a sibling recording",
}


def main():
    rmap = sc.load_reference_map()
    align = {a["recording_id"]: a for a in sc.load_alignment()["pairs"]}

    accepted, rejected = [], []
    for p in rmap["pairs"]:
        a = align[p["recording_id"]]
        rec = dict(p, alignment=a)
        (accepted if a["accepted"] else rejected).append(rec)

    # split adaptation
    for rec in accepted:
        if rec["recording_id"] in PROMOTE_TO_DEV:
            rec["s2a_split"] = "DEV"
            rec["split_note"] = PROMOTE_TO_DEV[rec["recording_id"]]
        elif rec["split"] == "TRAIN":
            rec["s2a_split"] = "TRAIN"
        elif rec["split"] == "TEST_GENERALIZATION":
            rec["s2a_split"] = "TEST"
        elif rec["split"] == "DEV":
            rec["s2a_split"] = "DEV"
        else:  # sealed recording: reference conditioning may NEVER feed training/selection
            rec["s2a_split"] = "SEALED_EXCLUDED"
            rec["split_note"] = "sealed S1E/S1W/S1R recording: excluded from S2A training/DEV/TEST by protocol section 12"

    train_songs = {r["song_id"] for r in accepted if r["s2a_split"] == "TRAIN"}
    dev_songs = {r["song_id"] for r in accepted if r["s2a_split"] == "DEV"}
    test = [r for r in accepted if r["s2a_split"] == "TEST"]
    test_disjoint = [r for r in test if r["song_id"] not in train_songs]
    test_overlap = [r for r in test if r["song_id"] in train_songs]
    for r in test_disjoint:
        r["split_note"] = "song-disjoint from TRAIN"
    for r in test_overlap:
        r["s2a_split"] = "TEST_EXCLUDED_SONG_OVERLAP"
        r["split_note"] = "song also present in TRAIN; excluded from song-disjoint TEST holdout"

    counts = {}
    for r in accepted:
        counts.setdefault(r["s2a_split"], {"recordings": 0, "songs": set()})
        counts[r["s2a_split"]]["recordings"] += 1
        counts[r["s2a_split"]]["songs"].add(r["song_id"])

    gate = {
        "TRAIN": {"recordings": counts["TRAIN"]["recordings"],
                  "songs": len(counts["TRAIN"]["songs"])},
        "DEV": {"recordings": counts["DEV"]["recordings"],
                "songs": len(counts["DEV"]["songs"])},
        "TEST": {"recordings": len(test_disjoint),
                 "songs": len({r["song_id"] for r in test_disjoint})},
    }
    verdict = {
        "train_ok": gate["TRAIN"]["recordings"] >= sc.GATE_MIN_TRAIN_RECS
                    and gate["TRAIN"]["songs"] >= sc.GATE_MIN_TRAIN_SONGS,
        "dev_ok": gate["DEV"]["recordings"] >= sc.GATE_MIN_DEV_RECS,
        "test_ok": gate["TEST"]["recordings"] >= sc.GATE_MIN_TEST_RECS,
    }
    gate["verdict"] = ("PASS" if all(verdict.values()) else "FAIL REFERENCE_CORPUS_INSUFFICIENT")
    gate["preferred_met"] = {
        "TRAIN": gate["TRAIN"]["recordings"] >= sc.GATE_PREF_TRAIN_RECS
                 and gate["TRAIN"]["songs"] >= sc.GATE_PREF_TRAIN_SONGS,
        "DEV": gate["DEV"]["recordings"] >= sc.GATE_PREF_DEV_RECS,
        "TEST": gate["TEST"]["recordings"] >= sc.GATE_PREF_TEST_RECS,
    }

    # decode accepted references to 32 kHz mono for training/inference
    ffmpeg = __import__("imageio_ffmpeg").get_ffmpeg_exe()
    for ref_id in sorted({r["ref_id"] for r in accepted}):
        src = os.path.join(sc.S2A_REF22K, ref_id + ".wav")
        dst = os.path.join(sc.S2A_REF32K, ref_id + ".wav")
        if not os.path.exists(dst):
            subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                            "-i", src, "-ac", "1", "-ar", "32000", dst], check=True)
        info = sf.info(dst)
        assert info.samplerate == 32000 and info.channels == 1, info

    out = {"pairs": accepted, "rejected": [
        {"recording_id": r["recording_id"], "family": r["family"],
         "s2a_split_provenance": r["split"], "reason": "; ".join(
             k for k, v in r["alignment"]["checks"].items() if not v)}
        for r in rejected],
        "gate": gate, "promotions": PROMOTE_TO_DEV}
    sc.save_json(os.path.join(sc.S2A_PRIVATE, "S2A_SPLIT.private.json"), out)

    print(json.dumps({k: (v if k != "preferred_met" else v) for k, v in gate.items()},
                     indent=1, default=list))
    print("rejected:", [(r["recording_id"], r["family"]) for r in out["rejected"]])
    for r in accepted:
        print(f'{r["s2a_split"]:26s} {r["recording_id"]} {r["family"]:14s} '
              f'{r["song_id"]} offset={r["alignment"]["offset_s"]:+8.3f}')


if __name__ == "__main__":
    main()
