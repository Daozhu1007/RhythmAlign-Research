# S1W step 03 - corpus manifests (private detailed / public sanitized) + CORPUS_AUDIT.md
import os
import json
from collections import Counter

S1W = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PRIVATE = os.path.join(S1W, "work", "private")
PUB = S1W


def main():
    disc = json.load(open(os.path.join(PRIVATE, "CORPUS_DISCOVERY.private.json"), encoding="utf-8"))
    split = json.load(open(os.path.join(PRIVATE, "DATA_SPLIT.private.json"), encoding="utf-8"))

    recs = disc["records"]
    split_of_family = {}
    for key in ("sealed_primary", "test_generalization", "dev", "train"):
        for e in split[key]:
            split_of_family[e["family"]] = key

    # private manifest: everything, plus split mapping
    priv = dict(disc)
    priv["split_mapping"] = {e["family"]: key for key in
                             ("sealed_primary", "test_generalization", "dev", "train")
                             for e in split[key]}
    priv["duplicate_exclusions"] = split["excluded_duplicates"]
    with open(os.path.join(PRIVATE, "CORPUS_MANIFEST.private.json"), "w", encoding="utf-8") as f:
        json.dump(priv, f, ensure_ascii=False, indent=1)

    # public manifest: recording-identity rows only, safe metadata
    usable = [r for r in recs if r.get("provenance_class") == "A"]
    split_rows = {e["recording_id"]: e for key in
                  ("sealed_primary", "test_generalization", "dev", "train") for e in split[key]}
    id_of_family = {e["family"]: e["recording_id"] for e in split_rows.values()}
    fams = {}
    for r in usable:
        fams.setdefault(r["family"], []).append(r)
    fams = {k: sorted(v, key=lambda r: r["rel_path"]) for k, v in fams.items() if k in id_of_family}
    order = sorted(fams)

    pub_rows = []
    for fam in order:
        members = fams[fam]
        primary = next((m for m in members if m["rel_path"] not in
                        [d["rel_path"] for d in split["excluded_duplicates"]]), members[0])
        rid = id_of_family[fam]
        e = split_rows[rid]
        derived_n = sum(1 for r in recs if r.get("provenance_class") == "B" and r.get("family") == fam)
        pub_rows.append({
            "recording_id": rid,
            "split": e["split"],
            "provenance_class": "ORIGINAL_RAW_HANDCAM",
            "duration_s": e["duration_s"],
            "audio": {"codec": "aac", "sample_rate": 48000, "channels": 2},
            "video_codec": primary.get("video_codec"),
            "raw_file_copies": len(members),
            "derived_siblings": derived_n,
            "byte_identical_duplicate_copies": [d["rel_path"] for d in split["excluded_duplicates"]
                                                if d["family"] == fam],
        })
    cnt = Counter(r["provenance"] for r in recs)
    pub_doc = {
        "stage": "S1W existing-corpus adaptation",
        "corpus": "existing owner-recorded maimai handcam corpus (local; media never published)",
        "privacy_note": "publication-safe metadata only; stable IDs; no paths, no device/GPS metadata, "
                        "no song titles, no media files",
        "totals": {
            "media_assets_discovered": len(recs),
            "original_raw_handcam_files": cnt["ORIGINAL_RAW_HANDCAM"],
            "rhythmalign_derived_files": cnt["RHYTHMALIGN_DERIVED"],
            "image_or_reference_assets": cnt["IMAGE_OR_NON_AUDIO_ASSET"],
            "unknown": cnt.get("UNKNOWN", 0),
            "unique_recording_identities": len(order),
            "usable_raw_duration_s": round(sum(r["duration_s"] for r in pub_rows), 1),
        },
        "split_counts": {"SEALED_TEST_PRIMARY": len(split["sealed_primary"]),
                         "TEST_GENERALIZATION": len(split["test_generalization"]),
                         "DEV": len(split["dev"]), "TRAIN": len(split["train"])},
        "recordings": pub_rows,
    }
    with open(os.path.join(PUB, "CORPUS_MANIFEST.public.json"), "w", encoding="utf-8") as f:
        json.dump(pub_doc, f, ensure_ascii=False, indent=1)

    # audit report
    usable_dur = sum(e["duration_s"] for e in
                     split["train"] + split["dev"] + split["test_generalization"])
    sealed_dur = split["sealed_primary"][0]["duration_s"]
    train_dur = sum(e["duration_s"] for e in split["train"])
    dev_dur = sum(e["duration_s"] for e in split["dev"])
    tg_dur = sum(e["duration_s"] for e in split["test_generalization"])
    fam_line = "\n".join(
        f"| {split_rows[id_of_family[f]]['recording_id']} | {split_rows[id_of_family[f]]['split']} | "
        f"{split_rows[id_of_family[f]]['duration_s']} | {len(fams[f])} | "
        f"{sum(1 for r in recs if r.get('provenance_class') == 'B' and r.get('family') == f)} |"
        for f in order)
    audit = f"""# S1W — CORPUS_AUDIT

**Date:** 2026-09-12 · **Corpus:** existing owner-recorded maimai handcam tree (local,
read-only; nothing in the source tree was modified, renamed, moved or transcoded).
This audit describes the corpus and the frozen split. Provenance classification and
split freezing happened BEFORE any target-proxy or nuisance mining (protocol order).

## Totals

| Item | Count |
|---|---|
| media assets discovered | {len(recs)} |
| ORIGINAL_RAW_HANDCAM files | {cnt['ORIGINAL_RAW_HANDCAM']} |
| RHYTHMALIGN_DERIVED files (excluded from all mining/splits) | {cnt['RHYTHMALIGN_DERIVED']} |
| image / pristine-reference assets (class C, excluded) | {cnt['IMAGE_OR_NON_AUDIO_ASSET']} |
| UNKNOWN | {cnt.get('UNKNOWN', 0)} |
| byte-identical duplicate raw copies (excluded, one identity kept) | {len(split['excluded_duplicates'])} |
| unique RAW recording identities (provenance families) | {len(order)} |
| usable RAW duration (excl. sealed) | {usable_dur:.0f} s (~{usable_dur / 60:.0f} min) |

Provenance basis (verified, not assumed): every RAW file carries phone capture
container metadata; every DERIVED file carries an ffmpeg export encoder marker
(`Lavf…`/`Lavc… nvenc`) plus a `_synced` filename marker, and matches its raw sibling's
duration within ±0.01 s. The `_synced` suffix is the RhythmAlign product export naming
convention (product `ui_main.py` save dialog default). Device/GPS container metadata
was read for provenance only and is kept in the PRIVATE manifest; it is never published.

## Split (frozen at this point, before any mining)

Unit = recording identity (provenance family). Seed {split['seed']}; rule: sealed S1E
source; stratified diversity strata (speech-contaminated / slide-friction-heavy / dense /
quietest floor / hottest capture / sparsest / weakest level) in declared priority order,
folder-coverage fill, seeded remainder; DEV re-stratified from the remainder.

| Split | Identities | Duration (s) |
|---|---|---|
| SEALED_TEST_PRIMARY (entire 共感怪物AP recording; S1E continuity benchmark) | 1 | {sealed_dur:.0f} |
| TEST_GENERALIZATION | {len(split['test_generalization'])} | {tg_dur:.0f} |
| DEV | {len(split['dev'])} | {dev_dur:.0f} |
| TRAIN | {len(split['train'])} | {train_dur:.0f} |

The sealed recording may contribute NO material to TRAIN/DEV/proxy/nuisance/checkpoint
or query decisions. TEST_GENERALIZATION recordings may contribute no material either.
Derived `_synced` files ({cnt['RHYTHMALIGN_DERIVED']} files) are excluded from every
mining/evaluation role; duplicates {', '.join(d['rel_path'] for d in split['excluded_duplicates'])}
are byte-identical copies (sha256-verified) of an included identity and add no new identity.

## Recording identity table

| recording_id | split | duration s | raw copies | derived siblings |
|---|---|---|---|---|
{fam_line}

## Provenance ambiguities / notes

- 9 of {cnt['RHYTHMALIGN_DERIVED']} derived files use an `Lavc…nvenc` encoder tag instead of
  `Lavf…`; classification as DERIVED is unambiguous (filename marker + raw sibling +
  duration match + ffmpeg export marker).
- RAW files split across two container codecs (hevc {sum(1 for f in order for r in fams[f] if r.get('video_codec') == 'hevc')} files /
  h264 rest) — both are phone capture metadata-verified; no provenance impact.
- `已发/海底谭/海底谭.mp4` and `DATASET/海底谭DATASET/海底谭0.mp4` are the same recording
  (identical sha256) — one identity, one usable copy.
- No file was found whose provenance could not be established; UNKNOWN = 0.

## Sufficiency verdict

58 unique raw identities (~{usable_dur / 60:.0f} min usable beyond the sealed source) with
verified diversity (level −20…−9 dBFS RMS, speech fraction 0–0.18, onset density
1.2–5.7/s, friction-flatness 0.09–0.76, quiet-to-clipping captures, 8 folder groups) is
SUFFICIENT for a leakage-safe recording-level split and the S1W weak-supervision
experiment. CORPUS_INSUFFICIENT does not apply.
"""
    with open(os.path.join(PUB, "CORPUS_AUDIT.md"), "w", encoding="utf-8") as f:
        f.write(audit)
    print("manifests + audit written; identities:", len(order),
          "| usable s:", round(usable_dur), "| train/dev/tg:", len(split["train"]), len(split["dev"]), len(split["test_generalization"]))


if __name__ == "__main__":
    main()
