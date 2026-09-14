# S2A — REFERENCE_CORPUS_AUDIT

**Date:** 2026-09-15 · **Question:** how many ORIGINAL_RAW_HANDCAM recordings have a
trustworthy local pristine song reference? (Protocol section 8.)

## Search scope and rules

Searched conservatively INSIDE the existing handcam corpus tree only (the S1W/S1R
corpus, local and read-only). Allowed sources: owner-collected pristine song audio
files already present in the tree (the S1W corpus audit had classified them as
provenance class C and excluded them from that stage's evidence — they are exactly
S2A's raw material). Explicitly forbidden and NOT used:

- no downloads of copyrighted music
- no reference inferred from `_synced` product output (all 37 derived files excluded)
- no processed RhythmAlign output treated as pristine
- no song extracted from the raw handcam itself
- no product-repository assets (the applied-system `final_pack` references carry no
  verifiable pairing provenance and belong to separate in-flight work)

## Result

| Item | Count |
|---|---|
| pristine reference audio files found in-tree | 22 (mp3; 21 distinct song folders; one song folder holds 2 files of the same song) |
| candidate raw↔reference pairs (folder-local or exact name correspondence) | 31 |
| pairs PASSING conservative alignment (step 02) | **24** |
| alignment-rejected pairs | 7 |

Reference formats: `.mp3` only (credibly the clean song source, colocated with the
handcam of the same song). See `REFERENCE_MANIFEST.public.json` for anonymized IDs
and per-pair alignment statistics; private paths stay in `work/private/`.

## Split consequence (protocol sections 11-12)

| Split | Recordings | Distinct songs | Notes |
|---|---|---|---|
| TRAIN | 16 | 14 | preferred minimum ≥12 recs / ≥8 songs: MET |
| DEV | 3 | 3 | preferred minimum ≥3 recs: MET (see note) |
| TEST | 5 | 5 | **every TEST song is absent from TRAIN (song-disjoint)**; preferred ≥4: MET |

**GATE: PASS — all preferred minimums met** (the reduced pilot floor was not
needed). Notes, recorded honestly:

1. Native-DEV recordings with references numbered only 2, and one of them failed
   conservative alignment (see ALIGNMENT_AUDIT), leaving 1 — below even the reduced
   floor (2). Two TRAIN recordings with fully-accepted alignments were PROMOTED to
   S2A-DEV (declared in the manifest `promotions` field; their songs remain
   represented in TRAIN via sibling recordings). The "DEV songs distinct from
   TRAIN" preference is therefore met by only 1 of 3 DEV recordings — documented
   as impractical, not hidden.
2. The sealed S1E/S1W/S1R primary recording's only candidate reference FAILED the
   song-identity check. The sealed recording therefore contributes nothing to S2A
   (it may never move into training anyway); the historical continuity sentinels
   cannot be used with reference conditioning.
3. Song overlap TRAIN↔DEV exists for 2 of 3 DEV recordings (same song re-recorded
   by the same project family); TEST is fully song-disjoint, which is the direction
   that protects the generalization claim.
