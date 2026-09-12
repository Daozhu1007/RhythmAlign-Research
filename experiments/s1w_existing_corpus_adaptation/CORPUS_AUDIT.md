# S1W — CORPUS_AUDIT

**Date:** 2026-09-12 · **Corpus:** existing owner-recorded maimai handcam tree (local,
read-only; nothing in the source tree was modified, renamed, moved or transcoded).
This audit describes the corpus and the frozen split. Provenance classification and
split freezing happened BEFORE any target-proxy or nuisance mining (protocol order).

## Totals

| Item | Count |
|---|---|
| media assets discovered | 162 |
| ORIGINAL_RAW_HANDCAM files | 59 |
| RHYTHMALIGN_DERIVED files (excluded from all mining/splits) | 37 |
| image / pristine-reference assets (class C, excluded) | 66 |
| UNKNOWN | 0 |
| byte-identical duplicate raw copies (excluded, one identity kept) | 1 |
| unique RAW recording identities (provenance families) | 58 |
| usable RAW duration (excl. sealed) | 9640 s (~161 min) |

Provenance basis (verified, not assumed): every RAW file carries phone capture
container metadata; every DERIVED file carries an ffmpeg export encoder marker
(`Lavf…`/`Lavc… nvenc`) plus a `_synced` filename marker, and matches its raw sibling's
duration within ±0.01 s. The `_synced` suffix is the RhythmAlign product export naming
convention (product `ui_main.py` save dialog default). Device/GPS container metadata
was read for provenance only and is kept in the PRIVATE manifest; it is never published.

## Split (frozen at this point, before any mining)

Unit = recording identity (provenance family). Seed 20260912; rule: sealed S1E
source; stratified diversity strata (speech-contaminated / slide-friction-heavy / dense /
quietest floor / hottest capture / sparsest / weakest level) in declared priority order,
folder-coverage fill, seeded remainder; DEV re-stratified from the remainder.

| Split | Identities | Duration (s) |
|---|---|---|
| SEALED_TEST_PRIMARY (entire 共感怪物AP recording; S1E continuity benchmark) | 1 | 163 |
| TEST_GENERALIZATION | 14 | 2409 |
| DEV | 7 | 1212 |
| TRAIN | 36 | 6018 |

The sealed recording may contribute NO material to TRAIN/DEV/proxy/nuisance/checkpoint
or query decisions. TEST_GENERALIZATION recordings may contribute no material either.
Derived `_synced` files (37 files) are excluded from every
mining/evaluation role; duplicates DATASET\海底谭DATASET\海底谭0.mp4
are byte-identical copies (sha256-verified) of an included identity and add no new identity.

## Recording identity table

| recording_id | split | duration s | raw copies | derived siblings |
|---|---|---|---|---|
| recording_0001 | TRAIN | 158.21 | 1 | 0 |
| recording_0002 | TRAIN | 157.67 | 1 | 1 |
| recording_0003 | TEST_GENERALIZATION | 149.53 | 1 | 0 |
| recording_0004 | TRAIN | 172.5 | 1 | 0 |
| recording_0005 | TEST_GENERALIZATION | 171.22 | 1 | 0 |
| recording_0006 | TRAIN | 180.12 | 1 | 0 |
| recording_0007 | TEST_GENERALIZATION | 181.53 | 1 | 0 |
| recording_0008 | DEV | 190.72 | 1 | 1 |
| recording_0009 | TRAIN | 150.83 | 1 | 1 |
| recording_0010 | TRAIN | 149.74 | 1 | 0 |
| recording_0011 | TRAIN | 148.48 | 1 | 0 |
| recording_0012 | TRAIN | 175.81 | 1 | 1 |
| recording_0013 | DEV | 177.66 | 1 | 1 |
| recording_0014 | DEV | 181.29 | 1 | 0 |
| recording_0015 | DEV | 168.34 | 1 | 0 |
| recording_0016 | TEST_GENERALIZATION | 188.8 | 1 | 1 |
| recording_0017 | TRAIN | 172.63 | 1 | 1 |
| recording_0018 | TEST_GENERALIZATION | 179.9 | 1 | 0 |
| recording_0019 | TRAIN | 188.61 | 1 | 0 |
| recording_0020 | TRAIN | 189.42 | 1 | 1 |
| recording_0021 | SEALED_TEST_PRIMARY | 163.07 | 1 | 0 |
| recording_0022 | TEST_GENERALIZATION | 166.19 | 1 | 0 |
| recording_0023 | TRAIN | 164.27 | 1 | 1 |
| recording_0024 | TEST_GENERALIZATION | 151.02 | 1 | 1 |
| recording_0025 | TRAIN | 155.56 | 1 | 1 |
| recording_0026 | TRAIN | 147.82 | 1 | 1 |
| recording_0027 | TRAIN | 151.3 | 1 | 1 |
| recording_0028 | TRAIN | 165.42 | 1 | 1 |
| recording_0029 | TRAIN | 188.71 | 1 | 1 |
| recording_0030 | TRAIN | 175.74 | 1 | 0 |
| recording_0031 | TRAIN | 173.55 | 1 | 1 |
| recording_0032 | TRAIN | 128.73 | 1 | 0 |
| recording_0033 | TEST_GENERALIZATION | 172.33 | 1 | 0 |
| recording_0034 | TRAIN | 163.29 | 1 | 1 |
| recording_0035 | TRAIN | 187.95 | 1 | 0 |
| recording_0036 | TRAIN | 170.5 | 1 | 1 |
| recording_0037 | TEST_GENERALIZATION | 160.21 | 1 | 0 |
| recording_0038 | TRAIN | 168.62 | 1 | 1 |
| recording_0039 | TRAIN | 170.47 | 1 | 1 |
| recording_0040 | TRAIN | 164.12 | 1 | 1 |
| recording_0041 | TRAIN | 152.06 | 1 | 1 |
| recording_0042 | DEV | 145.79 | 1 | 1 |
| recording_0043 | TEST_GENERALIZATION | 151.89 | 1 | 1 |
| recording_0044 | TRAIN | 158.74 | 1 | 1 |
| recording_0045 | TRAIN | 145.92 | 1 | 1 |
| recording_0046 | DEV | 180.69 | 1 | 0 |
| recording_0047 | TRAIN | 173.29 | 1 | 1 |
| recording_0048 | TRAIN | 170.62 | 1 | 0 |
| recording_0049 | TEST_GENERALIZATION | 146.73 | 1 | 0 |
| recording_0050 | DEV | 167.68 | 1 | 0 |
| recording_0051 | TRAIN | 191.17 | 1 | 1 |
| recording_0052 | TRAIN | 195.22 | 1 | 1 |
| recording_0053 | TEST_GENERALIZATION | 203.52 | 1 | 0 |
| recording_0054 | TEST_GENERALIZATION | 206.55 | 1 | 1 |
| recording_0055 | TRAIN | 163.88 | 1 | 1 |
| recording_0056 | TRAIN | 174.51 | 1 | 1 |
| recording_0057 | TRAIN | 172.89 | 1 | 0 |
| recording_0058 | TEST_GENERALIZATION | 179.69 | 1 | 1 |

## Provenance ambiguities / notes

- 9 of 37 derived files use an `Lavc…nvenc` encoder tag instead of
  `Lavf…`; classification as DERIVED is unambiguous (filename marker + raw sibling +
  duration match + ffmpeg export marker).
- RAW files split across two container codecs (hevc 42 files /
  h264 rest) — both are phone capture metadata-verified; no provenance impact.
- `已发/海底谭/海底谭.mp4` and `DATASET/海底谭DATASET/海底谭0.mp4` are the same recording
  (identical sha256) — one identity, one usable copy.
- No file was found whose provenance could not be established; UNKNOWN = 0.

## Sufficiency verdict

58 unique raw identities (~161 min usable beyond the sealed source) with
verified diversity (level −20…−9 dBFS RMS, speech fraction 0–0.18, onset density
1.2–5.7/s, friction-flatness 0.09–0.76, quiet-to-clipping captures, 8 folder groups) is
SUFFICIENT for a leakage-safe recording-level split and the S1W weak-supervision
experiment. CORPUS_INSUFFICIENT does not apply.
