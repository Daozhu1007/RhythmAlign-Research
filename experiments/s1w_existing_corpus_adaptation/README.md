# RhythmAlign S1W — Existing-Corpus Weakly Supervised CLAPSep Domain Adaptation

**Stage ID:** S1W · **Date:** 2026-09-12 · **Status:** see `S1W_STATUS.json`
(pre-listening stopping point: `AWAITING_HUMAN_LISTENING`).

S1W is the first actual model-training stage of the reframed RhythmAlign program.
It keeps the S1E conclusion (CLAPSep is the only pretrained system with real source
selectivity) and asks one question:

> Does weak supervision mined conservatively from existing real handcam recordings
> move CLAPSep toward "the physical interaction that actually happened" and away
> from "everything else in the arcade" — WITHOUT deleting authentic interaction?

Preservation matters as much as suppression. No vision, no chart timing, no synthetic
Foley, no new field capture, no production-repo changes, no representation redesign
(the bounded-mask question is explicitly deferred, see section 42 of the protocol).

## What happened here (one paragraph)

The corpus tree was discovered (162 assets) and provenance-classified with verified
container-metadata evidence (phone-capture metadata for 59 raw files; ffmpeg export
markers + `_synced` product naming for 37 derived files; 58 unique raw recording
identities after one byte-identical duplicate was collapsed). The split was frozen
BEFORE any mining: 36 TRAIN / 7 DEV / 14 TEST_GENERALIZATION recording identities +
the S1E source kept SEALED. Conservative audio-only mining yielded 156 target-proxy
events (238 s: strong/ordinary/weak impacts + friction) from 32 TRAIN recordings and
a 150-clip acoustic + 42-clip pristine nuisance bank (speech is provably absent from
TRAIN/DEV — documented). 2,560 ten-second mixtures (y = guard·(s + gain·n), exact
bookkeeping, 50/20/20/10 distribution) trained ONLY the CLAPSep decoder (43.5M
params) with a preservation-aware loss; everything else (CLAP, audio encoder, LoRA,
BN state, query policy Q1) stayed at the S1E zero-shot configuration, for which the
reproduced output is byte-identical to S1E's stored result. On constructed DEV
mixtures the adapted model improved nuisance suppression by ~9.4 dB while improving
target reconstruction 3.3×, weak-hit preservation 3.3×, friction preservation 4×,
and flipping the zero-shot HF-distortion signature positive. The frozen real test,
generalization test and the 18-item blind listening pack then complete the stage;
the decision itself waits for the owner's listening pass.

## Deliverables map

| File | Purpose |
|---|---|
| [CORPUS_AUDIT.md](CORPUS_AUDIT.md) | Corpus totals, provenance evidence, frozen split |
| [CORPUS_MANIFEST.public.json](CORPUS_MANIFEST.public.json) | Publication-safe per-recording metadata |
| [DATA_SPLIT.public.json](DATA_SPLIT.public.json) | Frozen recording-identity split (public) |
| [TARGET_PROXY_AUDIT.md](TARGET_PROXY_AUDIT.md) | Proxy mining method, yield, contamination, limitations |
| [NUISANCE_BANK_MANIFEST.public.json](NUISANCE_BANK_MANIFEST.public.json) | Nuisance classes/counts; documented absences |
| [TRAINING_DATA_MANIFEST.public.json](TRAINING_DATA_MANIFEST.public.json) | Mixture counts and isolation summary |
| [TRAINING_PLAN.md](TRAINING_PLAN.md) / [TRAINING_CONFIG.json](TRAINING_CONFIG.json) | Declared plan and configuration |
| [TRAINING_LOG.public.md](TRAINING_LOG.public.md) | Full DEV trajectory |
| [DEV_RESULTS.md](DEV_RESULTS.md) | DEV selection evidence (axes kept separate) |
| [CHECKPOINT_MANIFEST.public.json](CHECKPOINT_MANIFEST.public.json) | Which checkpoint (hashes); binaries stay local |
| [REAL_TEST_RESULTS.md](REAL_TEST_RESULTS.md) | Frozen sealed 6-group × 3-method results |
| [GENERALIZATION_RESULTS.md](GENERALIZATION_RESULTS.md) | Post-freeze generalization diagnostics |
| [FAILURE_CASES.md](FAILURE_CASES.md) | Honest negative observations |
| [listening_pack/](listening_pack/) | 18 anonymous items + instructions + gain audit + public order |
| [S1W_PRELISTENING_REPORT.md](S1W_PRELISTENING_REPORT.md) | Answers the 22 pre-listening questions |
| [S1W_STATUS.json](S1W_STATUS.json) | Machine-readable stage status |
| `scripts/` | Full pipeline (00–14) |
| `work/`, `checkpoints/` | LOCAL ONLY (audio, recipes, private manifests, sealed key, checkpoints) |

## How to reproduce

```bash
cd experiments/s1w_existing_corpus_adaptation
PY=work/.venv/Scripts/python.exe   # Python 3.10, torch/torchaudio 2.11.0+cu126
$PY scripts/00_discover_corpus.py        # discover + classify provenance (read-only)
$PY scripts/01_extract_and_diagnostics.py# working audio + split-design diagnostics
$PY scripts/02_freeze_splits.py          # FREEZE split (seed 20260912)
$PY scripts/03_write_corpus_manifests.py # manifests + CORPUS_AUDIT.md
$PY scripts/04_parity_check.py           # zero-shot parity vs stored S1E output
$PY scripts/05_mine_target_proxy.py      # TARGET_PROXY bank (TRAIN only)
$PY scripts/05b_mine_dev_proxies.py      # DEV proxies (evaluation mixtures only)
$PY scripts/06_write_proxy_audit.py      # TARGET_PROXY_AUDIT.md + quality gate
$PY scripts/07_mine_nuisance.py          # nuisance banks (TRAIN + DEV tags)
$PY scripts/08_build_mixtures.py         # constructed mixtures, exact bookkeeping
$PY scripts/10_sanity.py                 # 13 implementation sanity checks
$PY scripts/09_train.py                  # primary training run (max 2000 updates)
$PY scripts/14_write_reports.py          # DEV_RESULTS + checkpoint manifest
$PY scripts/11_real_test.py              # frozen sealed test (post-freeze only!)
$PY scripts/12_generalization.py         # post-freeze generalization diagnostics
$PY scripts/13b_rebuild_listening_pack_fixed_gain.py  # 18-item blind pack v2 (one common per-group gain; key sealed)
```

## Rules respected

- Product repository untouched (verified clean and in sync before and after).
- Source corpus read-only; no rename/move/trim/transcode; derived `_synced` files excluded
  from every role; sealed S1E source contributed nothing to any mining/decision.
- Split frozen before mining; leakage checked programmatically (`logs/10_sanity.json`).
- No synthetic Foley; no vision/chart signals; separator outputs never used as targets.
- Fixed gain everywhere: no per-output/per-method/per-item normalization; the listening
  pack applies exactly one common gain per group derived from the raw source only
  (raw peak to −1 dBFS headroom), so each item keeps its frozen inference scale and
  large loudness loss between items stays visible as decision evidence (validated in
  `listening_pack/LISTENING_PACK_GAIN_AUDIT.md`; v1 used invalid per-item −3 dBFS peak
  normalization and was regenerated — assignment/order unchanged).
- CLAPSep vendor code and checkpoints stay LOCAL (license position unchanged).
- Nothing private committed: no media, no paths, no device/GPS metadata, no key.
