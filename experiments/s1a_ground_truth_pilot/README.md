# RhythmAlign S1a — Ground-Truth Acquisition Pilot + Representation Feasibility

**Stage ID:** S1a · **Date:** 2026-09-10 · **Status:** tooling complete and validated on
synthetic fixtures; real capture **PENDING** (see [CAPTURE_REQUIRED.md](CAPTURE_REQUIRED.md)).

S1a answers exactly two questions and nothing more:

- **Q1 (acquisition).** Can we acquire trustworthy real target interaction audio that is
  close enough to the intended handcam-microphone domain to supervise/evaluate learned
  extraction?
- **Q2 (representation).** Given exact target truth in a constructed mixture, can the
  current CLAPSep bounded magnitude-mask representation faithfully represent the desired
  target?

S1a is NOT S1b training, NOT R5.7, NOT detector tuning, NOT contact-detection research,
NOT production integration, NOT a new separator architecture, NOT a paper experiment, and
NOT a claim that the grand goal succeeded. No model was trained in S1a.

## Research principle applied

Results first. If the representation cannot recover the target even with oracle knowledge
of the true source, we stop before training. If the recording setup cannot produce
trustworthy target evidence, we stop and report the acquisition problem. A failed S1a that
prevents an invalid S1b is a success of this stage.

## Stage status and decision

- **Q1:** PENDING — no real pilot recordings exist yet. All acquisition tooling, the
  capture protocol, sync procedure and QC path are built and validated on synthetic
  fixtures. The minimum exact file set the user must record is in
  [CAPTURE_REQUIRED.md](CAPTURE_REQUIRED.md), with a one-page checklist in
  [CAPTURE_CHECKLIST.md](CAPTURE_CHECKLIST.md).
- **Q2:** answerable partially on synthetic fixtures (oracle machinery validated), fully
  only once real T1 target recordings exist and real T2 mixtures are constructed.
- **Final decision:** `PENDING — CAPTURE REQUIRED` ([S1A_DECISION.json](S1A_DECISION.json),
  §13 of [S1A_REPORT.md](S1A_REPORT.md)).

## Deliverables map

| File | Purpose |
|---|---|
| [S1A_PLAN.md](S1A_PLAN.md) | Stage plan: scope, truth tiers, panels, oracles, decision gate |
| [ACQUISITION_PROTOCOL.md](ACQUISITION_PROTOCOL.md) | One-session pilot design (channels, blocks, truth-quality flags) |
| [CAPTURE_CHECKLIST.md](CAPTURE_CHECKLIST.md) | Concise printable checklist for the capture session |
| [SYNC_PROTOCOL.md](SYNC_PROTOCOL.md) | Sync-event procedure and clock-map fitting |
| [raw/README.md](raw/README.md) | Exactly where to place future recordings |
| [CAPTURE_REQUIRED.md](CAPTURE_REQUIRED.md) | Minimum exact files the user must record |
| [PILOT_MANIFEST.json](PILOT_MANIFEST.json) | Provenance manifest (hashes, configs, inspected sources) |
| [QC_REPORT.md](QC_REPORT.md) | Ingest/QC report (synthetic fixtures until real capture) |
| [MIXTURE_RECIPES.json](MIXTURE_RECIPES.json) | Exact T2 mixture recipes (fixture panel) |
| [REPRESENTATION_DIAGNOSTICS.md](REPRESENTATION_DIAGNOSTICS.md) | CLAPSep representation audit + oracle design |
| [REPRESENTATION_RESULTS.json](REPRESENTATION_RESULTS.json) | Oracle/evaluation numbers (fixture panel) |
| [LISTENING_INSTRUCTIONS.md](LISTENING_INSTRUCTIONS.md) | Blind listening instructions (mapping hidden) |
| [listening_key.json](listening_key.json) | Anonymous-filename key — do not open before rating |
| [S1A_DECISION.json](S1A_DECISION.json) | Machine-readable stage decision |
| [S1A_REPORT.md](S1A_REPORT.md) | Full stage report (Part 18 structure) |
| [scripts/](scripts/) | Reusable ingest/QC/mixture/oracle/evaluation tooling |
| [tests/](tests/) | Unit + synthetic validation (Part 15 six cases included) |
| [fixtures/](fixtures/) | Synthetic fixture media generated for tooling validation |

## Erratum

Before any new work, the inherited Holdout-D "known rest" [6, 8] s interval was withdrawn
as ground truth: manual inspection of the original video confirmed real gameplay taps in
that interval. See [../s0_research_reframing/ERRATUM.md](../s0_research_reframing/ERRATUM.md).
Old Holdout-D is exploratory evidence only; no historical file was modified.

## How to run the tooling

```bash
# Environment A (primary, no torch): Python 3.10 with numpy/scipy/soundfile/librosa/matplotlib
cd experiments/s1a_ground_truth_pilot

# 1. Environment audit + provenance manifest (also hashes inspected CLAPSep sources)
python scripts/00_env_audit.py

# 2. Generate synthetic fixture media (ONLY while real capture is missing)
python scripts/90_generate_synthetic_fixtures.py

# 3. Ingest/QC the fixtures end-to-end (inventory -> PCM -> QC -> sync demo)
python scripts/91_run_synthetic_pipeline.py

# 4. Unit + synthetic validation (Part 15)
python -m pytest tests/ -q

# Environment B (repo venv with torch, only for the exact-convention receipt):
D:/Code/RhythmAlign/experiments/r2_target_separation/.venv/Scripts/python.exe \
    scripts/92_stft_equivalence_check.py
```

Once real recordings exist they go in `raw/<session_id>/` (see `raw/README.md`), then the
same scripts ingest them with `--session <id>`; fixture artifacts remain clearly labeled
`SYNTHETIC_FIXTURE` and are never mixed into real-data results.
