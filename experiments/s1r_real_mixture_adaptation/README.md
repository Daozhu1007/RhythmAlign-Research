# S1R — Real-Mixture Adaptation

**Stage question:** can CLAPSep be adapted using authentic real handcam mixtures
WITHOUT destroying the useful source selectivity already present in the zero-shot
model?

**FINAL VERDICT: B — REAL-MIXTURE ADAPTATION IS VIABLE BUT NO CLEAR PRODUCT GAIN.**
The adaptation repaired S1W's catastrophic failure (zero collapse, zero passthrough,
teacher selectivity audibly preserved, DEV consistency −25%, generalization safe on
14 unseen recordings), but blinded listening found S1R indistinguishable from
zero-shot: 0/6 clear group preferences (bar ≥3/6), no group worse. Interpretation:
teacher-only / consistency-dominated distillation preserves the teacher but adds no
new information. See [S1R_REPORT.md](S1R_REPORT.md),
[S1R_DECISION.json](S1R_DECISION.json),
[listening_pack/HUMAN_LISTENING_RESULTS.md](listening_pack/HUMAN_LISTENING_RESULTS.md).

S1R follows S1W's clean negative result (constructed weak-supervision task learned
successfully; catastrophic collapse on real handcams). It attacks that failure mode
directly: the training domain IS the authentic raw handcam distribution, the frozen
zero-shot CLAPSep is a conservative anchor (never an oracle), and the S1W failure
modes are made explicitly expensive (anti-collapse guard, cross-context consistency,
minimal parameter movement). This is NOT synthetic remixing, NOT a complex-mask
experiment, NOT aligned-reference conditioning, NOT another model zoo.

## Method summary

1. **Frozen teacher.** Exact zero-shot CLAPSep (S1E/S1W checkpoints, audio query Q1,
   zero negative), bitwise parity re-verified before any use.
2. **Phase 0 — teacher viability audit.** Multi-context (early/canonical/late 10-s
   views of the same central ~6 s) + ±3 dB gain views on real TRAIN/DEV windows;
   multi-metric consistency diagnostics define HIGH_CONFIDENCE_TEACHER_ANCHOR and the
   teacher gate (protocol §15). Result: **PASS (preferred)** — see
   [TEACHER_VIABILITY_REPORT.md](TEACHER_VIABILITY_REPORT.md).
3. **Real-mixture training** (if gate passes): student = zero-shot init; minimal
   trainable surface (final mask head + upper decoder layers); losses = teacher-anchor
   distillation + cross-context consistency + scale equivariance + anti-collapse
   hinge + L2-SP weight anchor (protocol §§20-27). Micro-pilot gate before the
   primary run (≤1,500 updates, DEV-only selection, anti-collapse gate before any
   test).
4. **Evaluation:** frozen DEV identities only for selection; TEST_GENERALIZATION
   (14 recordings) and SEALED_TEST_PRIMARY (6 S1E challenge groups: RAW vs ZERO-SHOT
   vs S1R) only after full freeze; fixed-gain 18-item blind listening pack if
   machine-side results are valid.

## Public deliverables (this directory)

| file | content |
|---|---|
| `TEACHER_VIABILITY_REPORT.md` | Phase 0 audit + gate decision |
| `TEACHER_ANCHOR_MANIFEST.public.json` | anonymized anchor manifest (stable IDs, no local paths) |
| `TRAINING_PLAN.md` / `TRAINING_CONFIG.json` | declared-a-priori training design |
| `TRAINING_LOG.public.md` | training/eval narrative (numbers; details in logs/) |
| `CHECKPOINT_MANIFEST.public.json` | selected checkpoint identity (hashes; no weights) |
| `DEV_RESULTS.md` | real-DEV diagnostics per checkpoint axis |
| `GENERALIZATION_RESULTS.md` | 14 held-out recordings, descriptive proxies |
| `REAL_TEST_RESULTS.md` | sealed primary groups machine-side |
| `FAILURE_CASES.md` | honest negatives of this stage |
| `S1R_PRELISTENING_REPORT.md` | answers protocol §52 questions before listening |
| `S1R_REPORT.md` / `S1R_DECISION.json` | final stage report + verdict B record |
| `S1R_STATUS.json` | machine-readable stage status |
| `listening_pack/` | 18-item blind pack instructions, gain audit, unblinded listening results |

## Privacy / scope guards

Raw audio, checkpoints, private manifests with local paths, blind keys and listening
WAVs stay local under ignored `work/`, `checkpoints/` and `logs/` (repo .gitignore).
The product repository D:\Code\RhythmAlign is read-only for this stage. Source media
are never mutated. Recording identities are published as stable anonymized IDs only.

## Prior-stage records (read-only inputs)

- `../s1w_existing_corpus_adaptation/` — frozen splits, corpus audit, raw32k
  extractions, negative S1W result (verdict C)
- `../s1e_existing_corpus_feasibility/` — sealed challenge groups, stored zero-shot
  outputs (parity reference)
- `../r3_audio_query/` — vendored CLAPSep (read-only), query Q1, checkpoints
