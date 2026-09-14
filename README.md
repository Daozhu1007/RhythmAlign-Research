# RhythmAlign Research

Research workspace for **RhythmAlign** audio extraction, denoising, evaluation, and
target-sound-separation experiments.

## Relationship to the product

- **RhythmAlign** (`github.com/Daozhu1007/RhythmAlign`) is the **production
  application**.
- **RhythmAlign-Research** (this repository) is the **exploratory research workspace**
  that studies how to improve the product's audio component. It was migrated out of the
  product repository on 2026-09-12 so each can evolve independently. Nothing here is
  shipped with the product, and nothing here should be read as a statement about
  production readiness.

## Research goal

Extract **authentic player–machine physical interaction sounds** (key taps, slider
friction, button impacts) from noisy rhythm-game handcam recordings, while suppressing
unrelated arcade audio: cabinet music playback, NPC/announcer speech, and environmental
interference. The hard core of the problem is that target and interference overlap in
time and spectrum, the target is unscripted and has no reference stem, and only
single-channel phone audio is available.

## What kind of repository this is

This is a **research record**: code, methodology, metrics, decisions, provenance — and
deliberately preserved **negative findings**. The program is exploratory; many early
experiments are negative or have been superseded by later stages, and those failed
attempts are kept as evidence, not hidden.

**Not included in this repository** (kept local-only):

- raw handcam recordings, copyrighted song audio, listening-pack audio, or any
  recordings of people's voices,
- model checkpoints / pretrained weights / training artifacts,
- bulk numeric outputs, video frame dumps, caches, and virtualenvs.

`migration/MIGRATION_MANIFEST.public.json` records the SHA-256 of every local artifact
so integrity can be verified without publishing the binaries. Large or non-public
artifacts are referenced by manifests, not committed.

## Stage chronology (compact)

| Stage | Question | Outcome |
|---|---|---|
| **R1** golden sample | Can handcam + reference audio be aligned and differenced? | Alignment works; waveform cancellation ceiling ~5.9% coherent share — cancellation alone is dead |
| **R2** AudioSep | Can a pretrained text-queried separator extract taps? | **Negative** — output is a uniform attenuated copy of the mixture, not source-selective |
| **R3** CLAPSep / audio queries | Does audio-queried separation behave better? | Partially — source selectivity visible, but far from usable extraction |
| **R4–R5.6** contact reconstruction | Can oracle/auto contact timelines drive reconstruction? | Detection pipeline matured (dev→frozen→blind holdouts), but built on contaminated eval assumptions; **R5-FIX** repaired validity — core conclusion survived only in weakened form |
| **Independent review** (R0–R5.6 audit) | Is the evidence chain sound? | Found factual/eval defects; drove the reframing |
| **S0** research reframing | What is the right problem statement? | Major reframing to *target-sound extraction under natural overlap*; literature map, architecture options, minimal decisive experiment |
| **S1a** ground-truth pilot | Can exactly-matched ground truth be acquired, and is a bounded magnitude-mask representation sufficient? | Tooling + protocols + oracle machinery validated on synthetic fixtures (unit tests pass); real one-session capture **still pending** |
| **S1E** existing-corpus feasibility | Are current pretrained frontier systems good enough as-is? | **CURRENT PRETRAINED FRONTIER INSUFFICIENT.** AudioSep repeats R2's attenuated-copy failure; SoloAudio fails out-of-distribution (deletes/invents); specialist pipeline transparent but weak. **CLAPSep shows genuine local source selectivity** (~9 dB relative music suppression with transients preserved) and is the **current adaptation substrate** |
| **S1W** weak-supervision adaptation | Does weak supervision from existing recordings move CLAPSep toward the target without deleting it? | **VERDICT C — WEAK-SUPERVISION ADAPTATION FAILS.** Constructed task learned (DEV wins), but zero transfer to real handcams: adapted output near-silent on every real group (~28–33 dB below RAW; c8 14.3 dB) — authentic interaction deleted with the nuisance (listening-confirmed). Zero-shot CLAPSep keeps useful selectivity (esp. Taiko prompt) with muffling/underwater/discontinuity defects. Root cause: synthetic-proxy distribution overfit + chunk/context sensitivity; bounded mask NOT implicated. Next (recommended, not started): **real-mixture adaptation** |
| **S1R** real-mixture adaptation | Can CLAPSep be adapted on authentic real mixtures WITHOUT destroying zero-shot selectivity? | **VERDICT B — VIABLE BUT NO CLEAR PRODUCT GAIN.** Yes to the stage question: trained on real mixtures only (teacher-anchored, 5% trainable), zero collapse/passthrough at every eval, DEV context inconsistency −25%, 14-recording generalization safe (S1W was −27.7 dB), sealed groups within ±0.2 dB RMS of zero-shot with slightly better transients. Blinded headphone listening: no deletion heard, c3 selectivity audibly survives — but **S1R ≈ zero-shot to the ear (0/6 clear preferences)**. Teacher-only distillation adds no new information; S1R is frozen as the stable real-mixture baseline. Next: **S2A reference-conditioned extraction** (aligned pristine song reference = product-native new information source) |

**Current conclusion.** No tested pretrained system solves the problem out of the box.
S1W showed synthetic-proxy adaptation collapses on real material; **S1R showed
real-mixture adaptation can be made stable — and that a student distilled only from
its teacher gains no product-level advantage over the teacher** (verdict B: viable,
no clear gain). The stable S1R checkpoint is the adaptation substrate going forward.
The recommended next stage — **S2A reference-conditioned real-mixture extraction**
(aligned pristine song reference as a genuinely new, product-native information
source, added as a near-no-op adapter on the frozen S1R base) — is the next decisive
question and **has not been started**.

## Repository layout

```
experiments/
  r1_golden_sample/            … R1–R5.6: historical preliminary experiments
  r2_target_separation/           (scripts + reports + metrics; bulk audio/weights
  r3_audio_query/                  kept local; see each REPORT.md)
  r4_contact_reconstruction/
  r45_conservative_windows/
  r5_auto_contact_detection/
  r55_burst_event_splitting/
  r56_precision_cleanup/
  r5_fix_validity_repair/
  independent_review_r0_r56/   independent audit of the R0–R5.6 evidence chain
  s0_research_reframing/       S0: problem reframing, literature map, architecture options
  s1a_ground_truth_pilot/      S1a: acquisition protocol + representation feasibility
  s1e_existing_corpus_feasibility/  S1E: pretrained-system bake-off + decision record
migration/                     migration provenance + sanitized artifact manifest
PUBLIC_REPO_AUDIT.md           pre-publication audit (sizes, exclusions, privacy scan)
PUBLICATION_MANIFEST.json      what is tracked in Git and why
LICENSE_STATUS.md              why this repository has no LICENSE file yet
RESEARCH_WORKFLOW.md           default workflow for future completed stages
```

Each experiment directory contains its own `REPORT.md` (or decision JSON), scripts,
and small JSON metrics/logs. `work/`, `checkpoints/`, media files, and
`third_party/` clones are intentionally absent (see `.gitignore` and
`PUBLIC_REPO_AUDIT.md` for the full exclusion rationale, and `THIRD_PARTY.md` for
upstream links to AudioSep / CLAPSep / SoloAudio).

## Status & caveats

- Research is **exploratory**; stage verdicts carry explicit scope limits and
  provisionality notes in their decision JSONs (e.g. S1E's decision awaits the
  recorded blind listening pass).
- S1a's real pilot capture has **not** happened yet; all S1a numbers are from
  synthetic fixtures and carry no real-acoustic feasibility meaning.
- No novelty claims are made anywhere in this repository, and no result here implies
  production readiness of RhythmAlign.
