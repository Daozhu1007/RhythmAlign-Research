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

> **POST-HOC INTEGRITY NOTE (2026-09-15, RTA-0 audit).** Scope amendments to
> this table, from `docs/reviews/RTA0_INTEGRITY_AUDIT.md`: R1/S1E "dead"
> phrasings are recording/configuration-scoped (no universal cancellation
> ceiling established); S1W's root cause is a supported explanation, not an
> isolated cause; S1R's interpretation applies to its recipe (and its training
> signal's documented 2–9 kHz transient guard was not implemented as intended
> — evaluation metrics unaffected); S2A's headline is superseded by
> "NO USEFUL REFERENCE-CONTENT EFFECT WAS DEMONSTRATED BY THE RECORDED S2A
> IMPLEMENTATION" (adapter geometry compromised one-to-one TF alignment; the
> "S1R" comparator reused the post-training base model, so S2A-vs-S1R deltas
> are not a frozen-original-S1R comparison; within-model reference ablations
> remain valid).

| Stage | Question | Outcome |
|---|---|---|
| **R1** golden sample | Can handcam + reference audio be aligned and differenced? | Alignment works; waveform cancellation ceiling ~5.9% coherent share — cancellation alone is dead *(scope: the tested cancellation variants on the selected recording; not a universal ceiling)* |
| **R2** AudioSep | Can a pretrained text-queried separator extract taps? | **Negative** — output is a uniform attenuated copy of the mixture, not source-selective |
| **R3** CLAPSep / audio queries | Does audio-queried separation behave better? | Partially — source selectivity visible, but far from usable extraction |
| **R4–R5.6** contact reconstruction | Can oracle/auto contact timelines drive reconstruction? | Detection pipeline matured (dev→frozen→blind holdouts), but built on contaminated eval assumptions; **R5-FIX** repaired validity — core conclusion survived only in weakened form |
| **Independent review** (R0–R5.6 audit) | Is the evidence chain sound? | Found factual/eval defects; drove the reframing |
| **S0** research reframing | What is the right problem statement? | Major reframing to *target-sound extraction under natural overlap*; literature map, architecture options, minimal decisive experiment |
| **S1a** ground-truth pilot | Can exactly-matched ground truth be acquired, and is a bounded magnitude-mask representation sufficient? | Tooling + protocols + oracle machinery validated on synthetic fixtures (unit tests pass); real one-session capture **still pending** |
| **S1E** existing-corpus feasibility | Are current pretrained frontier systems good enough as-is? | **PRETRAINED FRONTIER INSUFFICIENT AS EVALUATED** *(scope: the configurations successfully evaluated; gated/unavailable systems were not judged)*. AudioSep repeats R2's attenuated-copy failure; SoloAudio fails out-of-distribution (deletes/invents); specialist pipeline transparent but weak. **CLAPSep shows genuine local source selectivity** (~9 dB relative music suppression with transients preserved) and is the **current adaptation substrate** |
| **S1W** weak-supervision adaptation | Does weak supervision from existing recordings move CLAPSep toward the target without deleting it? | **VERDICT C — WEAK-SUPERVISION ADAPTATION FAILS.** Constructed task learned (DEV wins), but zero transfer to real handcams: adapted output near-silent on every real group (~28–33 dB below RAW; c8 14.3 dB) — authentic interaction deleted with the nuisance (listening-confirmed). Zero-shot CLAPSep keeps useful selectivity (esp. Taiko prompt) with muffling/underwater/discontinuity defects. Supported explanations: synthetic-proxy distribution mismatch + chunk/context sensitivity (individual contributions not isolated); bounded mask NOT implicated. Next (recommended, not started): **real-mixture adaptation** |
| **S1R** real-mixture adaptation | Can CLAPSep be adapted on authentic real mixtures WITHOUT destroying zero-shot selectivity? | **VERDICT B — VIABLE BUT NO CLEAR PRODUCT GAIN.** Yes to the stage question: trained on real mixtures only (teacher-anchored, 5% trainable), zero collapse/passthrough at every eval, DEV context inconsistency −25%, 14-recording generalization safe (S1W was −27.7 dB), sealed groups within ±0.2 dB RMS of zero-shot with slightly better transients. One owner listening pass found no clear overall preference over zero-shot (0/6 clear preferences; not an equivalence result). This strongly anchored recipe improved context consistency without demonstrating a clear product gain over the teacher. S1R is frozen as the stable real-mixture baseline |
| **S2A** reference-conditioned extraction | Does the product-native aligned pristine song reference provide enough NEW information to beat the S1R teacher? | **CURRENT STATUS: NO USEFUL REFERENCE-CONTENT EFFECT WAS DEMONSTRATED BY THE RECORDED S2A IMPLEMENTATION** (originally headlined "verdict C — REFERENCE CONDITIONING DOES NOT HELP" before the RTA-0 audit). Data gate passed at preferred minimums (24/31 conservatively aligned in-tree references; TRAIN 16/14, DEV 3/3, TEST 5/5 song-disjoint); minimal near-no-op adapter (12k params, exact S1R parity at init — a no-op check only) + mask head trained; two runs (run 1 invalidated by a sign bug, recorded). Within-model reference ablations: correct/wrong/zero behaviorally identical (median correct-vs-wrong gap 0.000 dB) → §26 stop **REFERENCE_ADAPTER_IGNORED**; machine listening gate failed → no human listening. Post-hoc audit: adapter geometry (stride-2 upsample + crop) compromises one-to-one TF alignment; the "S1R" comparator reused the post-training base model so S2A-vs-S1R deltas are not a frozen-original-S1R comparison; the experiment does NOT decide whether a correctly implemented reference-conditioned architecture can help |

**Current conclusion.** No tested pretrained system solves the problem out of the box.
The adaptation line is now answered at all three planned information levels:
S1W (synthetic weak supervision) collapsed; **S1R (real-mixture, teacher-only)
is stable but gains nothing clear over its teacher** (verdict B); **S2A (aligned
pristine reference, the product's native extra input) demonstrated no useful
reference-content effect through the recorded implementation** — an
implementation whose adapter geometry and comparator isolation were
post-hoc-audited as compromised (`docs/reviews/RTA0_INTEGRITY_AUDIT.md`). The
stable S1R checkpoint remains the best learned model. The dominant bottleneck is
now **insufficient trustworthy real-target supervision / evaluation truth**:
every learner to date was trained and judged against proxies.

**TRAINING STATUS: PAUSED PENDING TRUSTWORTHY REAL-TARGET EVIDENCE.**

No further adaptation stage is authorized before real-target evidence exists.
The next step is **acquisition and evaluation, not another learner**:
`experiments/rta1_real_target_headroom_pilot/` is prepared and
`READY_FOR_REAL_TARGET_CAPTURE` — the owner's only task is one controlled
capture session set (~15 minutes of usable material). Pre-registered success
criteria and decision rules are in that stage.

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
  s1w_existing_corpus_adaptation/   S1W: synthetic-proxy weak-supervision adaptation (failed transfer)
  s1r_real_mixture_adaptation/      S1R: real-mixture teacher-anchored adaptation (stable, no product gain)
  s2a_reference_conditioned_real_mixture/  S2A: aligned-reference adapter (no demonstrated effect)
  rta1_real_target_headroom_pilot/  RTA1: real-target capture + bounded-mask oracle headroom (PREPARED)
docs/reviews/                  post-hoc integrity audit + senior review record (2026-09-15)
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
