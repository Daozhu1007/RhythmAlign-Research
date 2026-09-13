# S1R — TRAINING_PLAN

**Date:** 2026-09-14 · status: executed after the Phase 0 teacher gate PASSED
(preferred thresholds; see [TEACHER_VIABILITY_REPORT.md](TEACHER_VIABILITY_REPORT.md))
and after bitwise zero-shot parity reproduction.

## Goal

Adapt CLAPSep to the authentic handcam domain using REAL raw mixtures ONLY, WITHOUT
destroying the useful source selectivity zero-shot already has. The first success
criterion is "adapt without catastrophic forgetting / collapse / passthrough";
beating zero-shot is secondary (protocol section 3).

## Training domain (what changed vs S1W)

- Training inputs are AUTHENTIC REAL RAW HANDCAM MIXTURES (the frozen TRAIN
  recording identities), used as they are. No synthetic target+nuisance remixing,
  no S1W mixtures, no proxy targets, no foley (protocol section 8).
- Each training example is one high-confidence real region presented under two
  different 10-s inference contexts plus one known-gain view of the same content;
  the aligned central ~6 s is the comparison region (protocol section 20).
- Teacher targets are FROZEN zero-shot outputs precomputed once
  (`work/private/teacher_precompute.private.json`); the teacher never runs during
  training and its weights are never updated.

## Student and scope (protocol sections 17-18)

- Student initializes from the EXACT frozen zero-shot CLAPSep weights
  (sha256-verified checkpoints; NOT from the S1W adapted checkpoint, which is a
  recorded negative artifact).
- Trainable (option B — final separation/mask head + upper decoder layers):
  `decoder_model.mask_net` (final mask head),
  `decoder_model.layers.3` (finest decoder stage),
  `decoder_model.skip.3` (final skip transform incl. its FiLM query conditioning),
  `decoder_model.inverse_patch_embed` (patch→wav reconstruction head).
- Frozen: CLAP text/query encoder, audio_branch (+LoRA), `film`, `layers.0..2`,
  `skip.0..2`, decoder `spec_norm` BatchNorm running state (eval mode).
- Exact trainable parameter counts are recorded in TRAINING_CONFIG.json and the
  training log at launch.

## Objective (protocol sections 21-27; declared a priori)

On the shared aligned central crops:

```
L = 1.00 * teacher_anchor            (waveform L1 + 0.5 * MR-STFT log-mag, per view,
                                      averaged over canonical/context/gain views)
  + 0.75 * real_context_consistency  (student view A vs view B of the SAME region)
  + 0.25 * scale_equivariance        (F(g*x) vs g*F(x), g in {-3 dB, +3 dB})
  + 1.00 * anti_collapse             (6 dB hinge under teacher + transient-band floor
                                      4.5 dB on top-energy teacher frames; guard only)
  + 0.10 * pretrained_weight_anchor  (L2-SP toward theta_zero_shot)
```

Consistency terms are waveform L1 + 0.5 * MR-STFT log-mag. No SI-SDR, no visual
onsets, no chart timing, no leakage proxies, no sealed-test feedback (section 28).

## Run plan (protocol sections 29-30)

1. **Micro-pilot**: ~200 optimizer updates, DEV eval every 50. Must show: finite
   loss, finite nonzero gradients, checkpoint save/reload, no collapse, no raw
   passthrough, preserved teacher-anchor behavior, measurably decreasing real
   cross-context inconsistency on TRAIN/DEV. Any collapse/passthrough/constant-output
   shortcut STOPS the stage segment immediately.
2. **Primary run**: max 1,500 optimizer updates, DEV eval every 250, early stop if
   DEV saturates. AdamW lr 2e-5 (lower than S1W's 1e-4 — smaller adaptation surface,
   forgetting is the primary risk), weight decay 0.01, grad clip 5.0, microbatch 1,
   grad accumulation 8, bf16 autocast, seed 20260914.
3. Max 2 full runs (second seed only if the first is scientifically promising);
   aggregate GPU budget ≤ 24 h (S1W-class hardware: RTX 4060 8 GB).
4. **Checkpoint selection**: DEV-only, using the section 32 validity checks (no
   collapse; no raw passthrough; anchor selectivity preserved; consistency improves;
   no fidelity-proxy regression) — never lowest-consistency-at-any-cost. Sealed
   primary test untouched until everything is frozen (section 34).
5. Before any test evaluation: machine-side DEV anti-collapse gate (section 33).
