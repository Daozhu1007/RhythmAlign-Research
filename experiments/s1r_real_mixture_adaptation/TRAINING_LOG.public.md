# S1R — TRAINING_LOG (public narrative)

**Stage:** real-mixture adaptation · **Date:** 2026-09-14 · **Seed:** 20260914
Full numeric logs: `logs/` (machine-readable JSON; local). Config:
[TRAINING_CONFIG.json](TRAINING_CONFIG.json).

## Environment reconstruction (disclosed)

The S1E/S1W virtualenv had been deleted after S1W, so S1R recreated it: Python 3.12,
torch 2.14.0+cu126 (same major version as S1E/S1W), torchaudio 2.11.0+cu126 (only the
Resample transform is used), torchvision 0.29.0+cu126 (laion_clap dependency). One
detour is recorded: installing torchvision 0.22 first silently downgraded torch to
2.7.0 and broke the torchaudio C extension; the matched set was restored. Zero-shot
parity was then reproduced BITWISE (max |Δ| = 0.0 vs the S1E stored output on
c6_dense_golden), so the reconstructed environment is faithful for model behavior.

## Trainable scope

16,082,566 trainable parameters = **5.03% of the model** (vs S1W: the entire decoder,
43.5 M): `decoder_model.mask_net` (final mask head), `decoder_model.layers.3` (finest
decoder stage), `decoder_model.skip.3` (final skip transform incl. FiLM query
conditioning), `decoder_model.inverse_patch_embed`. Frozen: CLAP encoders,
audio_branch + LoRA, film, decoder layers 0-2, skips 0-2, spec_norm BatchNorm state.

## Micro-pilot (protocol section 29) — PASS

200 updates, DEV eval every 50, 622 s wall, peak 3.42 GB VRAM.

- loss finite and decreasing (0.0087 → 0.0081 per example), gradients finite and
  nonzero (norm ≈ 0.46-0.55), checkpoints saved and reloaded.
- NO output collapse (0 anchors below −6 dB vs teacher; DEV median ratio −0.04..−0.24 dB).
- NO raw-passthrough drift (passthrough margin ≤ +0.0105; guard 0.05).
- teacher-anchor divergence stayed tiny (median L1 ≈ 0.006 vs 0.089 full-passthrough scale).
- real cross-context consistency (DEV anchors, combined L1+MR-STFT):
  **0.0609 → 0.0493 (−19%)**; TRAIN-anchor consistency 0.0100 → 0.0097.
- alignment across context views verified by the metric itself (shared central crop).

## Primary run (protocol section 30) — COMPLETED, no guards tripped

1,500 updates max, DEV eval every 250, AdamW lr 2e-5, grad-accum 8, bf16 autocast.
3,287 s wall (0.91 GPU-h of the 24 h budget), peak 3.42 GB. One run; no second seed
needed (the result is a stable conservative-adaptation outcome, not a borderline race).

| update | DEV consistency (L1+MR-STFT) | DEV median student/teacher dB | worst anchor dB | anchor div L1 | passthrough margin | click retention dB | all §32 checks |
|---|---|---|---|---|---|---|---|
| zero-shot | 0.0609 | +0.00 | −0.09 | 0.0008 | +0.0004 | −0.00 | (reference) |
| 250 | 0.0493 | +0.07 | −0.23 | 0.0069 | +0.0184 | +0.02 | ✓ |
| 500 | 0.0458 | −0.07 | −0.94 | 0.0081 | +0.0103 | +0.04 | ✓ |
| 750 | 0.0485 | −0.22 | −1.00 | 0.0059 | +0.0027 | +0.01 | ✓ |
| 1000 | 0.0494 | +0.11 | −0.25 | 0.0081 | +0.0238 | −0.00 | ✓ |
| 1250 | 0.0483 | −0.07 | −0.43 | 0.0074 | +0.0225 | −0.01 | ✓ |
| 1500 | 0.0455 | +0.03 | −0.48 | 0.0068 | +0.0136 | +0.10 | ✓ |

Improvement saturated early (most of the consistency gain is reached by update 250-500);
the run was continued to the declared budget as planned and no checkpoint showed any
collapse or passthrough tendency at any point.

## Selection and gates (DEV only)

- All six primary-run checkpoints pass ALL section-32 validity checks.
- **Selected: `full_upd01500.ckpt`** (lowest DEV consistency, 0.0455) → frozen as
  `checkpoints/s1r_selected.ckpt`, sha256 `e1fade5e…` (full hash in
  [CHECKPOINT_MANIFEST.public.json](CHECKPOINT_MANIFEST.public.json)).
- Section-33 anti-collapse gate: **PASS** (collapse rate 0.0; median ratio +0.03 dB;
  worst anchor −0.48 dB; passthrough margin +0.014). No pilot checkpoint was a
  selection candidate (by design).

Training then STOPPED touching the model; generalization and sealed evaluations used
the frozen artifact with frozen inference settings.
