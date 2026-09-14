# S2A — ARCHITECTURE: minimal aligned-reference adapter (protocol sections 13-16)

## Base (frozen)

- **Checkpoint:** `experiments/s1r_real_mixture_adaptation/checkpoints/s1r_selected.ckpt`
  (sha256 `e1fade5e…`, verified before any use; the failed S1W artifact is NOT used).
- **Parity before modification (section 13):** the S1R checkpoint reproduces the
  stored sealed-group outputs (|ΔRMS| ≤ 0.08 dB, waveform corr ≥ 0.999 — cross-
  process kernel-selection drift; in-process recomputation is bit-identical), and
  in-process, the S2A decoder replication matches the official CLAPSep path with
  max|Δ| = 0.0 (`logs/04_parity_check.json`).

## Reference conditioning (section 15's preferred shape)

ONE new component — a small TF-domain CNN adapter:

```
inputs (per 10-s chunk, 32 kHz):
  mix log-magnitude  (T=1001, F=513)   from the FROZEN model STFT
  ref log-magnitude  (T, F)            aligned pristine segment, same frozen STFT
  both standardized per-chunk (scale-free: gain views leave the adapter blind)

adapter:
  Conv2d(2→24, 7×7) → ReLU
  Conv2d(24→24, 5×5) → ReLU
  ConvTranspose2d(24→8, 4×4, /2) → ReLU
  ConvTranspose2d(8→1, 5×5)          <- ZERO-initialized (the section-15
                                        "zero-initialized residual gate")

fusion (late, at the mask logits):
  mask = sigmoid( S1R_mask_logits + delta ),  delta = adapter(mix, ref)
```

- **~12k adapter parameters.** Trainable surface = adapter + the final mask head
  (`decoder_model.mask_net`) = **15.49 M params = 4.85%** of the model; the rest of
  S1R (encoder, film, lower decoder, STFT/ISTFT, CLAP) stays frozen.
- **AT INITIALIZATION THE MODEL IS EXACTLY S1R** (section 14): the zero-init head
  makes delta ≡ 0, verified as max|Δ| = 0.0 vs the official S1R forward for
  CORRECT, ZERO and WRONG references (`logs/04_parity_check.json`). The zero-init
  head is also the residual gate: it receives gradient from update 1 (no
  zero-gradient deadlock), so "gated residual conditioning" is implemented without
  a separate scalar gate that would deadlock at exactly-zero.
- Fusion is deliberately LATE (mask logits): the reference can only re-weight the
  mask S1R would have produced — it cannot create content, and wholesale behavior
  changes require the gate to grow, which the anchor losses resist.

## Reference-input ablations are first-class (section 16)

Every evaluation path supports `CORRECT_REFERENCE` (aligned pristine segment),
`ZERO_REFERENCE` (all-zero waveform → adapter input standardizes to zeros), and
`WRONG_REFERENCE` (pristine segment of a DIFFERENT held-out TEST song). A model
whose outputs are identical under all three is not reference-conditioned and fails
the causal gate (`s2a_deveval.check_validity`).

## What the adapter must learn (sections 17-22)

Training pairs authentic raw mixtures with the counterfactual injection
`x_aug = x + α·T(m)` (α ~ U[0.12, 0.45]; T = small delay / gain / spectral tilt /
short room coloration ONLY — the authentic mixture is never removed or replaced,
and no synthetic target stem exists anywhere). The injection-invariance loss makes
reference-explained EXCESS music nuisance; the S1R anchor keeps everything else
fixed; anti-collapse and the anchor protect transients, weak taps and friction.
Section 22 is respected by design: no loss term suppresses onset coincidence with
the song; the reference enters as spectral/content side information only.

## Why this is NOT S1W again (section 19)

S1W synthesized the ENTIRE supervision world (target proxy + independently sampled
nuisance) and collapsed on real audio. S2A begins from the authentic full handcam
mixture and injects only a KNOWN, reference-correlated perturbation; the
product-domain target/background acoustics of every training example are authentic
field recordings.
