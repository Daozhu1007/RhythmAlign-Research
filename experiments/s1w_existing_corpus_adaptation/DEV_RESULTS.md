# S1W — DEV_RESULTS

**Date:** 2026-09-12 · Selection used DEV constructed mixtures ONLY (160 fixed examples,
64-example eval subset; seed 20260912). The sealed primary test was not viewed at
any point during training or selection. Axes are kept separate per protocol;
`composite` = preservation ratio + suppression gain (dB/10) is reported but never hides
the components. Guardrail: weak-stratum preservation must not regress > 10% vs zero-shot.

## DEV trajectory (constructed DEV mixtures; fixed gain)

| update | recon_l1_mix | leakage_ratio_db | target_distortion_db | hf_retention_db | recon_l1_t_only | n_only_out_rms_db | weak_recon_l1 | friction_recon_l1 | composite | weak_guard |
|---|---|---|---|---|---|---|---|---|---|---|
| zero-shot | 0.0611 | -0.77 | 3.23 | -2.38 | 0.0228 | -18.7 | 0.0400 | 0.0677 | — | — |
| 250 | 0.0186 | -10.08 | -4.86 | 5.12 | 0.0001 | -45.2 | 0.0122 | 0.0173 | 5.407 | OK |
| 500 | 0.0183 | -10.15 | -4.92 | 5.01 | 0.0002 | -47.2 | 0.0119 | 0.0169 | 5.464 | OK |
| 750 | 0.0199 | -9.71 | -4.67 | 4.79 | 0.0001 | -38.3 | 0.0140 | 0.0172 | 5.089 | OK |
| 1000 | 0.0182 | -10.28 | -5.06 | 5.42 | 0.0001 | -45.4 | 0.0121 | 0.0168 | 5.533 | OK |
| 1250 | 0.0186 | -10.04 | -4.82 | 4.96 | 0.0013 | -48.7 | 0.0123 | 0.0188 | 5.143 | OK |
| 1500 | 0.0182 | -10.26 | -5.06 | 5.32 | 0.0001 | -46.5 | 0.0121 | 0.0166 | 5.531 | OK |
| 1750 | 0.0184 | -10.27 | -5.10 | 5.38 | 0.0001 | -45.3 | 0.0123 | 0.0164 | 5.482 | OK |
| 2000 | 0.0183 | -10.17 | -4.98 | 5.14 | 0.0001 | -47.4 | 0.0124 | 0.0165 | 5.481 | OK |

**Selected checkpoint:** update 1000 (composite 5.533).

## Reading

- Suppression: leakage_ratio_db -0.77 → -10.28 dB
  (more negative = more nuisance removed on exact constructed mixtures; ~9.4 dB gain).
- Preservation: recon_l1_mix 0.0611 → 0.0182;
  weak-hit error 0.0400 → 0.0119;
  friction error 0.0677 → 0.0164.
- Identity: target-only recon_l1 collapses to ~0.00013 (F(s)≈s learned).
- Target-absent: nuisance-only output -18.7 → -48.7 dB rms.
- HF/transient: pre-emphasis retention -2.38 → 5.42 dB
  (zero-shot NEGATIVE = crisp content distorted — the S1E muffling signature; adapted is positive).

These are CONSTRUCTED-mixture metrics (exact bookkeeping allows exact metrics).
They are NOT real-recording ground-truth claims (protocol section 39).
