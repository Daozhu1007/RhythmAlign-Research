# S1W — TRAINING_LOG (public)

**Run:** primary seed 20260912 · AdamW lr 0.0001 (decoder only) · effective
batch 8 · 2000 optimizer updates · wall clock
1895 s · peak VRAM 3.42 GB
(RTX 4060 Laptop 8 GB, bf16 autocast). Full DEV axis set: DEV_RESULTS.md.

| update | recon_l1_mix | leakage_ratio_db | weak_recon_l1 | friction_recon_l1 | n_only_out_rms_db | composite |
|---|---|---|---|---|---|---|
| 250 | 0.0186 | -10.08 | 0.0122 | 0.0173 | -45.2 | 5.407 |
| 500 | 0.0183 | -10.15 | 0.0119 | 0.0169 | -47.2 | 5.464 |
| 750 | 0.0199 | -9.71 | 0.0140 | 0.0172 | -38.3 | 5.089 |
| 1000 | 0.0182 | -10.28 | 0.0121 | 0.0168 | -45.4 | 5.533 |
| 1250 | 0.0186 | -10.04 | 0.0123 | 0.0188 | -48.7 | 5.143 |
| 1500 | 0.0182 | -10.26 | 0.0121 | 0.0166 | -46.5 | 5.531 |
| 1750 | 0.0184 | -10.27 | 0.0123 | 0.0164 | -45.3 | 5.482 |
| 2000 | 0.0183 | -10.17 | 0.0124 | 0.0165 | -47.4 | 5.481 |

**Selected:** update 1000 (composite 5.533;
weak-preservation guardrail OK at every eval point).
Machine-readable numbers: TRAINING_LOG.public.json.
