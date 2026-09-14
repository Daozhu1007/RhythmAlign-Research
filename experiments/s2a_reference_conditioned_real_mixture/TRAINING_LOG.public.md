# S2A — TRAINING_LOG (public)

**Parity before training:** sealed parity PASS (max |ΔRMS| 0.0762 dB); initial S1R parity exact (max|Δ| = 0.0) for CORRECT / ZERO / WRONG reference modes; adapter output-head gradient at step 0: 1.103e-01.

Trainable surface: 15490674 params (4.85% of model) = reference adapter + final mask head; all other S1R parameters frozen.

## Micro-pilot (section 23)

- updates: 200 · lr 2e-05 · seed 20260915 · wall 629.3 s · peak VRAM 4.955 GB · stop: COMPLETED

| upd | train zero-vs-correct (dB) | med adv vs S1R (dB) | med gap zero (dB) | med gap wrong (dB) | click ret (dB) | consistency | collapse12 | ratio vs S1R (dB) |
|---|---|---|---|---|---|---|---|---|
| 50 | 0.0 | -0.0 | 0.0 | 0.0 | -0.0042 | 0.0371 | 0.0 | -0.1657 |
| 100 | 0.0 | -0.0014 | -0.0002 | 0.0001 | -0.0323 | 0.0381 | 0.0 | -0.26 |
| 150 | 0.0 | 0.0007 | 0.0003 | 0.0001 | 0.002 | 0.0378 | 0.0 | -0.1318 |
| 200 | 0.0 | 0.0028 | 0.0008 | 0.0001 | -0.0268 | 0.0378 | 0.0 | -0.0709 |

## Primary run (section 24)

- updates: 1500 · lr 2e-05 · seed 20260915 · wall 4049.5 s · peak VRAM 4.955 GB · stop: COMPLETED

| upd | train zero-vs-correct (dB) | med adv vs S1R (dB) | med gap zero (dB) | med gap wrong (dB) | click ret (dB) | consistency | collapse12 | ratio vs S1R (dB) |
|---|---|---|---|---|---|---|---|---|
| 250 | 0.001 | -0.0003 | 0.0002 | 0.0001 | -0.0161 | 0.038 | 0.0 | -0.1866 |
| 500 | 0.001 | 0.0016 | 0.0007 | 0.0001 | -0.0293 | 0.0373 | 0.0 | -0.138 |
| 750 | -0.0 | -0.0014 | -0.0003 | 0.0001 | -0.0076 | 0.0373 | 0.0 | -0.2722 |
| 1000 | 0.001 | 0.0074 | 0.0012 | 0.0002 | 0.0012 | 0.0374 | 0.0 | -0.1147 |
| 1250 | 0.001 | 0.0007 | -0.0002 | 0.0 | -0.0189 | 0.0374 | 0.0 | -0.2701 |
| 1500 | -0.0 | -0.0004 | 0.0003 | -0.0 | -0.0371 | 0.0379 | 0.0 | -0.2663 |


## DEV selection (section 25)

- outcome: **NO CHECKPOINT PASSED ALL CHECKS**
- failing axis: `correct_ref_causally_used` (best median causal gap 0.0012 dB vs >= 1.0 dB required; section-26 preferred evidence >= 1.5 dB)
- S1R reference row (same code, zero adapter): consistency 0.0415, click -0.0019 dB, suppression advantage exactly 0 (sanity check)
- documentation checkpoint for the held-out negative result: full_upd01000.ckpt
