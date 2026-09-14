# S2A — REFERENCE_ABLATION_RESULTS (protocol sections 16/21/26)

**Date:** 2026-09-15 · Evaluation checkpoint: `full_upd01000.ckpt` (documentation
of a null effect; see DEV_RESULTS for the selection failure record). The causal
ablation triad is mandatory: a model that behaves identically for CORRECT and
WRONG references is not meaningfully reference-conditioned. **That is exactly
what is observed.**

## DEV (15 windows, 3 recordings)

| quantity (median) | value | gate |
|---|---|---|
| suppression advantage vs S1R (correct ref, α=0.3 injection) | +0.0074 dB best checkpoint-wise; ≤ 0.01 dB anywhere | > 0 dB (section 25) — FAILed by most rungs |
| CORRECT vs ZERO_REFERENCE gap | 0.0012 dB best | ≥ 1.0 dB (section 25) — **FAIL** |
| CORRECT vs WRONG_REFERENCE gap | 0.0002 dB best | ≥ 1.5 dB preferred (section 26) — **FAIL** |
| wrong-reference safety (clean mixture, median/min ratio) | −0.27 dB / −0.63 dB | ≥ −3 / ≥ −6 dB — PASS |

## Held-out TEST (20 windows, 5 song-disjoint recordings)

| quantity (median) | value |
|---|---|
| suppression advantage vs S1R (controlled injection) | **+0.013 dB** (max +0.027; 0/20 windows ≥ 1.5 dB) |
| CORRECT vs ZERO_REFERENCE gap | +0.004 dB |
| CORRECT vs WRONG_REFERENCE gap | **+0.000 dB** |
| transient (click) retention: S2A vs S1R | −0.386 vs −0.386 dB (identical) |
| output RMS: S2A vs S1R | within ±0.02 dB on every window |
| collapse flags (correct or wrong ref) | none |
| ref-active TF attenuation (S2A correct) | −5.3 dB median — identical to zero/wrong ref rows (the attenuation comes from the frozen S1R mask, not the reference) |

## Reading

1. The model's behavior is statistically indistinguishable across CORRECT, ZERO,
   and WRONG reference inputs — on DEV and on held-out songs, on clean mixtures
   and under controlled injection. The tiny positive advantages (≤ 0.03 dB) are
   uniform across reference modes (i.e., "a reference exists" at most, never
   "the right reference content").
2. Wrong-reference safety was never at risk: with the reference gate effectively
   closed, wrong references do nothing at all — safe by nullity, not by learned
   rejection.
3. This is the section-26 stop condition in its purest form:
   **REFERENCE_ADAPTER_IGNORED** — human listening is not warranted, and no
   claim of reference-driven suppression may be made for any checkpoint of this
   stage.
