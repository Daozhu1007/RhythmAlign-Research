# S2A — REAL_TEST_RESULTS (held-out machine-side; protocol sections 28-29)

**Date:** 2026-09-15 · Frozen checkpoint `full_upd01000.ckpt`
(sha256 in `CHECKPOINT_MANIFEST.public.json`); protocol frozen before the run
(`logs/09_real_test.json` → `freeze`). Five song-disjoint held-out recordings
(DRD, 共感觉, 延误列车, 白39, 红宙天 — anonymized as recording_0005/0022/0033/0049/0054),
4 windows each. Methods: RAW, ZERO-SHOT CLAPSep, S1R (frozen base), S2A
CORRECT / ZERO / WRONG reference. Descriptive proxies only — no stems exist;
causal ablations are kept separate from quality claims
(also see REFERENCE_ABLATION_RESULTS.md).

## Machine-side summary (20 windows)

| metric | value |
|---|---|
| S2A(correct) RMS vs S1R | within ±0.02 dB on every window (median −0.01) |
| S1R RMS vs RAW | −4.9…−6.1 dB (unchanged from the S1R record) |
| transient click-band retention vs RAW: S1R / S2A | −0.386 / −0.386 dB median |
| ref-active TF attenuation: S2A correct vs zero vs wrong | −5.3 vs −5.3 vs −5.3 dB median (identical) |
| collapse flags | none in 60 S2A method-windows |
| raw-passthrough tendency | none (S2A sits ~5–6 dB under RAW, same as S1R) |

## What the machine side says — stated plainly

**S2A with the correct aligned reference behaves exactly like S1R without a
reference.** The reference-conditioned model neither suppresses the recording's
real music more than S1R (ref-active attenuation identical across all three
reference modes), nor damages anything (transients, levels, stability all
identical). The section-30 promotion gate fails at its third condition —
"correct reference demonstrably matters" — so the stage ends WITHOUT human
listening, per protocol sections 26 and 44.

Honesty notes: descriptive proxies only (no ground-truth stems); the controlled
injection numbers are causal but synthetic-nuisance evidence; the real-mixture
reference-attenuation metric is descriptive. The product repository was never
modified; no copyrighted audio left the local corpus.
