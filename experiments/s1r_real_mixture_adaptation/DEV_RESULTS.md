# S1R — DEV_RESULTS (real DEV mixtures only; selection evidence)

**Date:** 2026-09-14 · Selection used the 7 frozen DEV recording identities
(39 high-confidence teacher-anchor windows) ONLY. Axes kept separate — no single
composite decided anything; the combined consistency number below is just the sum of
its two published components. Full data: `logs/05_selection.json` (local).

## Zero-shot reference vs selected S1R student (`full_upd01500`, frozen)

| DEV-anchor metric (median over 39 anchors) | zero-shot | S1R | reading |
|---|---|---|---|
| cross-context consistency, waveform L1 | 0.0153 | **0.0112** (−27%) | real robustness improved |
| cross-context consistency, MR-STFT log-mag | 0.0456 | **0.0343** (−25%) | real robustness improved |
| student/teacher output RMS ratio | +0.00 dB | +0.03 dB | no suppression drift |
| worst anchor ratio | −0.09 dB | −0.48 dB | far from the −12 dB collapse line |
| anchor divergence from frozen teacher (L1) | 0.0008 | 0.0068 | ≪ 0.089 full-passthrough scale |
| passthrough margin (corr to raw, student − teacher) | +0.0004 | +0.0136 | ≤ 0.05 guard; no passthrough |
| gain-equivariance error (waveform L1, ±3 dB corrected) | 0.0321 | 0.0323 | unchanged |
| click-band retention vs teacher | −0.00 dB | **+0.10 dB** | transients slightly fuller |
| friction-texture flatness delta vs teacher | — | 0.0000 | no friction-texture regression |
| output level vs raw window | −4.43 dB | −4.47 dB | same attenuation class as zero-shot |
| catastrophic collapse rate (>12 dB under teacher) | 0.0 | **0.0** | S1W's failure absent |
| collapse rate (>6 dB under teacher) | 0.0 | **0.0** | guard never engaged |

## Checkpoint validity (protocol section 32)

Every primary-run checkpoint (250…1500) satisfied ALL validity conditions at every
DEV evaluation: no collapse; no raw passthrough; teacher-anchor selectivity
preserved; cross-context consistency improved vs zero-shot; no transient/fidelity
proxy regression. Selection therefore picked on the consistency axis alone among
valid candidates — `full_upd01500` (0.0455) edged `full_upd0500` (0.0458); the
difference is within evaluation noise, and both are conservative outcomes.

## Anti-collapse gate (protocol section 33) — PASS

`collapse_rate_12db = 0.0`, `median ratio = +0.03 dB` (defensible band), worst anchor
−0.48 dB (no widespread >12 dB suppression), passthrough margin +0.014 (no widespread
passthrough). The gate passed before any TEST/SEALED data was touched.

## Interpretation guard

Consistency is not correctness. These DEV numbers establish that real-mixture
adaptation is STABLE and preserves verified zero-shot behavior; they do not by
themselves establish product superiority. The sealed groups and the human listening
pack carry that question.
