# S1R — Listening Pack Gain Audit (v1, final)

**Date:** 2026-09-14 · Pack: 18 items = 6 groups × {RAW, ZERO-SHOT, S1R}, built once
under the preregistered fixed-gain protocol (no per-item normalization ever existed
in this stage; the S1W v1 mistake is recorded in the S1W pack history and was not
repeated). This audit is ROLE-KEYED only; the anonymous item ↔ method mapping is
never printed here and stays sealed in the local key until the listening pass is
done — safe to read while blind.

## Protocol validation

- **One common gain per group, derived from the group's RAW source only** (raw peak
  brought to 0.7 ≈ −3.1 dBFS), applied identically to that group's zero-shot and S1R
  items. No per-item peak/RMS/loudness normalization, no method-specific gain.
- **Relative-level preservation:** for every item, the packed file's RMS was compared
  to its source RMS + 20·log₁₀(group gain). Maximum absolute error across all 18
  items: **0.00001 dB** — tolerance ≤ 0.05 dB: PASS with four orders of magnitude of
  margin. (Numeric rows: `work/private/PACK_GAIN_AUDIT.private.json`, local only.)
## Per-group relative levels (dB, role-keyed; packed file measurements)

| Group | Common gain applied | ZERO-SHOT − RAW after packing | S1R − RAW after packing |
|---|---|---|---|
| c1_speech_npc_a | +3.99 dB | −4.37 | −4.41 |
| c2_speech_npc_b | +3.68 dB | −4.05 | −3.97 |
| c3_taiko_prompt | +3.94 dB | −4.22 | −4.12 |
| c6_dense_golden | +3.64 dB | −3.52 | −3.42 |
| c7_weak_taps | +4.15 dB | −5.56 | −5.38 |
| c8_slide_friction | +4.27 dB | −5.11 | −5.06 |

Reading: within every group the separation methods sit 3.4–5.6 dB under the RAW item
— the methods' genuine attenuation behavior (same class as S1E/S1W) — and ZERO-SHOT
vs S1R sit within 0.2 dB of each other, reflecting their machine-side parity. These
loudness relationships are real model behavior, deliberately NOT normalized away;
large loudness differences between methods of the same group would themselves be
evidence (they do not occur here). Expect no near-silent items: that failure mode
(S1W's) is absent from this stage by construction and by measurement.
