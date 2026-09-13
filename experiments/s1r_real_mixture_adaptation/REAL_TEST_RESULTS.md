# S1R — REAL_TEST_RESULTS (frozen sealed primary test, machine-side)

**Date:** 2026-09-14 · Run AFTER weights, query, and inference protocol were fully
frozen. Source: the SAME sealed S1E/S1W recording and the SAME six product-relevant
groups. Exactly three methods — RAW, ZERO-SHOT CLAPSep, S1R STUDENT. The S1W adapted
checkpoint is a known failed model and is excluded by design (protocol section 36).
Fixed gain: no normalization of any method output. Inference: exact 10-s chunk
overlap-add, identical for both models. Descriptive proxies only (no stems exist).

| group | zero-shot click ret (dB) | S1R click ret (dB) | RMS raw→zero-shot→S1R (dBFS) | S1R − zero-shot RMS |
|---|---|---|---|---|
| c1_speech_npc_a | −0.35 | **−0.33** | −14.7 → −19.0 → −19.1 | −0.04 |
| c2_speech_npc_b | −0.27 | **−0.23** | −16.4 → −20.4 → −20.4 | +0.08 |
| c3_taiko_prompt | −1.11 | **−1.03** | −17.4 → −21.5 → −21.4 | +0.10 |
| c6_dense_golden | −0.26 | **−0.21** | −17.3 → −20.8 → −20.7 | +0.10 |
| c7_weak_taps | −0.59 | **−0.55** | −15.0 → −20.5 → −20.4 | +0.18 |
| c8_slide_friction | −0.38 | **−0.34** | −13.9 → −18.9 → −18.9 | +0.05 |

## What the machine-side says — stated plainly

**S1R behaves like a slightly refined zero-shot on the sealed groups.** On all six
groups the student keeps click-band transients marginally CLOSER to raw than
zero-shot (+0.02…+0.08 dB), holds output level within ±0.18 dB of zero-shot, and
shows zero collapse — on the same groups where the S1W adapted model output ≈ −50 dB
(near-silence). Mid-band (150–2000 Hz) levels match zero-shot within 0.1 dB; on the
c3 Taiko group S1R's friction-texture flatness is slightly LOWER than zero-shot's
(0.153 vs 0.192), i.e., marginally less smearing of that texture.

**The machine side alone cannot claim a separation-quality WIN.** The training design
was deliberately conservative (teacher anchoring, L2-SP, 5% trainable), and the
measured deltas over zero-shot are small: better transient retention and stability,
essentially equal suppression. Whether the small robustness/fidelity refinements are
audible and preferable is exactly what the 18-item blind pack is for — per protocol
section 40, the human pass decides between verdicts A and B; the machine side has
already ruled out the failure modes that forced verdict C in S1W.

Honesty notes: no speech-suppression proxy is claimed for c1/c2 (in-music NPC speech
is VAD-invisible — S1E finding). No SI-SDR/recall/attenuation claims (no ground
truth). The fixed-gain pack audit validates relative levels to ≤0.05 dB
(`listening_pack/LISTENING_PACK_GAIN_AUDIT.md`).
