# S1W Listening Pack — Gain Protocol Audit (pack v2)

Pack v1 was **invalid**: it peak-normalized every item independently to −3 dBFS,
violating the preregistered S1W rule (no per-item normalization; a common
input-derived gain convention only). Independent normalization amplifies
near-silent residue and hides the magnitude of target deletion. The historical
v1 mistake is deliberately recorded here and in `LISTENING_INSTRUCTIONS.md`.

## v2 gain rule (regenerated from the same frozen outputs)

- `one common gain per group derived from the RAW source peak only (headroom target -1 dBFS; native scale kept if raw peak is already below target), applied identically to raw/zero-shot/adapted; no per-item normalization of any kind`
- Same seed (20260912) and generation loop as v1 → the anonymous method
  assignment and playback order are unchanged (verified programmatically
  against the sealed v1 key: unchanged = True). The
  item↔method mapping itself is never printed or exposed; this audit is
  role-keyed (RAW / ZERO-SHOT / ADAPTED) only and is safe to read while blind.
- No retraining, no checkpoint change, no inference change: the 12 frozen
  input files are hashed below and were consumed read-only.

## Relative RMS preservation (before packing vs after packing)

Values are RMS differences in dB within the same challenge group, measured on
the frozen float outputs (before) and read back from the packed PCM_16 WAVs
(after). Tolerance: 0.05 dB (resampling/encoding only).

| Group | Common gain | ZERO-SHOT − RAW before → after (dB) | ADAPTED − RAW before → after (dB) | Max abs err (dB) |
|---|---|---|---|---|
| c1_speech_npc_a | -0.94 dB | -4.35 → -4.35 | -32.72 → -32.72 | 0.000 |
| c2_speech_npc_b | -1.06 dB | -4.09 → -4.09 | -30.73 → -30.73 | 0.000 |
| c3_taiko_prompt | -1.15 dB | -4.18 → -4.18 | -27.36 → -27.36 | 0.000 |
| c6_dense_golden | -1.51 dB | -3.51 → -3.51 | -31.13 → -31.12 | 0.000 |
| c7_weak_taps | -0.80 dB | -5.49 → -5.49 | -32.24 → -32.24 | 0.000 |
| c8_slide_friction | -0.94 dB | -4.94 → -4.94 | -14.29 → -14.29 | 0.000 |

**Result: relative gain preservation PASS (max |err| = 0.0004 dB ≤ 0.05 dB).**

Large loudness loss between items of the same group (e.g. ADAPTED tens of dB
below RAW) is genuine model behavior — suppression/deletion magnitude — and is
preserved exactly by packaging; it is itself decision evidence.

## Frozen input integrity (read-only, unchanged)

| Frozen input | sha256 (first 16) |
|---|---|
| adapted_c1_speech_npc_a | f635ed86c9b09abd… |
| adapted_c2_speech_npc_b | 7ac0390198501f40… |
| adapted_c3_taiko_prompt | 3e090a0bd8af4777… |
| adapted_c6_dense_golden | 56ec11fc9f8b02b4… |
| adapted_c7_weak_taps | 6cd61f5601b98978… |
| adapted_c8_slide_friction | ea0c664267bed9f9… |
| raw_c1_speech_npc_a | 81f329a6c39c4601… |
| raw_c2_speech_npc_b | 3061bdbb278af852… |
| raw_c3_taiko_prompt | 01ca3571fe6e5572… |
| raw_c6_dense_golden | 183bded3bf8388bc… |
| raw_c7_weak_taps | 35f8355785a81832… |
| raw_c8_slide_friction | 2c407679b1c9a291… |
| zeroshot_c1_speech_npc_a | a0fb146f6248b5d9… |
| zeroshot_c2_speech_npc_b | 9809b084543be9f2… |
| zeroshot_c3_taiko_prompt | 3fc0eeb85d3c0918… |
| zeroshot_c6_dense_golden | 7715c5fc33ef8bce… |
| zeroshot_c7_weak_taps | 3750b561f8cf469d… |
| zeroshot_c8_slide_friction | be7eb34d8c97bbd9… |
