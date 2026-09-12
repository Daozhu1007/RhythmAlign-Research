# S1W — TARGET_PROXY_AUDIT

**Date:** 2026-09-12 · **Label:** every mined event is TARGET_PROXY, NOT ground truth.
Mined from TRAIN-split recordings ONLY (36 identities; sealed and DEV/TEST_GENERALIZATION
recordings contributed nothing). Audio-only signals; no vision, no chart/note timing,
no separator outputs, no synthetic Foley.

## Method (declared before mining)

- Impact events: adaptive click-band (1.5–10 kHz) spectral-flux onset picking; per-event
  transient-to-stationary contrast (click band, peak vs pre-onset); context gates:
  silero-VAD speech rejection (< 0.4 max prob in ±0.5 s), clipping rejection, and a
  pre-onset stationary-level cap (per-recording mid-band p85, excludes the loudest 15 %
  contexts = dense-music moments). Ranking prefers high contrast in quiet context —
  NOT the loudest absolute onsets. Minimum 1.5 s spacing per class.
- Friction / sustained contact: long-contact spectral continuity spans — sustained
  1–6 kHz energy ABOVE the recording median with an onset-density floor (< 0.03
  onset-frames per frame over 1 s, i.e., no sharp attacks) and low VAD; 2 s cuts at the
  max-coverage position (≥ 0.55), stationary mid-band level ≤ p75. Absolute spectral
  flatness alone proved music-dominated (only fired in music gaps) and was replaced by
  this onset-absence morphology during mining; the change was made on TRAIN diagnostics
  only, before any model training or test evaluation.
- Classes: strong_impact (contrast ≥ 12 dB), impact (6–12 dB), weak_contact (3–6 dB,
  quiet-context), friction/sustained contact (continuity spans).

## Yield

| Class | Events | Seconds | Train recordings |
|---|---|---|---|
| strong_impact | 36 | 54.0 | 22 |
| impact | 81 | 121.5 | 28 |
| weak_contact | 31 | 46.5 | 17 |
| friction | 8 | 16.0 | 7 |
| **total** | **156** | **238.0** | **32/36** |


- Per-event music-bleed estimate (pre-onset band level minus event peak): median
  9.3 dB, IQR 6.0…12.7 dB.
  Negative values mean the context around the event is quieter than the event peak —
  i.e., the transient dominates its window. This is an ESTIMATE of relative music
  bleed inside the proxy cuts, not a measured stem ratio.

## Quality gate (protocol section 20)

- Required: ≥ 3 distinct TRAIN recordings AND ≥ 3 minutes (180 s) conservative material
  with sharp-tap, weak, dense, sparse AND slide/friction coverage.
- Result: **PASS at the preferred level** — 238.0 s ≥ 180 s across
  32 recordings (36-recording split), all five interaction
  strata represented (strong 36, ordinary 81,
  weak 31, friction 8 from 7 recordings).
  No reduced-pilot relaxation was needed.

## Known limitations (honest)

1. Proxies are cut from real mixed recordings: every event carries residual
   cabinet-music bleed (quantified above) and room reverb. The mixture bookkeeping in
   training is still exact (y = s_proxy + n with declared gains); the IMPERFECTION is
   inside s_proxy itself.
2. VAD rejects only acoustically clear speech; S1E showed masked NPC speech is
   VAD-invisible. Speech contamination of proxies cannot be excluded by VAD alone —
   mitigated only by the context gates. Listed as a standing risk for the proxy bank.
3. weak_contact events (3–6 dB contrast) are the closest to the noise floor of their
   contexts; some may be music transients rather than player interaction. They are
   kept as a separate class so training emphasis can be analyzed per class.
4. Friction events are the rarest class (8 cuts, 7 recordings): sustained friction
   without impacts or music onsets is genuinely scarce in this corpus.
5. No per-event verification by listening was performed (owner listening budget is
   reserved for the primary benchmark); verification is deferred and disclosed.
