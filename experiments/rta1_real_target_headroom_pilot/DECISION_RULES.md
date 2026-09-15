# RTA1 — Pre-Registered Decision Rules

Fixed before any real capture is ingested. These are **decision margins, not
claims of universal perceptual thresholds**. The headroom question is scored
on the **16 held-out mixtures (sessions 02+03)**.

## The pre-registered success requirement (Outcome A test)

On **≥ 12 / 16** held-out mixtures, with the passing set covering **both**
held-out sessions and **all four** target strata, the optimized bounded oracle
must achieve, versus **BOTH** frozen baselines (zero-shot CLAPSep and original
S1R):

- **≥ 6 dB** reconstruction-SNR advantage, and
- target-path distortion **≤ −15 dB NMSE**, and
- **≥ 10 dB** nuisance attenuation, and
- **no omitted scored weak event**, and
- **no broken scored friction/tail interval**, and
- scored event levels within **±1 dB** of truth.

## Outcomes

**Outcome A — BOUNDED ORACLE CLEARLY BEATS MODELS.**
Interpretation: the representation has substantial headroom; the dominant
limitation is more likely **target supervision / source selection**, not the
representation. Future research *may* justify small real-target supervised
learning — as a new, separately authorized proposal.

**Outcome B — BOUNDED ORACLE FAILS, COMPLEX ORACLE CLEARLY SUCCEEDS.**
Interpretation: real evidence now implicates **representation / phase
freedom**. A phase-capable learner becomes justified as the next hypothesis —
again, separately authorized, not self-started.

**Outcome C — FROZEN MODELS ALREADY APPROACH THE ORACLE.**
Interpretation: little useful representational/selection headroom is shown on
this panel. Do not invent another training objective. The honest next
question moves elsewhere (task definition, evaluation, or product scope).

**Outcome D — TRUTH / ORACLE VALIDITY INSUFFICIENT.**
Interpretation: the measurement problem is unresolved
(`TRUTH_QUALITY_INSUFFICIENT` on too much material, or oracle validity tests
failing on real data). **Stop training.** The pilot reports the measurement
problem; no learning of any kind is authorized.

Ambiguity rule: if A vs C remains genuinely ambiguous after the waveform
evidence, the ≤ 6-item listening check (EVALUATION_PROTOCOL §Human listening)
may be run once, and its outcome recorded as supporting evidence — never as an
equivalence claim.

## Anti-goals (recorded so future stages cannot drift)

- No "universal ceiling" or class-level failure claims from this panel.
- No redefinition of success after seeing results.
- No silent exclusion of failed mixtures (exclusions are recorded with
  reasons, and the denominator stays 16 unless QC disqualifies a mixture
  before any scoring).
- No new learner starts without an explicit new authorization following the
  RESEARCH_WORKFLOW gate.
