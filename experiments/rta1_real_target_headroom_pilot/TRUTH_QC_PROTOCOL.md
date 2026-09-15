# RTA1 — Truth QC Protocol

Purpose: decide which captured takes may be used as scored target truth.
**A take is not ground truth merely because it is quiet.**

Tool: `scripts/03_truth_qc.py` (automatic measurements) + a short manual
review pass. Output per take: **PASS / REVIEW / FAIL** with reasons; takes
that cannot establish the contamination requirement are additionally marked
**TRUTH_QUALITY_INSUFFICIENT** and are never forced into use.

## Automatic checks

| check | method | FAIL if | REVIEW if |
|---|---|---|---|
| clipping | fraction of samples ≥ 0.99 · full scale (per channel) | > 0.1 % of samples in any channel | 0.01–0.1 % |
| decode / integrity | full decode, duration vs session metadata | decode error, or duration off by > 5 % | 2–5 % |
| channels | per-channel RMS; per-channel silence runs | a channel is entirely silent | dropouts: silent runs > 2 s below −80 dBFS while other channel active |
| noise floor | 5th percentile of 0.5-s frame RMS in take-A rests | floor unusable (no rests present) | floor within 20 dB of scored-active level |
| scored-active vs rest | 75th-percentile frame RMS inside declared active regions vs 20th-percentile inside declared rest regions (medians when regions are derived) | — | separation < 20 dB ⇒ cannot establish the contamination requirement ⇒ `TRUTH_QUALITY_INSUFFICIENT` for truth use |
| continuity | non-finite samples, zero runs, container anomalies | non-finite samples or file truncation | repeated encoder gaps |

Notes:

- Scored activity for take A = the time regions the session file marks as
  containing deliberate interaction (per-strata regions may be supplied in a
  region file; otherwise the top-level "active spans" from the manual pass).
  The script never infers truth from low RMS alone: a uniformly quiet take
  with no measurable active/rest separation is `TRUTH_QUALITY_INSUFFICIENT`,
  not automatically accepted.
- The **~20 dB requirement**: residual unrelated contamination should be at
  least 20 dB below scored target activity where measurable. "Where
  measurable" is honest: if the take contains no usable rest span, the
  separation is recorded as `NOT_ESTABLISHABLE` and the take needs an explicit
  manual decision, recorded with reasons.

## Manual review (short, after the automatic pass)

For each take flagged REVIEW (and spot-check PASS takes): listen to 3 random
10-s spans + 2 rest spans; note any of — audible music, neighbor machines,
voices, phone handling noise, AGC pumping. Record verdict + reasons in the QC
output (the script supports `--manual-json` to merge these flags).

## Outcomes

- **PASS** — may be used as scored target truth (numbers recorded).
- **REVIEW** — usable only after the manual pass gives an explicit, recorded
  okay; reasons preserved.
- **FAIL** — excluded from truth use (may still serve as descriptive
  evidence; never silently deleted).
- **TRUTH_QUALITY_INSUFFICIENT** — the measurement problem is unresolved for
  this take; per [DECISION_RULES.md](DECISION_RULES.md) this is a legal
  overall outcome (Outcome D) and stops the pilot honestly.
