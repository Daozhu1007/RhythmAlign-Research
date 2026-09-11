# RhythmAlign S0 erratum — Holdout-D "known rest" interval [6, 8] s

**Recorded 2026-09-10 (during S1a).** This is a correction of record, written after the
S0 reports were frozen. Historical files are intentionally left unchanged for provenance;
this document is the authoritative correction.

## What was corrected

The previously inherited Holdout-D interval **[6, 8] s** of the 18 s Holdout-D excerpt
(corresponding approximately to **66–68 s of the original source video**) was described
in earlier material as a **known rest** — a "hands away from glass" interval with no real
interaction — and was used as ground truth for "rest exposure" validity checks.

Subsequent manual inspection of the original video **confirmed that real gameplay taps
occur inside this interval**.

## Consequences

1. **[6, 8] s is NOT a valid known-rest ground-truth interval.** It must not be cited as
   verified rest in any current or future analysis.
2. **Withdrawn conclusion — detector "false exposure at rest":** the specific finding that
   E0 "fails" the known-rest validity test by exposing 1.425 s of gain>0 support inside
   [6, 8] s (and the shifted-control comparison 1.425 s → 1.726 s built on the same
   interval) is withdrawn. Detector output inside [6, 8] s may correspond to **real
   interaction**, so "exposure" there cannot be scored as false exposure.
3. **Withdrawn conclusion — oracle digital-zero behavior:** the oracle gate's exactly-zero
   output inside [6, 8] s cannot be interpreted as *correct silence*. Under the corrected
   annotation it is instead consistent with the oracle **closing on real events**, i.e. it
   is evidence of possible target loss, not evidence of correct rest handling.
4. **Historical files remain unchanged.** In particular
   `experiments/r5_fix_validity_repair/REPORT.md` §5 ("已知休息段分析 G1 rest = [6, 8] s"),
   `corrected_comparison_matrix.json` (`rest columns`, `primary_questions.*` rest entries),
   `scripts/00_freeze_historical.py` (`rest_stats(t0=6.0, t1=8.0)`) and the
   `independent_review_r0_r56` rest-exposure discussion are preserved as-is; readers must
   apply this erratum when interpreting them.
5. **Not invalidated by this correction:** the corrected C1 fade-bug finding, the corrected
   HPSS finding, and the S0 decision to acquire new synchronized ground truth. The
   envelope click-artifact finding (start-of-span click artifacts inflating old C1 onset
   counts) is independent of the rest annotation.
6. **Status of old Holdout-D:** old Holdout-D data is **exploratory evidence only** and
   must never be used as independent S1 ground truth, as a known-rest interval, or as an
   oracle-events reference.

## Scope note

Per the S1a stage instructions, no attempt is made in this stage to repair or re-annotate
the historical Holdout-D detector/annotation artifacts. The affected quantities are
flagged here; any future re-analysis must re-annotate [6, 8] s from the original video
before drawing rest-related conclusions.
