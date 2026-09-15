# RTA1 — Truth Definition

## The target (the only definition used by this pilot)

**The acoustic contribution of the actual player–machine physical interaction,
as it arrives at the recording phone/microphone position.**

Preserved properties (all part of the truth — none may be discarded):

- **timing** — when events actually happened at the machine,
- **source-image level** — how loud each event is at the phone position,
  including the level differences between weak and strong events,
- **room response** — the reflections/coloration of the arcade space between
  machine and phone,
- **decay/tails** — how each impact rings out and dies,
- **weak contacts** — grazes, light taps, partial presses,
- **continuous friction** — slider travel, palm drag, button-release texture.

## Excluded (nuisance — never part of the target)

- cabinet playback music (the game's own song through the cabinet speakers),
- neighboring machines (music, impacts, system sounds),
- NPC / system / announcer speech from any cabinet,
- unrelated human speech (staff, bystanders, the player's own voice),
- ambient unrelated noise (HVAC, doors, traffic).

## Non-negotiable constraints

1. **The phone-domain recording is the only automatic truth candidate.** The
   target truth candidate must come from the same class of phone/microphone
   geometry the product uses. A close/contact microphone must never be treated
   as phone-domain waveform truth, even if it sounds cleaner.
2. **No learned cleanup.** The target is never synthesized, denoised,
   separated, or "repaired" by any learned model. Deterministic, documented
   operations (gain, trimming, reproducible mono derivation) are allowed on
   derived copies only; originals stay untouched.
3. **Quiet is not truth.** A quiet take is a *candidate*. It becomes scored
   target truth only after passing the QC gate
   ([TRUTH_QC_PROTOCOL.md](TRUTH_QC_PROTOCOL.md)) — including the requirement
   that residual unrelated contamination be measurable at least ~20 dB below
   scored target activity where measurable. If it cannot be established, the
   take is marked `TRUTH_QUALITY_INSUFFICIENT` and is not forced into use.
4. **Secondary sensors are evidence, not truth.** Close mic / second phone /
   video are event-and-timing evidence for QC and alignment. They are not
   automatic ground-truth waveforms.

## Native channels

Capture preserves native channels (no downmix at ingestion). Whether the two
native channels carry useful independent spatial information is **measured**
(`02_channel_diagnostic.py`), not assumed. Later mono working signals are
derived reproducibly and documented in manifests.
