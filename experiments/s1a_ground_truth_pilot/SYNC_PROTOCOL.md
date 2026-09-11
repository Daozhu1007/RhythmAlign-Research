# S1a sync protocol

**Goal:** one reproducible procedure that maps every recorder's clock onto a common
timeline well enough for waveform-level QC and exact constructed mixtures, while keeping
clock offset/drift separate from physical acoustic propagation delay.

## 1. Sync events

Two events per session, recorded by every channel (audio + video):

- one **near the start** (after all recorders are rolling, before block 2);
- one **near the end** (after block 4, while everything is still rolling).

Each event is:

1. count down out loud "3 — 2 — 1" (visible on video, audible on all mics);
2. one **loud clap** directly in front of the phone (this is the audio anchor);
3. immediately after the clap, one **firm screen tap** on the machine (secondary anchor).

The countdown makes anchor identification trivial and verifiable on video. Anchor
positions are marked in `raw/<session_id>/anchors.csv` (template in `raw/README.md`).

If a phone restart was unavoidable, add one extra clap+tap at each restart.

## 2. Clock map

For every auxiliary channel `X` (aux recorder, contact channel, video track audio),
fit a linear clock map to the target channel `T` (the phone-position recording):

```
t_T = a + b · t_X
```

- anchors: onset times of the same acoustic events (start clap, end clap; the screen tap
  as a check) measured independently on each channel;
- with exactly 2 anchors: exact linear solve; with ≥3 anchors: least-squares fit, report
  residuals;
- `a` = offset (seconds), `b` = drift ratio (dimensionless). Drift of ~10–100 ppm
  (typical consumer clocks) shifts ~0.6–6 ms per minute; the map absorbs it globally;
- a withheld intermediate anchor (if captured) is used only for validation, never fitting.

Resampling to a common working rate uses a quality resampler (polyphase, documented
filter); original files are never modified.

## 3. Drift vs propagation delay — kept separate by construction

The fitted `a` mixes two physically different quantities:

- **clock offset** between recorders (what we want absorbed by the map);
- **acoustic propagation delay** (a source d meters from a microphone arrives
  d/343 s later; e.g., 0.30 m ⇒ 0.87 ms; a contact sensor on the cabinet body can lead
  the air mic by a comparable amount).

Procedure:

1. record the approximate geometry (phone↔machine distance, aux↔machine distance) in
   `session_notes.txt`;
2. the *expected* propagation difference between channels is computed from geometry and
   recorded in the manifest as `expected_propagation_s`;
3. the sync fit reports `a` and its residual; `a` is interpreted as clock offset plus the
   actual propagation difference; we never "correct" one channel by an arbitrary
   sub-millisecond shift to make peaks coincide — that would erase real geometry;
4. for contact-sensor evidence, sensor lead vs air arrival is expected and documented,
   never removed;
5. alignment tolerance for waveform-comparison work is declared per use: within-session
   paired channels recorded by one multichannel device are sample-locked; separately
   synchronized devices are NOT assumed phase-accurate across channels and are only used
   for event-level (±2 ms) or constructed-mixture (exact-by-construction) work.

## 4. Hard rules

- **Never** independently align a separator/model output to target truth. All alignment
  comes from this protocol's anchors, fixed before any model output exists.
- Anchor onsets are measured by the same audited onset routine on all channels
  (`scripts/sync_fit.py`), with the anchor regions excluded from scored content.
- Fit parameters, residuals, and validation-anchor error are stored in
  `manifests/sync_map_<session>.json` and copied into the pilot manifest.
- `sync_ok` requires: end-clap validation error ≤ 2 ms (separately recorded devices) or
  ≤ 0.1 ms (sample-locked multichannel); otherwise the session is flagged `sync_poor`
  and is not usable for waveform-exact claims (constructed mixtures remain valid because
  their truth is exact by construction from stored components).
