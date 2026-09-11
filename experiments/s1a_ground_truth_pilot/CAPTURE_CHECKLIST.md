# S1a capture checklist (print this page)

Before leaving for the arcade — bring:

- [ ] Phone/camera used for handcam recording (charged, storage free)
- [ ] Optional: linear recorder / second phone (charged)
- [ ] Optional: contact mic + recorder, close mic + stand (only if permitted/no invasive mounting)
- [ ] Tripod or stable phone mount, if available
- [ ] Operator permission confirmed (only needed for sensor placement / volume changes)

At the machine — setup:

- [ ] Phone settings: recording app set to highest quality / WAV if possible;
      noise-reduction / AGC / "enhancer" OFF if the option exists; note the settings
- [ ] Position the phone exactly where handcam video is normally shot
- [ ] Start ALL recorders; check levels: loudest expected tap below −6 dBFS
- [ ] Do not touch recorder gain again after this point

Recording (one continuous take if possible, ~5–8 min):

- [ ] **Sync start:** clap loudly ON CAMERA + one firm screen tap (count "3, 2, 1" out
      loud, clap on zero)
- [ ] **Quiet interaction, 3–5 min:** the 12-item sound menu — light taps, firm taps,
      each physical button, button releases, palm hits, slow slide, fast slide, long
      continuous slide (≥3 s), weak touches, strong hits, dense burst (5–10 s), sparse
      taps, several ≥3 s rests (hands away), then a natural gameplay-like combination
- [ ] **Playback-only, 1–2 min:** hands off; attract/music/ambience only
- [ ] **Ordinary gameplay, 1–2 min:** play normally (music on)
- [ ] **Sync end:** same clap + tap on camera, count out loud
- [ ] **Voice note (~30 s):** date, time, devices, positions, phone settings, music
      volume/mute state, anything unusual

After capture — at home:

- [ ] Copy files UNCHANGED (no trimming/conversion) to
      `experiments/s1a_ground_truth_pilot/raw/<session_id>/`
      - phone audio+video → `phone/`
      - auxiliary recorder tracks → `aux/`
      - pristine music files used by the cabinet → `pristine/` (only if known)
      - `session_notes.txt` filled in
- [ ] Tell the engineer the files are in place — ingest/QC runs from there

Do NOT: rename files arbitrarily, trim silence, convert codecs, "clean up" audio, or
delete anything. Raw means raw.
