# S1E — Listening Pack Instructions (READ FIRST)

**Purpose.** Blind listening is the PRIMARY decision evidence for S1E. The question is
not "which spectrogram looks clean" but:

> Before the owner performs a research-grade capture campaign, does any pretrained
> system already show enough real-world promise to justify continuing?

**Pack:** 36 anonymous items in `listening_pack/audio/` (`item_01.wav` … `item_36.wav`).
Each is 5–15 s, mono, peak-normalized to −3 dBFS **for comparability only** (applied
gains are recorded in `listening_key.json`; do not open the key until all ratings are
done). Playback order and clip grouping (but not method identity) are in
`playback_order.json`. Items are grouped 4–5 per challenge clip; within a group, one
item is the untouched raw recording and the others are method outputs. Method identities,
seeds and any generative nature are hidden until ratings are complete.

**Method.** Rate every item of a group before moving to the next group. You may replay
freely, at any volume, headphones recommended. For each item write the six ratings
below (0 = no / bad … 3 = yes / excellent, or n/a when the item contains none of that
content). Trust your ears over any document.

1. **Interaction present?** Do the real player hit/tap/release sounds survive?
2. **Weak hits preserved?** Are the quiet, light touches still audible and natural?
3. **Slides / friction / tails natural?** Any smeared, "underwater", or metallic quality?
4. **Speech reduced?** (clips grouped around 1:26–1:50 and the end-of-video announcer)
5. **Loud system prompt reduced?** (the group around 2:08–2:09) **And is any
   simultaneous player interaction damaged?**
6. **Arcade ambience / music reduced?**

Then per item note, one line each:

- **Artifacts?** (musical noise, chirps, phasing, pumping, hollow spectra)
- **Invented content?** anything that sounds synthesized or like a sound that "wasn't
  there" (this includes sounds that appear where the raw group member has silence)
- **Overall preference** vs the group's raw member: worse / equal / better.

**The product rule (most important):** a method that removes nuisance but noticeably
deletes authentic interaction **does NOT win**. If in doubt, compare the item directly
against the raw member of its group.

**After rating all 36 items**, open `listening_key.json`, un-blind, and transfer the
ratings into the group table at the end of `RESULTS.md` (or a free-form notes file —
`listening_ratings.md` next to this document). The stage verdict in `S1E_DECISION.json`
is provisional until this listening pass exists; the upgrade/downgrade rules are in
`RESULTS.md §Decision`.
