# S1R — Human Blind Listening Results (unblinded)

**Listening pass:** one pass, in playback order, per `LISTENING_INSTRUCTIONS.md`,
against the frozen 18-item pack (6 groups × RAW / ZERO-SHOT / S1R, one common
per-group gain derived from RAW only, relative levels preserved to ≤0.00001 dB).
**Unblinding:** the sealed key (`work/private/listening_key.private.json`) was
checked against the mapping relayed with the completed pass — all 18 items match.
The mapping below is no longer secret. **Final verdict: B — REAL-MIXTURE ADAPTATION
IS VIABLE BUT NO CLEAR PRODUCT GAIN** (see `../S1R_DECISION.json`).

## Protocol caveats (recorded before interpretation)

- **Playback device differs across stages.** S1W was heard primarily on speakers;
  this pass was heard on headphones. NO strict cross-stage claims about small timbre
  differences (e.g. underwater texture) between S1W and S1R are made anywhere. The
  S1R internal comparison (RAW vs ZERO-SHOT vs S1R heard within the same headphone
  session) remains valid.
- **Listening fatigue.** The owner experienced substantial fatigue; there is NO
  second listening pass, by design and by instruction.
- One pass, free-form observations only — no rankings, no numeric ratings. The
  per-item notes below are recorded as relayed, in brief paraphrase.

## Item → method mapping (verified against the opened key)

Pack RMS levels are the measured, gain-audited values from
`work/private/PACK_GAIN_AUDIT.private.json` (objective context; packaging verified —
max abs error 1e-05 dB).

| Item | Group | Method | Pack RMS (dBFS) | Owner observation (paraphrased) |
|---|---|---|---|---|
| item_01 | c3_taiko_prompt | RAW | −20.49 | Taiko system prompt extremely obvious; button sound resembles original handcam |
| item_02 | c7_weak_taps | ZERO-SHOT | −23.65 | Neighboring Taiko machine remains audible; unsure about missing interaction |
| item_03 | c1_speech_npc_a | RAW | −17.77 | NPC voice completely obvious |
| item_04 | c1_speech_npc_a | S1R | −22.18 | Button interaction remains very obvious |
| item_05 | c8_slide_friction | S1R | −22.04 | Neighboring Taiko music very obvious |
| item_06 | c7_weak_taps | S1R | −23.47 | Neighboring Taiko obvious; few button sounds; cannot tell deletion vs source segment |
| item_07 | c7_weak_taps | RAW | −18.09 | Same general conclusion as preceding weak-tap examples |
| item_08 | c3_taiko_prompt | ZERO-SHOT | −24.71 | Almost no NPC audible; interaction fairly crisp; some possible discontinuity; potentially very good |
| item_09 | c8_slide_friction | ZERO-SHOT | −22.08 | Neighboring Taiko music remains obvious |
| item_10 | c2_speech_npc_b | S1R | −23.53 | NPC remains obvious |
| item_11 | c2_speech_npc_b | ZERO-SHOT | −23.61 | Same segment and conclusion as item_10 |
| item_12 | c6_dense_golden | ZERO-SHOT | −24.00 | Almost no NPC/background noise; interaction excellent and appears complete; small underwater-like quality remains |
| item_13 | c1_speech_npc_a | ZERO-SHOT | −22.14 | NPC extremely obvious |
| item_14 | c8_slide_friction | RAW | −16.97 | Poor result; background noise not reduced, may perceptually feel worse; neighboring Taiko very obvious |
| item_15 | c6_dense_golden | RAW | −20.47 | Same segment family as item_12; interaction slightly better/more natural; possibly without underwater quality |
| item_16 | c2_speech_npc_b | RAW | −19.56 | NPC obvious |
| item_17 | c3_taiko_prompt | S1R | −24.60 | Almost no NPC audible; slight underwater quality / possible discontinuity; otherwise similar positive behavior to item_08 |
| item_18 | c6_dense_golden | S1R | −23.90 | Same conclusion as item_15 |

## Per-group blinded comparison (ZERO-SHOT vs S1R — the A-vs-B question)

| Group | ZERO-SHOT heard as | S1R heard as | Blinded comparison |
|---|---|---|---|
| c1_speech_npc_a | NPC extremely obvious (13) | Button interaction very obvious (04) | No explicit preference; ambiguous, at most a faint S1R edge on interaction character |
| c2_speech_npc_b | same as S1R (11: "same conclusion as 10") | NPC remains obvious (10) | TIE — owner could not distinguish |
| c3_taiko_prompt | almost no NPC; fairly crisp; possible discontinuity; "potentially very good" (08) | almost no NPC; slight underwater/discontinuity; "similar positive behavior to 08" (17) | TIE — both retain the useful Taiko-prompt selectivity; S1R mirrors zero-shot |
| c6_dense_golden | excellent, complete interaction; small underwater quality remains (12) | interaction slightly better/more natural, possibly without underwater (18, matched to RAW 15) | Slight S1R edge — closest to RAW's interaction naturalness |
| c7_weak_taps | neighbor Taiko audible; unsure about missing interaction (02) | few button sounds; cannot tell deletion vs source (06) | TIE / inconclusive both |
| c8_slide_friction | neighbor Taiko remains obvious (09) | neighbor Taiko very obvious (05) | TIE — neither suppresses the neighbor machine |

**Score: S1R clearly preferred in 0 of 6 groups** (c6 is a faint edge, not a clear
win; c1 is ambiguous). The protocol bar for a product-level claim was preference on
≥3/6 groups. Equally important: **no group where S1R is clearly WORSE than
zero-shot** — no perceived deletion of authentic interaction (the S1W failure
symptom is absent by ear), no raw-passthrough complaint about either separation
method, and the c3 positive selectivity pattern survives adaptation audibly.

## What the human evidence establishes

1. **S1R is perceptually stable and safe.** Nothing the owner heard suggests
   collapse, wholesale deletion, or passthrough — consistent with the frozen
   machine-side record (±0.2 dB RMS of zero-shot, zero collapse, transient
   retention slightly better on all six groups).
2. **S1R ≈ ZERO-SHOT to the ear.** In four of six groups the owner explicitly could
   not distinguish or gave identical conclusions; at most one faint edge (c6
   interaction naturalness) in S1R's favor, none against.
3. **Both models' dominant remaining nuisances are unchanged:** neighboring Taiko
   machines (c7/c8), NPC speech where present (c1/c2), and the same minor
   discontinuity/underwater character on the strongest-suppression items.
4. **Verdict B follows:** real-mixture adaptation is a valid, stable training
   foundation, but teacher-only / consistency-dominated adaptation added no
   perceptible product-level gain over the teacher. This motivates S2A: add a
   genuinely NEW information source (aligned pristine music reference) rather than
   distilling the teacher further.

## What this document deliberately does NOT claim

- No cross-stage S1W↔S1R timbre comparison (different playback devices).
- No suppression-magnitude claims from listening (no ground truth exists; machine
  proxies carry that evidence).
- No preference statistics — one pass, free-form notes, six groups.
