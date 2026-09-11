# License Status

**As of the initial publication (2026-09-12), this repository has NO license file.**
A public GitHub repository without an explicit license means all rights default to the
copyright holder; you may read it, but not legally reuse it. This is a deliberate,
temporary state.

## Why

The workspace mixes three categories of material with different licensing situations:

1. **Original research code and reports** — written for this research program
   (experiment scripts, validators, reports, decision records, protocols). These are
   the author's original work and could be licensed freely.
2. **Locally cloned third-party code** — AudioSep (MIT) and SoloAudio (MIT) were used
   locally but are **not committed** here (see `THIRD_PARTY.md`); their licenses do not
   constrain this repository's own licensing.
3. **CLAPSep-derived material** — three model files from the CLAPSep research code
   (Waveformer-derived, © Hao Ma @SDU) were copied locally during R3. No license file
   was present in that copy, and the modification status vs upstream is not
   established. Those files are **excluded from the repository**, but until the
   upstream licensing situation is clarified, it is premature to declare a clean
   repository-wide license for a program whose current adaptation substrate
   (S1W) builds directly on CLAPSep-class architecture.

## Rule applied

`LICENSE_STATUS.md` follows the bootstrap policy: **do not add a repository-wide
license while any licensing question is unresolved.** If/when the CLAPSep question is
resolved (or the substrate changes to cleanly-licensed components) and the author
decides on a license, this file will be replaced by the license grant itself.

Note: this decision is specific to the research repository. The production RhythmAlign
repository carries its own license and this research repository intentionally did
**not** copy it.
