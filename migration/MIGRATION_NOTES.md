# Migration Notes

Working notes for the 2026-09-12 migration of RhythmAlign research artifacts
from `D:\Code\RhythmAlign` to `D:\Code\RhythmAlign-Research`.
The authoritative file-by-file record is `MIGRATION_MANIFEST.json`.

## Scope decisions

- Migrated: the entire `experiments/` tree (R1–R5.x, R45, R55, R56,
  independent_review_r0_r56, S0, S1a, S1E), including research checkpoints,
  third_party research trees (AudioSep, CLAPSep, SoloAudio), generated WAV/MP4
  outputs, frames, listening packs, keys, manifests, and reports.
- NOT migrated (intentionally, documented here — not silently dropped):
  - `experiments/r2_target_separation/.venv/` (5.0 GB Python virtualenv)
  - `experiments/r3_audio_query/.venv/` (4.8 GB Python virtualenv)
  - all `__pycache__/` and `.pytest_cache/` directories
  These are environment/cache junk, recreatable from migrated scripts and
  requirement files; they contain no research information.
- NOT migrated: `Prompt/` in the product repo — product-development AI-task
  prompts (code review; no-audio-export fix EN/ZH), not research. It stays in
  `D:\Code\RhythmAlign`, untracked, per its own embedded instructions.

## `.gitignore` provenance

- Product `.gitignore` was byte-identical to `origin/main` before the
  migration and was not modified at any point.
- The research-specific ignore rules (`*.wav`, `*.mp4`, `work/`,
  `clips/audio/`, etc.) lived in the research-authored nested file
  `experiments/s1e_existing_corpus_feasibility/.gitignore`, which migrated
  together with the tree. The SoloAudio clone keeps its own upstream
  `.gitignore` files.
- No new ignore rules were authored in the product repository.

## Path compatibility findings (Phase 5 audit of the migrated copy)

The audit searched all migrated text files (`.py .md .json .yaml .yml .csv .txt`)
for absolute references to the old tree (`D:\Code\RhythmAlign` not followed by
`-Research`) and for relative escapes out of `experiments/`.

Findings — **no historical file was edited**; all references are documented here:

1. **Historical reports / manifests / logs (left as historical evidence).**
   Absolute `D:\Code\RhythmAlign\experiments\...` paths appear in
   `independent_review_r0_r56/INDEPENDENT_REVIEW.md` (127 occurrences),
   `s1a_ground_truth_pilot/fixtures/fixture_mixtures_recipe.json` (36),
   `r5_fix_validity_repair/HISTORICAL_BASELINE_MANIFEST.json` (35),
   `s0_research_reframing/S0_RESEARCH_STUDY.md` (18) and `EVIDENCE_AND_SCOPE.md` (12),
   `s1e_existing_corpus_feasibility/RUN_MANIFEST.json` (17) and `logs/*.json`,
   `r2/r3 logs`, `r1_golden_sample/work/audit_report.json`, and others.
   These record where evidence was produced; they are provenance, not
   executable configuration. They intentionally still name the old path.
2. **31 executable experiment scripts** carry absolute
   `D:\Code\RhythmAlign\experiments\...` defaults (r1: 1, r2: 3, r3: 5,
   r4: 7, r45: 2, r5: 2, r55: 3, r56: 2, s1a: 1, s1e: 3, plus r1 common
   helpers). They are historical per-experiment tooling whose outputs and
   manifests are already part of the migrated evidence, so they were **not**
   mass-edited. If any script is re-run in the future, its path constants
   must be pointed at `D:\Code\RhythmAlign-Research\experiments\...`
   (or a workspace-root constant introduced at that time). This is the one
   deliberate mechanical debt taken on by the migration.
3. **Product-module coupling that still works.**
   `r1_golden_sample/scripts/02_coarse_align.py` imports the production module
   `auto_sync` via `sys.path.insert(0, r"D:\Code\RhythmAlign")` — an absolute,
   read-only import. The product repository remains at `D:\Code\RhythmAlign`,
   so this coupling is unaffected by the migration.
4. **No relative escapes out of the research tree.** No migrated script uses
   `../../../`-style relative paths that would reach the product root, and
   none references product `config.json`, `assets/`, or `locales/`.
5. The S0 structural validators (`s0_research_reframing/scripts/validate_deliverables.py`,
   `check_workspace_evidence.py`) are self-relative and were re-run inside
   `D:\Code\RhythmAlign-Research\experiments\s0_research_reframing` — both
   **pass** post-migration.

## Legacy path dependencies

- Only item 3 above (works as-is) and item 2 (future re-runs need path
  updates) remain. Nothing else in the research workspace depends on the
  product repository's file layout.
