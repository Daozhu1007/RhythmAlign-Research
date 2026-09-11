# RhythmAlign Repository Hygiene — Final Migration Report

Date: 2026-09-12
Task: migrate the research laboratory out of the product repository without losing the laboratory notebook.
Predecessor document: `REPO_HYGIENE_AUDIT.md` (read-only Phase 0 audit).

## 1. Starting HEAD / branch

- Branch: `main` (attached), HEAD `44164d1a7321ec9a8ac2977346c3814c6383d227`
- Upstream: `origin/main` @ `44164d1a7321ec9a8ac2977346c3814c6383d227` (refreshed via `git fetch --prune`, exit 0)
- Ahead/behind: `0 / 0` — Phase 9 **CASE A** (HEAD == upstream; no ff-merge needed, no local commits existed)

## 2. Starting dirty-file counts

- `git status --porcelain=v2`: 2 entries — `? Prompt/`, `? experiments/`
- Tracked modified / added / deleted / staged: **0 / 0 / 0 / 0** at all times during the task
- Untracked files: 3 under `Prompt/`; 4,320 under `experiments/` (plus 75,347 ignored files under it, ~75k of which were `.venv`/`__pycache__` cache contents)

## 3. Starting disk footprint

- Product repo tree: 26 GB (`.git` 3.4 GB, `experiments/` 20 GB, product build outputs `dist` 1.4 GB / `release` 319 MB / `build` 74 MB, product `.venv` 764 MB)

## 4. What was migrated

**Everything under `experiments/`** (all 13 experiment directories: `r1_golden_sample`, `r2_target_separation`, `r3_audio_query`, `r4_contact_reconstruction`, `r45_conservative_windows`, `r5_auto_contact_detection`, `r55_burst_event_splitting`, `r56_precision_cleanup`, `r5_fix_validity_repair`, `independent_review_r0_r56`, `s0_research_reframing`, `s1a_ground_truth_pilot`, `s1e_existing_corpus_feasibility`) —
reports, scripts, configs, JSON manifests/results, listening keys and packs, logs, figures, frames, WAV/MP4 outputs, model checkpoints (r2/r3 AudioSep checkpoints, r3 best_model, s1e SoloAudio pretrained weights), and research third_party trees (AudioSep, CLAPSep, SoloAudio).

Method: `robocopy /E /COPY:DAT /DCOPY:DAT` (relative structure, filenames and timestamps preserved), copy-first, source untouched until validation passed.

## 5. Research destination size

`D:\Code\RhythmAlign-Research` = **9.7 GB** (`experiments/` 9.7 GB, `migration/` metadata 11 MB), plus this README.

## 6. File / hash verification status

- **5,601 files / 10,386,746,880 bytes (9.67 GB)** copied; robocopy reported 0 failed / 0 skipped
- SHA-256 computed on **every migrated file on both sides**: **5,601 / 5,601 match** → `Migration verification: PASS`
- File-set and size comparison: match (no missing, no extra, no size drift)
- 54 zero-byte files are legitimate empty `__init__.py` / `py.typed` files (hash-verified identical to source)
- S0 structural validators re-run inside the new workspace: **pass** (`validate_deliverables.py`, `check_workspace_evidence.py`)
- Per-file record: `MIGRATION_MANIFEST.json` (source path, destination path, size, sha256, category, tracked/ignored status, action). Categories: 3,999 audio/video outputs, 345 scripts/reports/configs, 1,257 model/checkpoint/third_party. Git provenance: 5,477 untracked, 124 untracked-but-ignored (s1e audio artifacts).
- Key deliverables spot-checked present: `S0_RESEARCH_STUDY.md`, `S0_DECISION.json`, `S1A_REPORT.md`, `S1A_DECISION.json`, `listening_key.json`, `S1E_RESULTS.md` set (`RESULTS.md`, `S1E_DECISION.json`, `RUN_MANIFEST.json`), `INDEPENDENT_REVIEW.md`, all `scripts/` trees.

## 7. What was deleted from the product working tree

Only `D:\Code\RhythmAlign\experiments\` (20 GB), **after** hash-verified migration, and only after confirming `git ls-files experiments` = 0 files (nothing under it was ever tracked). The 9.8 GB of `.venv`, `__pycache__`, `.pytest_cache` caches inside it were intentionally **not** migrated (recreatable environment/cache junk; documented in `MIGRATION_NOTES.md`).

## 8. What was restored from HEAD / upstream

**Nothing needed restoring.** The tracked working tree was byte-identical to `origin/main` before, during, and after the task (0 tracked modifications at every checkpoint). No tracked file was touched, so no `git checkout`/restore operations were required.

## 9. `.gitignore` state

- Starting state: tracked `.gitignore` **byte-identical to `origin/main`** (product patterns only).
- Research-only changes restored? **None existed to restore.** Research-specific ignoring lived in the research-authored nested file `experiments/s1e_existing_corpus_feasibility/.gitignore` (`*.wav`, `*.mp4`, `work/`, `clips/audio/`, `listening_pack/audio/`, `outputs/audio/`, `diagnostics/spectrograms/`), which migrated together with the tree; `.git/info/exclude` was and remains comment-only.
- New ignore rules introduced by this task: **NONE**. `.gitignore` was never modified; no `/experiments/` rule was added (the clean solution — the directory no longer exists — was used instead).

## 10. Remaining dirty files

Exactly one entry: `?? Prompt/` — three product-development AI-task prompt documents (`CodeReviewPrompt.md`, `FixNoAudioExport.md`, `FixNoAudioExport_zh.md`, 32 KB). Classified **KEEP**: they are product-side (code review audit; reliability-fix tasks), not research, and `FixNoAudioExport.md` itself contains a standing owner instruction that the pre-existing untracked `Prompt/` directory must not be deleted, modified, staged, or committed. They were preserved untouched. Classification summary: KEEP = 1 (`Prompt/`), RESTORE = 0, UNKNOWN = 0.

## 11. Final git status

```
$ git status --porcelain
?? Prompt/

$ git status
On branch main
Your branch is up to date with 'origin/main'.
Untracked files: Prompt/
nothing added to commit but untracked files present

$ git diff            -> empty
$ git diff --stat     -> empty
$ git diff --cached   -> empty
$ git diff --check    -> clean
```

The tracked tree is an exact match of upstream. `git status --porcelain` is not literally empty **only** because of the intentionally preserved `Prompt/` directory (see item 10 and the "why" below).

## 12. Unresolved path dependencies

- 31 historical experiment scripts contain absolute `D:\Code\RhythmAlign\experiments\...` path constants. They were deliberately **not** mass-edited (historical evidence rule). Future re-runs must repoint them at `D:\Code\RhythmAlign-Research\experiments\...`. Full list and rationale: `MIGRATION_NOTES.md`.
- `r1_golden_sample/scripts/02_coarse_align.py` imports production module `auto_sync` via absolute `sys.path.insert(0, r"D:\Code\RhythmAlign")` — still valid, since the product repo remains at that path (read-only import).
- Historical reports/manifests naming the old absolute path are intentional provenance and were left as-is.

## 13. Data intentionally NOT migrated

- `experiments/r2_target_separation/.venv/` (5.0 GB) and `experiments/r3_audio_query/.venv/` (4.8 GB) — Python virtualenvs
- all `__pycache__/` and `.pytest_cache/` directories
- Product-side ignored dirs in the repo root (`build/`, `dist/`, `release/`, `.venv/`, `.idea/` — product caches/build outputs, unrelated to research)
- `Prompt/` — intentionally kept **in the product repo** (not research material)
Within-experiment duplicate copies (e.g. `s1e/work/soloaudio_ckpt/*.pt` vs `s1e/third_party/SoloAudio/pretrained_models/*.pt`) were both preserved; no deduplication was attempted.

## 14. Commit / push confirmation

**No commit, no tag, no push, no PR, no history rewrite, no `git reset --hard`, no `git add` was performed.** The only Git commands run were read-only or ref-fetch: `fetch --prune`, `status`, `rev-parse`, `rev-list`, `log`, `ls-tree`, `ls-files`, `check-ignore`, `diff`, `show`, `symbolic-ref`.

## 15. Was exact upstream clean state achieved?

**Yes — for the entire tracked tree.** HEAD == `origin/main` (`44164d1`), `git diff` empty, `git diff --cached` empty, and nothing tracked differs from upstream byte-for-byte. `git status --porcelain` contains the single intentional `?? Prompt/` entry. It cannot be emptied without an owner decision because `Prompt/` is legitimate product-side content protected by an explicit standing instruction embedded in one of the files; this task forbids authoring a new ignore rule for it and forbids deleting it. Options for the owner (review item): keep as permanent untracked scratch area (recommended, current state), relocate it outside the repo, or authorize a `.gitignore` rule in a future product commit.

Product-side behavior is provably unchanged: no tracked byte was modified, and the lightweight-validation clause is satisfied trivially (`git diff` empty ⇒ nothing to test); no packaging/release was run, per instructions.

## Final state summary

```
Product repo:
D:\Code\RhythmAlign

Branch: main
Upstream: origin/main
HEAD:   44164d1a7321ec9a8ac2977346c3814c6383d227
Upstream HEAD: 44164d1a7321ec9a8ac2977346c3814c6383d227

git status --porcelain:
?? Prompt/

git diff:
<empty>

git diff --cached:
<empty>

D:\Code\RhythmAlign\experiments\
<does not exist>

Research:
D:\Code\RhythmAlign-Research

Migration verification:
PASS

Commit:
NO

Push:
NO
```
