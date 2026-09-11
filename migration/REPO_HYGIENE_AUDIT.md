# RhythmAlign Repository Hygiene Audit

Date: 2026-09-12
Scope: read-only audit of `D:\Code\RhythmAlign` prior to research migration to `D:\Code\RhythmAlign-Research`.
No files inside `D:\Code\RhythmAlign` were modified while producing this report (only `git fetch --prune` was run, which refreshes remote-tracking refs only).

## 1. Branch / HEAD / upstream state

| Item | Value |
|---|---|
| Current branch | `main` (attached; `refs/heads/main`) |
| HEAD commit | `44164d1a7321ec9a8ac2977346c3814c6383d227` |
| Configured upstream | `origin/main` |
| Upstream commit (after `git fetch --prune`, exit 0) | `44164d1a7321ec9a8ac2977346c3814c6383d227` |
| Ahead / behind | `0 / 0` (HEAD identical to upstream) |
| Legitimate unpublished product commits | **NONE** (no local-only commits) |

`git fetch --prune` succeeded, so upstream refs are fresh, not stale.

STOP-condition check: HEAD is attached to a normal local branch with a configured upstream — **no stop**.

## 2. Working-tree state (git status --porcelain=v2)

```
? Prompt/
? experiments/
```

| Category | Count |
|---|---|
| Staged files | 0 |
| Tracked modified files | 0 |
| Tracked added files | 0 |
| Tracked deleted files | 0 |
| Untracked top-level entries | 2 (`Prompt/`, `experiments/`) |
| Untracked files under `experiments/` | 4,320 |
| Untracked files under `Prompt/` | 3 |
| Ignored entries (working tree) | 39 top-level entries (expands to 75,347 files under `experiments/` plus product caches) |

## 3. Disk footprint

| Item | Size |
|---|---|
| Whole repository tree | 26 GB |
| `.git` | 3.4 GB |
| `experiments/` (research, untracked+ignored) | 20 GB |
| `dist/` (ignored product build output) | 1.4 GB |
| `.venv/` (ignored product virtualenv) | 764 MB |
| `release/` (ignored product build output) | 319 MB |
| `build/` (ignored product build output) | 74 MB |
| `assets/` (tracked product asset) | 760 KB |
| `Prompt/` (untracked) | 32 KB |

Free space on `D:` at audit time: 199 GB.

## 4. `experiments/` breakdown

| Directory | Size | Notes |
|---|---|---|
| `r2_target_separation` | 8.4 GB | incl. 5.0 GB `.venv` (junk), 5.6 GB checkpoints, 5.8 MB third_party/AudioSep |
| `r3_audio_query` | 7.2 GB | incl. 4.8 GB `.venv` (junk), 4.7 GB checkpoints, 73 KB third_party |
| `s1e_existing_corpus_feasibility` | 2.5 GB | incl. 1.2 GB third_party/SoloAudio (+2.2 GB pretrained weights), audio outputs under ignored `work/`, `clips/audio/`, `outputs/audio/`, `listening_pack/audio/` |
| `r1_golden_sample` | 253 MB | incl. `work/handcam_native.wav` (59.7 MB), `work/ref_native.wav` (51.1 MB), golden_video.mp4 |
| `s1a_ground_truth_pilot` | 244 MB | scripts, tests, fixtures, listening pack |
| `r56_precision_cleanup` | 203 MB | outputs incl. holdoutD_video.mp4 (30 MB) |
| `r55_burst_event_splitting` | 192 MB | outputs incl. holdoutC_video.mp4 |
| `r4_contact_reconstruction` | 174 MB | incl. frames/ (jpg), work/ npy |
| `r5_auto_contact_detection` | 160 MB | outputs incl. holdout_video.mp4 |
| `r5_fix_validity_repair` | 136 MB | listening_pack, outputs |
| `r45_conservative_windows` | 67 MB | outputs, logs |
| `s0_research_reframing` | 216 KB | docs/scripts |
| `independent_review_r0_r56` | 208 KB | review docs |

File counts under `experiments/`: 4,320 untracked + 75,347 ignored (of which ~75k are `.venv`/`__pycache__`/cache contents).

Untracked file types (top): 3,374 jpg (video frames), 272 wav, 185 py, 177 json, 174 png, 53 npy, 41 md, 14 log, 8 csv, 4 npz, 4 mp4, 3 yaml, 2 pt, 2 ckpt, 1 ipynb, 1 gz.

## 5. Largest individual files under `experiments/` (excluding `.venv`)

| Size | Path |
|---|---|
| 2,243.5 MB | `experiments/r3_audio_query/checkpoints/music_audioset_epoch_15_esc_90.14.pt` |
| 2,243.5 MB | `experiments/r2_target_separation/checkpoints/music_speech_audioset_epoch_15_esc_89.98.pt` |
| 1,206.2 MB | `experiments/r2_target_separation/checkpoints/audiosep_base_4M_steps.ckpt` |
| 564.3 MB | `experiments/s1e_existing_corpus_feasibility/work/soloaudio_ckpt/soloaudio_v2.pt` |
| 564.3 MB | `experiments/s1e_existing_corpus_feasibility/third_party/SoloAudio/pretrained_models/soloaudio_v2.pt` (byte-duplicate of the above) |
| 523.6 MB | `experiments/s1e_existing_corpus_feasibility/work/soloaudio_ckpt/audio-vae.pt` |
| 523.6 MB | `experiments/s1e_existing_corpus_feasibility/third_party/SoloAudio/pretrained_models/audio-vae.pt` (byte-duplicate of the above) |
| 169.6 MB | `experiments/r3_audio_query/checkpoints/best_model.ckpt` |
| 59.7 MB | `experiments/{r1_golden_sample,s1e_existing_corpus_feasibility}/work/handcam_native.wav` (duplicate pair) |
| 51.1 MB | `experiments/{r1_golden_sample,s1e_existing_corpus_feasibility}/work/ref_native.wav` (duplicate pair) |
| 23–30 MB | experiment output videos (`golden_video.mp4`, `holdout*_video.mp4`) |

## 6. Classification of dirty content

| Class | Content | Disposition |
|---|---|---|
| A. Product source changes | **None** (tracked tree is exactly at upstream) | n/a |
| B. Research source/scripts/reports | `experiments/**` md/py/json/yaml/csv/log/ipynb + nested `.gitignore` in s1e | Migrate |
| C. Research audio/video outputs | `experiments/**` wav/mp4/npy/npz/frames + s1e ignored audio dirs (`work/`, `clips/audio/`, `listening_pack/audio/`, `outputs/audio/`) | Migrate |
| D. Model/checkpoint/third_party assets | r2/r3 `checkpoints/`, s1e `third_party/SoloAudio` (+pretrained weights), r2 `third_party/AudioSep`, r3 `third_party/CLAPSep` | Migrate |
| E. Caches/temp/generated junk | `experiments/*/.venv` (9.8 GB), `**/__pycache__`, `**/.pytest_cache` (29 dirs) | NOT migrated (recreatable environment/cache junk); documented, not silently dropped |
| F. Unknown | `Prompt/` — 3 markdown AI-task prompts for **product** development (independent code review; no-audio-export fix EN/ZH). The fix prompt explicitly states the pre-existing untracked `Prompt/` directory must not be deleted/modified/staged/committed. | KEEP in product repo, untracked; classified KEEP, not research. Not migrated. |

Note: product-side ignored dirs (`build/`, `dist/`, `release/`, `.venv/`, `.idea/`, `__pycache__/`, `.pytest_cache/`, `tests/__pycache__/`) are product-local caches/build outputs, untouched by this migration and irrelevant to git cleanliness (ignored, never listed by `git status`).

## 7. Upstream tracking of `experiments/`

`git ls-tree -r --name-only @{u}` contains **no path under `experiments/`**. Upstream top level: `.claude .gitignore LICENSE README.md README_zh.md RELEASE.md RhythmAlign.iss RhythmAlign.spec app_info.py assets auto_sync.py bundled_update.json config.json diagnose_offset.py diagnostics.py locales requirements-dev.txt requirements.txt tests ui_main.py update.json update_checker.py`.

The goals "no `experiments/` in the product repo" and "exact upstream parity" do **not conflict**. No stop.

## 8. `.gitignore` state

- Tracked `.gitignore` is **byte-identical to `origin/main`** (`git diff @{u} -- .gitignore` is empty).
- Content covers only product patterns: `.venv/`, `__pycache__/`, `*.pyc`, `*.pyo`, `build/`, `dist/`, `release/`, `.idea/`, `Thumbs.db`, `Desktop.ini`.
- No research-only rules exist in the tracked `.gitignore`.
- `.git/info/exclude` contains no active rules (comments only). Global excludes file (`~/.config/git/ignore`) contains only `.claude/settings.local.json` patterns — unrelated to research.
- Research-specific ignoring is done by a **research-authored nested file**: `experiments/s1e_existing_corpus_feasibility/.gitignore` (`*.wav`, `*.mp4`, `*.flac`, `*.npy`, `work/`, `clips/audio/`, `listening_pack/audio/`, `outputs/audio/`, `diagnostics/spectrograms/`) plus SoloAudio's own clone `.gitignore` files. These live inside `experiments/` and will migrate with the tree; the product repo loses them when `experiments/` is removed.

## 9. Conclusion / go-forward

- Branch state: **Phase 9 CASE A** (HEAD == upstream). No ff-merge needed, no local commits to classify.
- No legitimate unpublished product commits exist.
- The only product-side working-tree item is untracked `Prompt/` (class F→KEEP, intentionally untracked per its own embedded instructions). Because it is legitimate product-side content that must be preserved, the final `git status --porcelain` will contain exactly `?? Prompt/` and cannot be made empty without owner instruction. This is reported rather than silently "fixed"; no new ignore rules will be authored.
- Migration set: all of `experiments/` **except** `.venv`, `__pycache__`, `.pytest_cache` = **5,601 files / 9.67 GB** (10,386,746,880 bytes), copy-first, SHA-256-verified before any source deletion.
