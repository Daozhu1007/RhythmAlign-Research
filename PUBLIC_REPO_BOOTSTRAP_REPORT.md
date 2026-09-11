# Public Repository Bootstrap Report

Date: 2026-09-12
Task: turn `D:\Code\RhythmAlign-Research` into the public research repository
`Daozhu1007/RhythmAlign-Research` following the 15-part bootstrap protocol.

## Product repository status

| Item | Value |
|---|---|
| Path | `D:\Code\RhythmAlign` |
| `git fetch --prune` | OK |
| `git status --porcelain` | empty (clean) |
| HEAD | `44164d1a7321ec9a8ac2977346c3814c6383d227` |
| Upstream (`origin/main`) | `44164d1a7321ec9a8ac2977346c3814c6383d227` |
| HEAD == upstream | YES |
| Modified during this task | **NO** — read-only verification only |

## Research repository

| Item | Value |
|---|---|
| URL | https://github.com/Daozhu1007/RhythmAlign-Research |
| Visibility | **PUBLIC** |
| Branch | `main` (default; tracks `origin/main`) |
| Initial commit | `cebc7ae7f155fbf20451058cf13e960680b8f772` — "research: publish RhythmAlign experimental history" |
| Report follow-up commit | this file |
| HEAD == origin/main | YES |
| Authentication | GitHub CLI, account `Daozhu1007` (keyring); no tokens displayed |

## Publication numbers

| Item | Value |
|---|---|
| Tracked files | 403 (402 in `PUBLICATION_MANIFEST.json` + the manifest itself) |
| Tracked payload | ~51.2 MB working tree; `.git` ≈ 49 MB |
| Largest tracked file | 1.36 MB (`migration/MIGRATION_MANIFEST.public.json`); no file > 3 MB |
| Local-only (ignored) artifacts | ≈ 9.65 GB (checkpoints ≈ 6.6 GB, SoloAudio clone+weights ≈ 1.2 GB, work/ trees ≈ 1.6 GB, media/frames/outputs) |
| Git LFS used | NO |

Composition of the tracked set: 74 small diagnostic PNG plots (46.4 MB), 130
metrics/decision/manifest JSONs (2.4 MB), 131 research scripts (0.8 MB), 39+6
reports/root documents (0.5 MB), 14 run logs, 1 CSV + 1 TXT, stage `.gitignore`,
sanitized migration manifest, migration reports.

## Secret scan result

**PASS.** Pattern scan of all non-vendored text files (API-key shapes, credential
files, PEM blocks, Authorization headers, e-mails, recording-metadata leaks) found no
credentials. Apparent hits were false positives (`task-specific` matching `sk-…`,
metric name `hf_retention`, public arXiv/DCASE URLs).

Privacy findings handled by exclusion (no destructive redaction):

- `experiments/r1_golden_sample/work/audit_report.json` embeds phone metadata from the
  source recording (GPS coordinates, device model, copyrighted-song metadata). It is
  inside the ignored `work/` tree and remains local-only, unmodified.
- 14 research files reference local media paths (`D:\Daozh\...`) — kept per policy as
  provenance in historical records; no media content published.

## Third-party / license result

- All three local third-party trees **excluded** from Git and documented by reference
  in `THIRD_PARTY.md`: AudioSep (MIT), SoloAudio (MIT, clone carried its own `.git`
  and `pretrained_models/`), CLAPSep (3 model files, no license file locally —
  licensing unclear → excluded from the initial publication per protocol).
- No model checkpoints redistributed.
- **No LICENSE file added.** Reason: mixed/unclear licensing posture (CLAPSep
  substrate) — documented in `LICENSE_STATUS.md`. An unlicensed public repository is
  accepted for the initial publication.

## Excluded artifact categories (verified absent from the tracked set)

audio/video (389 WAV, 4 MP4), video-frame JPGs (3,376), model checkpoints/weights
(8 files incl. 2.35 GB CLAP checkpoints), bulk arrays (`*.npy/*.npz`), `work/`
trees, `checkpoints/` directories, vendored `third_party/` trees, caches/venvs,
`.DS_Store`, the full local `migration/MIGRATION_MANIFEST.json` (3.0 MB, absolute
paths) and `migration/.git_ignored_list.txt` (7.6 MB). Verification: staged file list
cross-checked entry-by-entry against `PUBLICATION_MANIFEST.json` — exact match.

## Workflow policy

`RESEARCH_WORKFLOW.md` defines the long-term stage workflow
(experiment → validate → write report → commit → push → report SHA), one coherent
commit per completed stage, the publication invariants, and the authorization scope:
future agents may commit/push completed stages of `D:\Code\RhythmAlign-Research`
only; `D:\Code\RhythmAlign` requires separate product-release decisions.

## Stop conditions encountered

None triggered. The target GitHub repository did not exist before creation (verified
404, then created fresh); no force push was used; no ambiguous privacy decision
remained open at push time.
