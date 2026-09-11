# Public Repository Audit — RhythmAlign-Research

Date: 2026-09-12
Scope: pre-publication audit of `D:\Code\RhythmAlign-Research` before initializing the
public GitHub repository `Daozhu1007/RhythmAlign-Research`. Read-only; no Git state was
created while producing this document. The product repository
(`D:\Code\RhythmAlign`, HEAD `44164d1a`, upstream-identical, clean working tree) was
verified first and was not modified.

## 1. Totals

| Item | Value |
|---|---|
| Total size | 9.7 GB |
| Total files | 5,608 (5,601 migrated + migration reports/README added 2026-09-12) |
| Publishable proposal (after Part-6 ignore rules) | 397 files / ~50 MB |
| Excluded local bulk | ~9.65 GB |

## 2. Largest directories

| Directory | Size | Contents |
|---|---|---|
| `experiments/r2_target_separation` | 3.5 GB | `checkpoints/` 3.4 GB (AudioSep weights), `.venv`-less scripts/logs/outputs |
| `experiments/s1e_existing_corpus_feasibility` | 2.5 GB | `work/` 1.3 GB (incl. SoloAudio ckpt copies), `third_party/SoloAudio` 1.2 GB (clone + pretrained weights), clips, listening pack |
| `experiments/r3_audio_query` | 2.5 GB | `checkpoints/` 2.4 GB (CLAP + CLAPSep best_model), queries, outputs |
| `experiments/r1_golden_sample` | 253 MB | `work/` 169 MB, outputs (WAV/MP4), scripts |
| `experiments/s1a_ground_truth_pilot` | 244 MB | `work/` 155 MB, `fixtures/` 77 MB (46 synthetic WAVs) |
| `experiments/r56_precision_cleanup` | 203 MB | outputs (WAV/MP4/PNG), work |
| `experiments/r55_burst_event_splitting` | 191 MB | outputs, work (frame JPG dumps) |
| `experiments/r4_contact_reconstruction` | 174 MB | `work/` 103 MB, `frames/` 31 MB (901 JPGs), outputs |
| `experiments/r5_auto_contact_detection` | 160 MB | outputs, work (1,081 frame JPGs) |
| `experiments/r5_fix_validity_repair` | 136 MB | outputs, work, listening pack |
| `experiments/r45_conservative_windows` | 67 MB | outputs, work |
| `migration/` | 11 MB | manifest 3.0 MB, ignored-list 7.6 MB, 3 reports, verify script |
| `experiments/s0_research_reframing` | 216 KB | reports + decision JSONs + validators |
| `experiments/independent_review_r0_r56` | 208 KB | review report + evidence manifest + 1 plot |

## 3. Largest files

| File | Size |
|---|---|
| `experiments/r3_audio_query/checkpoints/music_audioset_epoch_15_esc_90.14.pt` | 2.35 GB |
| `experiments/r2_target_separation/checkpoints/music_speech_audioset_epoch_15_esc_89.98.pt` | 2.35 GB |
| `experiments/r2_target_separation/checkpoints/audiosep_base_4M_steps.ckpt` | 1.26 GB |
| `experiments/s1e.../work/soloaudio_ckpt/soloaudio_v2.pt` (+ identical copy in `third_party/SoloAudio/pretrained_models/`) | 592 MB ×2 |
| `experiments/s1e.../work/soloaudio_ckpt/audio-vae.pt` (+ identical copy in `third_party/SoloAudio/pretrained_models/`) | 549 MB ×2 |
| `experiments/r3_audio_query/checkpoints/best_model.ckpt` | 178 MB |
| `experiments/s1e.../work/handcam_native.wav` (+ r1 copy) | 63 MB ×2 |
| `experiments/s1e.../work/ref_native.wav` (+ r1 copy) | 54 MB ×2 |
| `holdoutD_video.mp4` / `holdoutC_video.mp4` / `holdout_video.mp4` / `golden_video.mp4` | 31/28/26/24 MB |

All are checkpoints, pretrained weights, recordings, or rendered video — every one is
excluded from publication by the ignore rules of Part 6.

## 4. File-extension census (whole workspace)

| Ext | Count | Disposition |
|---|---|---|
| .jpg | 3,376 | video-frame dumps (`frames/`, `work/ho_frames`, `work/hc_frames`) — excluded |
| .py | 963 | research scripts (~133 publishable) + ~830 vendored third_party — third_party excluded |
| .wav | 389 | recordings/outputs/fixtures — excluded |
| .json | 250 | metrics, decisions, manifests (~130 publishable; full migration manifest kept untracked) |
| .png | 182 | diagnostic/spectrogram plots (173 outside third_party; 74 publishable-size set) |
| .mdx | 124 | vendored diffusers docs (SoloAudio tree) — excluded |
| .md | 90 | reports/READMEs (~43 publishable; rest vendored) |
| .npy/.npz | 67 | bulk numeric arrays, all inside `work/` — excluded |
| .txt/.log/.csv/.yml/.yaml/.toml/.cfg/.ini/.sh/.config/.in | ~90 | mixed: vendored configs excluded; research logs/notes publishable |
| .pt/.ckpt | 8 | model checkpoints — excluded |
| .mp4 | 4 | rendered videos — excluded |
| .ipynb/.sample/.gz/.rev/.idx/.pack/.typed etc. | ~30 | vendored third_party / SoloAudio `.git` objects — excluded |
| .DS_Store | 13 | OS junk — excluded |

## 5. Existing experiment structure

13 experiment directories under `experiments/`, each typically
`scripts/ + outputs/ + work/ + logs/ (+ third_party/, listening_pack/, checkpoints/)`,
with top-level `REPORT.md` / decision JSONs. R1–R5.6 are the historical
cancellation/reconstruction line; `independent_review_r0_r56` audits it; S0 is the
research reframing; S1a the ground-truth pilot (synthetic fixtures only, real capture
pending); S1E the existing-corpus feasibility study (decision: B — adaptation needed).

## 6. Third-party trees (see also LICENSE_STATUS.md)

| Tree | Size | Upstream | License | Local modification | Publish? |
|---|---|---|---|---|---|
| `r2_target_separation/third_party/AudioSep` | 5.5 MB | github.com/Audio-AGC/AudioSep | MIT (© Xubo Liu) | none found; plus 2 downloaded checkpoints in `checkpoints/` | No — document + link instead |
| `r3_audio_query/third_party/CLAPSep` | 32 KB | CLAPSep (Waveformer-derived; © Hao Ma @SDU per file headers) | none present locally — unclear | unknown (3 model files copied during R3) | No — unclear licensing; documented by reference |
| `s1e_existing_corpus_feasibility/third_party/SoloAudio` | 1.2 GB | github.com/WangHelin1997/SoloAudio | MIT (© 2024 Helin Wang) | none found; contains its own `.git` and `pretrained_models/` | No — document + link; checkpoints not redistributed |

## 7. Audio / video / recording artifacts (all excluded)

389 WAVs (raw handcam extractions, reference song audio, separation outputs, listening
packs, synthetic fixtures incl. `nuisance_speech.wav`), 4 MP4s (handcam renders),
3,376 frame JPGs, r3 `queries/*.wav` (copyrighted-song excerpts used as queries).
No audio or video file is proposed for publication.

## 8. Caches / environments / generated bulk

No `.venv/`, `__pycache__/`, or `.pytest_cache/` exists (deliberately not migrated;
see `migration/MIGRATION_NOTES.md`). Bulk generated content lives in `work/`,
`checkpoints/`, `frames/`, and media extensions — all covered by ignore rules.
13 `.DS_Store` junk files excluded.

## 9. Migration metadata

- `migration/MIGRATION_MANIFEST.json` (3.0 MB, per-file absolute paths + SHA-256 for
  all 5,601 migrated files) — kept **untracked**; a sanitized
  `migration/MIGRATION_MANIFEST.public.json` (relative paths + hashes only) is published.
- `migration/.git_ignored_list.txt` (7.6 MB inventory of product-repo ignored files) —
  local-only, untracked.
- `migration/MIGRATION_NOTES.md`, `REPO_HYGIENE_AUDIT.md`, `REPO_HYGIENE_REPORT.md`,
  `verify_copy.py` — publishable (process provenance).

## 10. Secrets / privacy scan

Method: pattern scan of every non-vendored text file (`.py .md .json .txt .yml .yaml
.toml .cfg .ini .sh .csv .log`) for API-key shapes (`sk-ant-`, `sk-…{40,}`, `ghp_…`,
`github_pat_`, `xox…`, `AKIA…`, PEM blocks, `Authorization:` headers, `api_key=…`),
credential files (`.env*`, `*.key`, `*.pem`, `credentials*`), personal e-mails, and
recording-metadata leaks (`location`, `com.android`, device identifiers).

Result: **no credentials, tokens, or key material found.** Apparent hits were false
positives (the word shape `task-specific` matching `sk-…`, the metric name
`hf_retention` matching `hf_…`, public arXiv/DCASE URLs).

Privacy-relevant findings:

1. `experiments/r1_golden_sample/work/audit_report.json` embeds raw phone metadata from
   the source recording: **GPS coordinates** (`+28.0206+120.7055`), phone model, and
   copyrighted-song metadata. It sits in `work/` and is excluded from publication.
   No other file contains these strings. The file stays local, unredacted.
2. 14 text files reference local media paths (`D:\Daozh\Videos\舞萌手元\…` — private
   handcam recordings and copyrighted songs). Per the publication policy these absolute
   paths remain in historical research records for provenance; no audio content is
   published. The referenced song is copyrighted; only its filename appears.
3. No recordings of third parties exist outside synthetic fixtures; the S1a fixture
   speech nuisance WAVs are synthetic-mixture fixtures and stay local regardless.

## 11. Conclusion

A publishable research tree of 397 files / ~50 MB (code, reports, JSON metrics and
decisions, logs, small PNG plots, migration reports, sanitized manifest) can be staged
safely. All checkpoints, recordings, video, frame dumps, vendored third-party trees,
caches, and the full local migration manifest are excluded. Publication may proceed per
Parts 6–13 of the bootstrap task.
