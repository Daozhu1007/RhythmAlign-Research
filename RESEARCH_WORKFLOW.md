# Research Workflow

This document defines the default Git workflow for RhythmAlign-Research. It applies to
human researchers and to AI agents operating in this workspace.

## The default rule

A completed research stage follows:

```
experiment → validate → write report → commit → push → report commit SHA
```

Concretely:

1. **Run experiments locally.** Keep raw recordings, checkpoints, and large generated
   outputs untracked (the `.gitignore` covers the standard patterns: `work/`,
   `checkpoints/`, `*.wav`, `*.mp4`, `*.pt`, `*.npy`, `third_party/`, …). If a new
   artifact class appears, extend `.gitignore` in the same commit.
2. **Preserve the record.** Scripts, configs, metrics JSONs, decision JSONs, plots
   worth keeping, and the stage report stay in the stage directory. Large local
   artifacts that a reviewer needs to know about are referenced through a small
   manifest entry (path, role, size, SHA-256, generation script) rather than committed.
3. **Validate** the experiment (scripts run end-to-end, numbers reproduce, decision
   JSON matches the report).
4. **Review `git status` and `git diff`** before committing. Confirm nothing binary,
   private, or secret is staged (`git diff --cached --stat` should contain only the
   intended text/small-plot files).
5. **Commit** one coherent completed stage. Suggested message form:
   `s1w: <stage title> — <one-line verdict>`.
6. **Push** to `origin/main`.
7. **Report**: stage name, verdict, commit SHA, important deliverables, and whether
   large local artifacts were intentionally excluded (and which).

## Granularity

- **One coherent commit per completed research task/stage.** Do not commit every
  temporary debugging step; the history should read as a series of reviewable stages.
- Intermediate broken states live in the local workspace, not in the public history.
- A stage that fails validation does not get committed as "complete"; either finish it
  or commit it explicitly as a documented negative/superseded result — negative
  findings are legitimate research records here.

## Authorization

Future agents are **authorized to commit and push completed RhythmAlign-Research
stages** after validation, unless a task explicitly says otherwise. This authorization
applies **only** to `D:\Code\RhythmAlign-Research`.

It does **not** authorize committing or pushing to `D:\Code\RhythmAlign`. The product
repository follows separate product-release decisions.

## Publication invariants (never violated by a commit)

- No audio/video of any kind (`*.wav *.mp3 *.flac *.m4a *.mp4 *.mov *.mkv …`).
- No recordings of people, no copyrighted music, no handcam media.
- No model checkpoints or pretrained weights (`*.pt *.pth *.ckpt *.safetensors *.onnx`).
- No bulk numeric dumps (`*.npy *.npz`), no `work/`, no caches, no vendored
  third-party trees.
- No secrets, tokens, credentials, or `.env` files.
- New binary/plot exceptions: small generated scientific plots (PNG, ≲3 MB) and small
  manifests are fine; anything larger needs an explicit justification in the commit
  and an update to `PUBLICATION_MANIFEST.json` / `PUBLIC_REPO_AUDIT.md`.
