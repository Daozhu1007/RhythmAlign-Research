# raw/ — where future S1a recordings go

Create one sub-folder per session, named `<session_id>` (suggested form:
`YYYYMMDD_venue01`), and place files **unchanged** — no trimming, no conversion, no
renaming beyond the layout below.

```
experiments/s1a_ground_truth_pilot/raw/
└── <session_id>/
    ├── phone/                  REQUIRED — handcam-position device
    │   ├── phone_audio.<ext>   audio from the phone/camera (any format it natively produces)
    │   └── phone_video.<ext>   the synchronized video (if audio+video are one file,
    │                           put that single file here; ingest will note it)
    ├── aux/                    strongly preferred / optional channels
    │   ├── closemic.<ext>      close air mic near the interaction area (if available)
    │   ├── contact.<ext>       contact/piezo channel (if available; evidence only)
    │   └── linear.<ext>        separate linear recorder at phone position (if available)
    ├── pristine/               pristine music file(s) (only if known/available)
    │   └── <track>.<ext>
    ├── anchors.csv             sync anchor times (template below)
    └── session_notes.txt       filled template below
```

Only `phone/` is strictly required; the pipeline must tolerate missing auxiliary
channels. Do not delete or "clean up" anything.

## anchors.csv template

```csv
channel,event,approx_time_s,notes
phone,start_clap,,after countdown 3-2-1
phone,start_tap,,firm screen tap right after clap
phone,end_clap,,
phone,end_tap,,
aux_closemic,start_clap,,
aux_closemic,end_clap,,
```

`approx_time_s` may be left empty — the ingest tool locates onsets inside search windows
around these rows using the same audited routine on every channel; fill it (even roughly,
±5 s) to speed up and de-risk the search.

## session_notes.txt template

```
session_id      :
date / time     :
location        :
operator permission noted (Y/N; what for):
phone device / model       :
phone recording app + settings (codec, rate, AGC off?):
phone position (where relative to screen/buttons):
aux devices (model, rate, format; else "none"):
approx phone<->machine distance (m):
approx aux<->machine distance (m):
cabinet music state during quiet block (muted? lowest volume? attract loop?):
anything unusual (interruptions, restarts, external noise):
```

## What happens next (automated)

1. `scripts/01_media_inventory.py` — inventory, rates/channels/durations, hashes.
2. `scripts/02_extract_pcm.py` — lossless working PCM extraction (float32; native-rate
   archive untouched; 32 kHz working copies for the CLAPSep domain, documented resampler).
3. `scripts/03_qc_audio.py` — clipping, rest floor, RMS/spectrum reports, truth flags.
4. `scripts/04_sync_fit.py` — clock map `t_T = a + b·t_X`, residuals, validation.
5. Manifest + QC report update; candidate T1 segments proposed for human confirmation.
