# Phase 1 - media audit of handcam + clean reference (all source media read-only).
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from r1_common import (HANCAM_MP4, REF_MP3, WORK_DIR, audio_stats, channel_relation,
                       decode_audio, load_wav, probe_streams, run_ffmpeg, save_json)

report = {}

print("=== probing containers ===")
report["handcam_container"] = probe_streams(HANCAM_MP4)
report["ref_container"] = probe_streams(REF_MP3)
for name in ("handcam_container", "ref_container"):
    c = report[name]
    print(f"--- {name}: duration={c['format_duration']:.3f}s start={c['format_start']} bitrate={c['format_bit_rate']}")
    for st in c["streams"]:
        print(f"    stream {st['index']} {st['type']}: {st['desc']}")
        if st["type"] == "Audio":
            print(f"      sr={st.get('sample_rate')} layout={st.get('layout')} bit_rate={st.get('bit_rate')}")

print("=== decoding full-band stereo float PCM ===")
hand_wav = os.path.join(WORK_DIR, "handcam_native.wav")
ref_wav = os.path.join(WORK_DIR, "ref_native.wav")
if not os.path.exists(hand_wav):
    decode_audio(HANCAM_MP4, hand_wav)  # native sr, stereo
if not os.path.exists(ref_wav):
    decode_audio(REF_MP3, ref_wav)      # native sr, stereo

y, sr_h = load_wav(hand_wav)
r, sr_r = load_wav(ref_wav)
print(f"handcam decoded: sr={sr_h} shape={y.shape} n={y.shape[0]} dur={y.shape[0]/sr_h:.3f}s")
print(f"ref      decoded: sr={sr_r} shape={r.shape} n={r.shape[0]} dur={r.shape[0]/sr_r:.3f}s")

report["handcam_audio"] = {
    "sample_rate": sr_h, "samples": y.shape[0], "duration_s": y.shape[0] / sr_h,
    "channels": y.shape[1], "stats": audio_stats(y), "lr": channel_relation(y),
}
report["ref_audio"] = {
    "sample_rate": sr_r, "samples": r.shape[0], "duration_s": r.shape[0] / sr_r,
    "channels": r.shape[1], "stats": audio_stats(r), "lr": channel_relation(r),
}
for name, st in (("handcam", report["handcam_audio"]), ("ref", report["ref_audio"])):
    print(f"{name}: {st['stats']}")
    print(f"{name} L/R: {st['lr']}")

# Clipping in a stricter sense: consecutive runs at fullscale (>= 0.9995)
for key, x in (("handcam", y), ("ref", r)):
    full = np.abs(x) >= 0.9995
    runs = np.diff(np.concatenate([[0], full.any(axis=1).astype(np.int8), [0]]))
    starts = np.where(runs == 1)[0]
    ends = np.where(runs == -1)[0]
    n_runs = len(starts)
    longest = int(np.max(ends - starts)) if n_runs else 0
    report[f"{key}_audio"]["fullscale_runs"] = {"count": n_runs, "longest_samples": longest}
    print(f"{key} fullscale runs: count={n_runs} longest={longest} samples")

save_json(os.path.join(WORK_DIR, "audit_report.json"), report)
print("saved audit_report.json")
