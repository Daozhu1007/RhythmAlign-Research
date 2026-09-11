# Extract golden_raw.wav (sample-exact) and golden_video.mp4 (frame-accurate re-encode),
# then verify that the video's audio timeline matches golden_raw.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import signal as sig

from r1_common import (HANCAM_MP4, OUT_DIR, WORK_DIR, decode_audio, load_wav, load_json,
                       run_ffmpeg, save_json, save_wav, audio_stats)

sel = load_json(os.path.join(WORK_DIR, "golden_selection.json"))
start, dur = sel["start"], sel["duration"]
SR = 48000
s0 = int(round(start * SR))
s1 = int(round((start + dur) * SR))
ctx = int(round(3.0 * SR))  # +-3 s processing context, kept inside the decoded track portion

y, sr = load_wav(os.path.join(WORK_DIR, "handcam_native.wav"))
assert sr == SR
golden = y[s0:s1]
save_wav(os.path.join(OUT_DIR, "golden_raw.wav"), golden.astype(np.float32), SR)
save_wav(os.path.join(WORK_DIR, "golden_ctx.wav"), y[s0 - ctx:s1 + ctx].astype(np.float32), SR)
st = audio_stats(golden)
print(f"golden_raw.wav: {golden.shape[0]/SR:.3f}s stats={st}")

vid = os.path.join(OUT_DIR, "golden_video.mp4")
run_ffmpeg(["-ss", f"{start}", "-i", HANCAM_MP4, "-t", f"{dur}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "256k", vid])
print("golden_video.mp4 written")

# A/V check: decode the video's audio, cross-correlate its first 3 s against golden_raw.
tmp = os.path.join(WORK_DIR, "golden_video_audio.wav")
decode_audio(vid, tmp)
a, asr = load_wav(tmp)
b, _ = load_wav(os.path.join(OUT_DIR, "golden_raw.wav"))
n = int(3 * asr)
aa = a[:n].mean(axis=1)
bb = b[:n].mean(axis=1)
c = sig.correlate(bb, aa, mode="full", method="fft")
lag = np.argmax(c) - (n - 1)
print(f"A/V check: video-audio leads golden_raw by {lag/asr*1000:.2f} ms (should be ~0)")
save_json(os.path.join(WORK_DIR, "golden_extract.json"),
          {"start": start, "duration": dur, "samples": golden.shape[0],
           "video_av_offset_ms": lag / asr * 1000, "stats": st})
