# Coarse alignment using the CURRENT RhythmAlign analysis path (auto_sync._align_hybrid),
# on its standard 22050 Hz mono analysis copies, exactly like production.
# Also runs the onset fallback for cross-checking.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r"D:\Code\RhythmAlign")
import numpy as np
import librosa

import auto_sync  # production repo module, used read-only
from r1_common import HANCAM_MP4, REF_MP3, WORK_DIR, decode_audio, save_json

MONO_DIR = os.path.join(WORK_DIR, "mono_22050")
os.makedirs(MONO_DIR, exist_ok=True)
SR = 22050

y_video, _ = librosa.load(decode_audio(HANCAM_MP4, os.path.join(MONO_DIR, "handcam.wav"), sr=SR, channels=1), sr=None, mono=True)
y_music, _ = librosa.load(decode_audio(REF_MP3, os.path.join(MONO_DIR, "ref.wav"), sr=SR, channels=1), sr=None, mono=True)
print(f"mono analysis copies: handcam {len(y_video)/SR:.1f}s, ref {len(y_music)/SR:.1f}s")

offset, z_hybrid, _ = auto_sync._align_hybrid(y_video, y_music, SR, 512)
offset_onset, z_onset, _ = auto_sync._align_onset(y_video, y_music, SR, 512)
offset_chroma, z_chroma, _ = auto_sync._align_chroma(y_video, y_music, SR, 512)

# Convention check (from correlate(music, video)): returned offset = t_handcam - t_ref,
# i.e. at handcam time t, the reference time is t - offset.
out = {
    "convention": "returned offset = t_handcam - t_ref (seconds); t_ref = t_handcam - offset",
    "hybrid": {"offset": offset, "z_score": z_hybrid},
    "onset": {"offset": offset_onset, "z_score": z_onset},
    "chroma": {"offset": offset_chroma, "z_score": z_chroma},
    "note": f"handcam audio spans ref[{offset:.2f} .. {offset + len(y_video)/SR:.2f}]s if offset is correct",
}
for k, v in out.items():
    if isinstance(v, dict):
        print(k, {kk: round(vv, 4) for kk, vv in v.items()})
    else:
        print(k, v)
save_json(os.path.join(WORK_DIR, "coarse_offset.json"), out)
