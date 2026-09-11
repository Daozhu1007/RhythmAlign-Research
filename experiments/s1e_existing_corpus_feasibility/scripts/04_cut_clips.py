# S1E step 04 - cut challenge clips (48k stereo masters + 32k mono model copies).
# Clip list fixed a priori from 01/02/03 diagnostics + owner-named regions. No listening
# was used for selection (selection is diagnostics + owner guidance only).
import os
import sys
import hashlib
import json

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import signal as sig

from s1e_common import WORK_DIR, CLIP_DIR, SR, load_wav, save_wav, save_json, mmss
import resampy

CLIPS = [
    # id, t0, t1, ref_covered, challenge tags
    ("c1_speech_npc_a",   86.0,  94.0, True,  "npc_announcer_speech+interaction"),
    ("c2_speech_npc_b",   96.0, 104.0, True,  "npc_announcer_speech+interaction"),
    ("c3_taiko_prompt",  126.5, 131.5, True,  "loud_taiko_system_prompt(2:08-2:09)+interaction"),
    ("c4_ambience_start",  2.5,   8.5, False, "pre_track_ambience_only"),
    ("c5_announcer_end", 152.0, 159.0, False, "post_song_announcer_speech_no_interaction"),
    ("c6_dense_golden",   22.5,  37.5, True,  "dense_interaction+music (r1/r2/r3 golden window)"),
    ("c7_weak_taps",      38.0,  44.0, True,  "weak_button_taps+music"),
    ("c8_slide_friction", 51.5,  57.5, True,  "slide_friction+ordinary_interaction"),
]


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    y, _ = load_wav(os.path.join(WORK_DIR, "handcam_native.wav"))
    manifest = {"source": "D:\\Daozh\\Videos\\舞萌手元\\13.2\\共感觉\\AP\\共感怪物AP.mp4",
                "source_sha256_note": "413,372,548 bytes; not hashed (large); see RUN_MANIFEST",
                "sr": SR, "clips": []}
    for cid, t0, t1, refc, tags in CLIPS:
        i0, i1 = int(round(t0 * SR)), int(round(t1 * SR))
        seg = y[i0:i1]
        p48 = os.path.join(CLIP_DIR, "audio", f"{cid}_48k_stereo.wav")
        save_wav(p48, seg.astype(np.float32), SR)
        y16k = resampy.resample(np.mean(seg, axis=1).astype(np.float32), SR, 32000)
        p32 = os.path.join(CLIP_DIR, "audio", f"{cid}_32k_mono.wav")
        save_wav(p32, y16k.astype(np.float32), 32000)
        manifest["clips"].append({
            "id": cid, "t_start_video_s": t0, "t_end_video_s": t1,
            "t_start_mmss": mmss(t0), "t_end_mmss": mmss(t1),
            "duration_s": t1 - t0, "reference_covered": refc,
            "challenge_tags": tags,
            "file_48k": os.path.relpath(p48, CLIP_DIR), "sha256_48k": sha256(p48),
            "file_32k": os.path.relpath(p32, CLIP_DIR), "sha256_32k": sha256(p32),
            "rms_dbfs_48k": float(10 * np.log10(np.mean(seg ** 2) + 1e-12)),
            "peak_dbfs_48k": float(20 * np.log10(np.max(np.abs(seg)) + 1e-12)),
        })
        print(f"{cid}: {t1-t0:.1f}s [{mmss(t0)}-{mmss(t1)}] rms {manifest['clips'][-1]['rms_dbfs_48k']:.1f} dBFS")
    save_json(os.path.join(CLIP_DIR, "clip_manifest.json"), manifest)
    print("saved clip_manifest.json")


if __name__ == "__main__":
    main()
