# S1R step 07 - SEALED PRIMARY CONTINUITY TEST (protocol section 36).
#
# The SAME six product-relevant S1E/S1W groups, SAME sealed source recording,
# exactly three methods: RAW / ZERO-SHOT CLAPSep / S1R STUDENT.
# The S1W adapted checkpoint is a known failed model and is NOT included.
# Runs only after checkpoint+protocol freeze. Fixed gain: no normalization of
# method outputs; descriptive proxies only (no stems exist).
import os
import sys
import json
import numpy as np
import soundfile as sf
import torch

S1R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(S1R, "scripts"))
import s1r_common as rc  # noqa: E402
import s1r_model as sm  # noqa: E402
import s1r_infer as si  # noqa: E402

SEALED_SRC = r"D:\Daozh\Videos\舞萌手元\13.2\共感觉\AP\共感怪物AP.mp4"
D_STAR = -11.3747          # R1 alignment (video = ref + d*); drift negligible per R1
SONG_ACTIVE = (11.4, 150.9)

GROUPS = [
    ("c1_speech_npc_a", 86.0, 94.0),
    ("c2_speech_npc_b", 96.0, 104.0),
    ("c3_taiko_prompt", 126.5, 131.5),
    ("c6_dense_golden", 22.5, 37.5),
    ("c7_weak_taps", 38.0, 44.0),
    ("c8_slide_friction", 51.5, 57.5),
]

OUT = os.path.join(S1R, "work", "audio", "sealed")
CKPT = os.path.join(S1R, "checkpoints", "s1r_selected.ckpt")


def extract_audio(path, sr, mono, dst):
    import subprocess
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
           "-ss", str(D_STAR), "-i", path, "-vn", "-ar", str(sr)]
    cmd += ["-ac", "1"] if mono else ["-ac", "2"]
    cmd += ["-c:a", "pcm_f32le", dst]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-300:]


def band_db(x, lo, hi, sr=rc.SR):
    import librosa

    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    band = S[(freqs >= lo) & (freqs <= hi)].sum(axis=0)
    return round(float(10 * np.log10(band.mean() + 1e-12)), 3)


def main():
    import clapsep_lib  # noqa: E402

    os.makedirs(OUT, exist_ok=True)
    devicelike = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wav32 = os.path.join(OUT, "sealed_32k_mono.wav")
    if not os.path.exists(wav32):
        extract_audio(SEALED_SRC, rc.SR, True, wav32)
    mix, sr = sf.read(wav32, dtype="float32")
    assert sr == rc.SR

    student = sm.load_student_from_zero_shot(devicelike, CKPT)
    zeroshot = clapsep_lib.load_clapsep(devicelike)
    emb = clapsep_lib.embed_audio_query(zeroshot, rc.Q1_WAV)
    zeros = np.zeros((1, 512), dtype=np.float32)

    rows = []
    for gid, t0, t1 in GROUPS:
        a, b = int(t0 * rc.SR), int(t1 * rc.SR)
        raw = mix[a:b]
        zs = si.ola_separate(zeroshot, mix, a, b, emb, zeros, devicelike, student=False)
        st = si.ola_separate(student, mix, a, b, emb, zeros, devicelike, student=True)
        sf.write(os.path.join(OUT, f"{gid}_raw.wav"), raw, rc.SR)
        sf.write(os.path.join(OUT, f"{gid}_zeroshot.wav"), zs, rc.SR)
        sf.write(os.path.join(OUT, f"{gid}_s1r.wav"), st, rc.SR)

        def feats(y):
            return {
                "rms_db": round(rc.rms_db(y), 3),
                "click_2k_9k_db": band_db(y, 2000, 9000),
                "mid_150_2k_db": band_db(y, 150, 2000),
                "flat_frac": round(rc.flat_frac(y), 4),
            }

        f_raw, f_zs, f_st = feats(raw), feats(zs), feats(st)
        row = {"group": gid, "t": [t0, t1], "raw": f_raw, "zeroshot": f_zs, "s1r": f_st,
               "zs_click_ret_vs_raw": round(f_zs["click_2k_9k_db"] - f_raw["click_2k_9k_db"], 3),
               "s1r_click_ret_vs_raw": round(f_st["click_2k_9k_db"] - f_raw["click_2k_9k_db"], 3),
               "s1r_minus_zeroshot_rms_db": round(f_st["rms_db"] - f_zs["rms_db"], 3)}
        rows.append(row)
        print(json.dumps(row), flush=True)

    rc.save_json(os.path.join(S1R, "logs", "07_sealed_test.json"), {
        "methods": ["RAW", "ZERO-SHOT", "S1R"],
        "excluded_by_design": "S1W adapted checkpoint (known failed model, protocol section 36)",
        "protocol": "exact 10-s chunk overlap-add, identical for zero-shot and S1R",
        "rows": rows})
    print("SEALED TEST DONE")


if __name__ == "__main__":
    main()
