# S1W step 11 - FROZEN PRIMARY REAL TEST (protocol sections 12/33/38/39).
# Runs ONLY after training and checkpoint selection are completely frozen.
# 6 product-relevant S1E challenge groups x exactly 3 methods:
#   RAW (original sealed recording, 48k stereo), ZERO-SHOT CLAPSep (aq Q1),
#   ADAPTED CLAPSep (aq Q1, selected checkpoint).
# Fixed gain: no normalization of method outputs; resample only for pack material.
# Also builds the small LOCAL secondary product remix (aligned pristine + stem)
# with identical pristine gain across methods; song-active interval enforced.
import os
import sys
import json
import numpy as np
import soundfile as sf
import librosa
import torch

S1W = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(S1W, "scripts"))
import clapsep_train_lib as ctl  # noqa: E402
import clapsep_lib  # noqa: E402

SEALED_SRC = r"D:\Daozh\Videos\舞萌手元\13.2\共感觉\AP\共感怪物AP.mp4"
PRISTINE = r"D:\Daozh\Videos\舞萌手元\13.2\共感觉\共感觉.mp3"
D_STAR = -11.3747  # R1 alignment (video = ref + d*); drift negligible per R1
SONG_ACTIVE = (11.4, 150.9)  # reference coverage on video clock (S1E CHALLENGE_CLIPS)

GROUPS = [
    ("c1_speech_npc_a", 86.0, 94.0),
    ("c2_speech_npc_b", 96.0, 104.0),
    ("c3_taiko_prompt", 126.5, 131.5),
    ("c6_dense_golden", 22.5, 37.5),
    ("c7_weak_taps", 38.0, 44.0),
    ("c8_slide_friction", 51.5, 57.5),
]

OUT = os.path.join(S1W, "outputs", "audio")
LOG = os.path.join(S1W, "logs")


def extract_audio(ff, path, sr, mono, dst):
    import subprocess
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
           "-i", path, "-vn", "-ar", str(sr)]
    cmd += ["-ac", "1"] if mono else ["-ac", "2"]
    cmd += ["-c:a", "pcm_f32le", dst]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-300:]


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


def load_model_with(ckpt_path, device):
    model = ctl.load_model(device)
    ctl.set_trainable_scope(model)
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"], strict=True)
    return model


CHUNK = 320000  # exact 10 s @ 32 kHz — the trained chunk size


def chunked_ola_separate(model, mix32, start, end, emb, zeros, device):
    """Sliding-window separation with EXACT 10-s chunks (0.5 hann overlap-add).

    Zero-padding arbitrary-length clips to 10-s multiples (the S1E clip protocol)
    flips the adapted masker to passthrough (out-of-distribution silent tail
    dominating the global masker attention) — recorded as a finding. Exact-chunk
    OLA inference is the trained distribution and is applied IDENTICALLY to the
    zero-shot and adapted models (zero-shot behavior on exact chunks equals its
    official protocol; parity was verified chunk-exactly).
    """
    hop = CHUNK // 2
    # expand the region so every processed chunk is full-length content (no zero pad)
    s = max(0, start)
    n = end - start
    total = ((n + hop - 1) // hop) * hop + hop
    e = min(len(mix32), s + total)
    # ensure coverage of [start,end)
    while e - s < n:
        s = max(0, s - hop)
    region = mix32[s:e]
    window = np.hanning(CHUNK)
    acc = np.zeros(len(region), dtype=np.float64)
    wsum = np.zeros(len(region), dtype=np.float64)
    for i in range(0, len(region) - CHUNK + 1, hop):
        chunk = region[i:i + CHUNK]
        y = clapsep_lib.separate(model, chunk, emb, zeros, device)
        acc[i:i + CHUNK] += y * window
        wsum[i:i + CHUNK] += window
    wsum[wsum < 1e-6] = 1.0
    out = (acc / wsum).astype(np.float32)
    return out[(start - s):(start - s) + n]


def main():
    import imageio_ffmpeg
    device = torch.device("cuda")
    best = os.path.join(S1W, "checkpoints", "best.ckpt")
    assert os.path.exists(best), "frozen checkpoint missing — run 09/selection first"

    os.makedirs(OUT, exist_ok=True)
    ff = imageio_ffmpeg.get_ffmpeg_exe()

    wav48 = os.path.join(S1W, "work", "audio", "sealed_48k_stereo.wav")
    if not os.path.exists(wav48):
        extract_audio(ff, SEALED_SRC, 48000, False, wav48)
    mix48, sr48 = sf.read(wav48, dtype="float32")
    mix32 = librosa.resample(mix48.mean(axis=1), orig_sr=48000, target_sr=32000)

    emb = None
    zs_model = ad_model = None
    results = []
    for gid, t0, t1 in GROUPS:
        a, b = int(t0 * 48000), int(t1 * 48000)
        raw48 = mix48[a:b]
        sf.write(os.path.join(OUT, f"raw_{gid}_48k.wav"), raw48, 48000, subtype="FLOAT")
        raw32 = librosa.resample(raw48.mean(axis=1), orig_sr=48000, target_sr=32000)

        outs = {"raw": raw48}
        c0, c1 = int(t0 * 32000), int(t1 * 32000)
        if zs_model is None:
            zs_model = load_model_with(None, device)
            emb = ctl.embed_q1(zs_model)
        target_zs = chunked_ola_separate(zs_model, mix32, c0, c1, emb,
                                         np.zeros((1, 512), np.float32), device)
        torch.cuda.empty_cache()
        if ad_model is None:
            ad_model = load_model_with(best, device)
        target_ad = chunked_ola_separate(ad_model, mix32, c0, c1, emb,
                                         np.zeros((1, 512), np.float32), device)
        torch.cuda.empty_cache()

        for tag, x32 in (("zeroshot", target_zs), ("adapted", target_ad)):
            sf.write(os.path.join(OUT, f"clapsep_{tag}_{gid}_32k.wav"), x32, 32000, subtype="FLOAT")
            ups = librosa.resample(x32, orig_sr=32000, target_sr=48000)
            outs[tag] = np.stack([ups, ups], axis=1)

        # fixed-gain descriptive diagnostics (click band 2-9k, mid 150-2k)
        freqs = np.fft.rfftfreq(2048, 1 / 48000)
        click = (freqs >= 2000) & (freqs <= 9000)
        mid = (freqs >= 150) & (freqs <= 2000)

        def band_db(x):
            S = np.abs(librosa.stft(x.mean(axis=1), n_fft=2048, hop_length=512)) ** 2
            return (20 * np.log10(np.sqrt(S[click].mean(axis=0)) + 1e-12),
                    20 * np.log10(np.sqrt(S[mid].mean(axis=0)) + 1e-12))
        raw_c, raw_m = band_db(raw48)
        entry = {"group": gid, "t": [t0, t1], "rms_db": {k: db(v.mean(axis=1)) for k, v in outs.items()}}
        for tag in ("zeroshot", "adapted"):
            c, m = band_db(outs[tag])
            entry[f"{tag}_click_db"] = round(float(np.median(c - raw_c)), 2)
            entry[f"{tag}_mid_db"] = round(float(np.median(m - raw_m)), 2)
        results.append(entry)
        print(json.dumps(entry), flush=True)

    # secondary LOCAL product remix: pristine (identical gain) + estimated stem,
    # inside the song-active interval; outside it, bypass original audio.
    ref, _ = librosa.load(PRISTINE, sr=48000, mono=False)
    if ref.ndim == 1:
        ref = np.stack([ref, ref])
    pristine_gain = 10 ** (-6 / 20)  # fixed -6 dBFS-ish pristine level, identical all methods
    n = len(mix48)
    t_vec = np.arange(n) / 48000
    in_song = (t_vec >= SONG_ACTIVE[0]) & (t_vec <= SONG_ACTIVE[1])
    ref_at = np.zeros_like(mix48)
    # reference placed at d*: video time t plays ref sample (t - d*)
    start = int((SONG_ACTIVE[0] - D_STAR) * 48000)
    L = min(n - start, len(ref[0]))
    ref_at[start:start + L] = (pristine_gain * ref[:, :L]).T
    for gid, t0, t1 in GROUPS:
        a, b = int(t0 * 48000), int(t1 * 48000)
        for tag in ("raw", "zeroshot", "adapted"):
            if tag == "raw":
                stem, _ = sf.read(os.path.join(OUT, f"raw_{gid}_48k.wav"), dtype="float32")
            else:
                x32, _ = sf.read(os.path.join(OUT, f"clapsep_{tag}_{gid}_32k.wav"), dtype="float32")
                up = librosa.resample(x32, orig_sr=32000, target_sr=48000)
                stem = np.stack([up, up], axis=1)
            remix = stem.copy()
            mask = in_song[a:b]
            remix[mask] = 0.4 * stem[mask] + ref_at[a:b][mask]  # stem kept at fixed share
            sf.write(os.path.join(S1W, "work", "audio", "remix", f"remix_{tag}_{gid}.wav"),
                     remix, 48000, subtype="FLOAT")

    with open(os.path.join(LOG, "11_real_test.json"), "w", encoding="utf-8") as f:
        json.dump({"methods": ["raw", "zeroshot", "adapted"], "checkpoint": best,
                   "fixed_gain": "no normalization; 48k stereo masters + 32k mono model domain",
                   "results": results}, f, indent=1)
    print("real test done")


if __name__ == "__main__":
    main()
