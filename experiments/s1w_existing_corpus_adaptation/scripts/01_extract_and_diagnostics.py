# S1W step 01 - working audio (32 kHz mono) for every deduplicated RAW candidate
# + descriptive per-recording diagnostics used ONLY to stratify the split
# (diversity selection). No target labels are produced here.
import os
import re
import json
import subprocess
import numpy as np

S1W = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PRIVATE = os.path.join(S1W, "work", "private")
AUD = os.path.join(S1W, "work", "audio", "raw32k")

DUPLICATE_EXCLUDE = {r"DATASET\海底谭DATASET\海底谭0.mp4": "byte-identical duplicate of 已发/海底谭/海底谭.mp4 (sha256 6f49020b…)"}


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract32k(ff, src, dst):
    if os.path.exists(dst):
        return False
    tmp = dst + ".part.wav"
    p = subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                        "-vn", "-ac", "1", "-ar", "32000", "-c:a", "pcm_f32le", tmp],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src}: {p.stderr[-300:]}")
    os.replace(tmp, dst)
    return True


def diagnostics(wav_path, vad_model):
    import librosa
    import soundfile as sf
    import torch
    x, sr = sf.read(wav_path, dtype="float32")
    assert sr == 32000
    dur = len(x) / sr
    clip_frac = float(np.mean(np.abs(x) > 0.98))

    # 50 ms frame levels
    hop = 1600
    n = len(x) // hop
    fr = x[:n * hop].reshape(n, hop)
    fr_rms = np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-12
    fr_db = 20 * np.log10(fr_rms)
    rms_db = float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))
    floor_db = float(np.percentile(fr_db, 10))

    # click band 2-9 kHz onset diagnostics (descriptive only)
    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=320)) ** 2
    freqs = librosa.fft_frequencies(sr=32000, n_fft=1024)
    band = (freqs >= 2000) & (freqs <= 9000)
    sb = np.sqrt(S[band].sum(axis=0) + 1e-12)
    sb_db = 20 * np.log10(sb + 1e-12)
    flux = np.maximum(np.diff(sb_db, prepend=sb_db[0]), 0.0)
    thr = np.median(flux) + 6.0
    peaks = flux > np.maximum(thr, 3.0)
    onset_rate = float(peaks.sum() / dur)
    onset_level_db = float(np.median(sb_db[peaks])) if peaks.sum() > 0 else None

    # sustained friction proxy: flatness in 1-6 kHz on stationary-ish frames
    band16 = (freqs >= 1000) & (freqs <= 6000)
    Sb = S[band16] + 1e-12
    flat = np.exp(np.mean(np.log(Sb), axis=0)) / np.mean(Sb, axis=0)
    flat_frac = float(np.mean(flat > 0.25))

    # speech fraction (silero, 16 kHz)
    x16 = librosa.resample(x, orig_sr=32000, target_sr=16000)
    speech_s = 0.0
    with torch.no_grad():
        chunks = torch.tensor(x16, dtype=torch.float32).unfold(0, 512, 256)
        probs = []
        for i in range(0, chunks.shape[0], 512):
            probs.append(vad_model(chunks[i:i + 512].cuda(), 16000).squeeze(-1).cpu())
        p = torch.cat(probs).numpy()
    speech_frac = float(np.mean(p > 0.5))

    return {"duration_s": round(dur, 2), "rms_dbfs": round(rms_db, 2),
            "floor_dbfs": round(floor_db, 2), "clip_frac": round(clip_frac, 6),
            "onset_rate_per_s": round(onset_rate, 3),
            "median_onset_level_dbfs": None if onset_level_db is None else round(onset_level_db, 2),
            "flat_frac": round(flat_frac, 4), "speech_frac": round(speech_frac, 4)}


def main():
    import soundfile as sf
    from silero_vad import load_silero_vad
    disc = json.load(open(os.path.join(PRIVATE, "CORPUS_DISCOVERY.private.json"), encoding="utf-8"))
    ff = ffmpeg_exe()
    os.makedirs(AUD, exist_ok=True)

    raws = [r for r in disc["records"] if r.get("provenance_class") == "A"]
    out, done = [], 0
    vad_model = load_silero_vad().cuda()
    for r in raws:
        if r["rel_path"] in DUPLICATE_EXCLUDE:
            out.append({**r, "usable": False, "exclude_reason": DUPLICATE_EXCLUDE[r["rel_path"]]})
            continue
        fam = r["family"].replace(" ", "_")
        dst = os.path.join(AUD, f"{fam}.wav")
        extract32k(ff, r["abs_path"], dst)
        info = sf.info(dst)
        diag = diagnostics(dst, vad_model)
        assert abs(info.frames / info.samplerate - diag["duration_s"]) < 1.0
        out.append({**r, "usable": True, "work32k": dst, "diagnostics": diag})
        done += 1
        print(f"[{done:02d}] {r['family']:<14} {diag['duration_s']:>6}s rms {diag['rms_dbfs']:>7} floor {diag['floor_dbfs']:>7} "
              f"speech {diag['speech_frac']:.3f} onset/s {diag['onset_rate_per_s']:>6} flat {diag['flat_frac']:.3f} clip {diag['clip_frac']:.5f}")

    with open(os.path.join(PRIVATE, "RECORDING_DIAGNOSTICS.private.json"), "w", encoding="utf-8") as f:
        json.dump({"note": "descriptive split-design diagnostics; NOT labels; no target mining here",
                   "recordings": out}, f, ensure_ascii=False, indent=1)
    print("usable:", sum(1 for r in out if r["usable"]), "/", len(out))


if __name__ == "__main__":
    main()
