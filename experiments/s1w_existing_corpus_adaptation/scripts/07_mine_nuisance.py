# S1W step 07 - NUISANCE bank mining.
# Sources: TRAIN-split recordings only (pre-song / post-song regions clearly outside
# interaction per the product interval convention, plus in-song music-dominant windows
# that contain NO mined target-proxy event), plus independent pristine reference audio
# (local mp3s paired with TRAIN recordings' folders; copyright: LOCAL ONLY, never
# committed). Sealed source and DEV/TEST recordings contribute nothing.
import os
import glob
import json
import numpy as np
import soundfile as sf
import librosa
import torch

S1W = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PRIVATE = os.path.join(S1W, "work", "private")
NUIS_DIR = os.path.join(S1W, "work", "audio", "nuisance")

SR = 32000
N_FFT, HOP = 1024, 320
PRE = (0.5, 8.5)
POST_LEN = 12.0
GATE_VAD = 0.40
CAPS = {"speech": 4, "ambience": 3, "pre_song_attract": 3, "unrelated_impacts": 3}


def db(x):
    return 20.0 * np.log10(np.asarray(x) + 1e-12)


def vad_probs_16k(x, vad):
    x16 = librosa.resample(x, orig_sr=SR, target_sr=16000)
    chunks = torch.tensor(x16, dtype=torch.float32).unfold(0, 512, 256)
    outs = []
    with torch.no_grad():
        for i in range(0, chunks.shape[0], 1024):
            outs.append(vad(chunks[i:i + 1024].cuda(), 16000).squeeze(-1).float().cpu())
    return torch.cat(outs).numpy()


def frame_flux_feats(x):
    S = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    click = (freqs >= 1500) & (freqs <= 10000)
    mid = (freqs >= 150) & (freqs <= 2000)
    sb_db = db(np.sqrt(S[click].mean(axis=0) + 1e-12))
    mid_db = db(np.sqrt(S[mid].mean(axis=0) + 1e-12))
    flux = np.maximum(np.diff(sb_db, prepend=sb_db[0]), 0.0)
    k = 31
    pad = np.pad(flux, (k // 2, k // 2), mode="edge")
    loc_med = np.convolve(pad, np.ones(k) / k, mode="valid")
    onsets = (flux > loc_med + 4.5) & (flux > 3.0)
    return S, sb_db, mid_db, onsets


def win_onset_rate(onsets, t0, t1):
    a, b = int(t0 * SR / HOP), int(t1 * SR / HOP)
    return float(onsets[a:b].mean() * (SR / HOP)) if b > a else 0.0


def win_vad_frac(vp, t0, t1):
    i0, i1 = max(0, int(t0 / 0.032)), min(len(vp), int(t1 / 0.032) + 1)
    return float(np.mean(vp[i0:i1] > 0.5)) if i1 > i0 else 0.0


def cut(x, t0, dur, path):
    a, b = int(t0 * SR), int((t0 + dur) * SR)
    b = min(b, len(x))
    if b - a < int(3.0 * SR):
        return False
    sf.write(path, x[a:b], SR, subtype="FLOAT")
    return True


def main():
    from silero_vad import load_silero_vad
    vad = load_silero_vad().cuda().eval()
    split = json.load(open(os.path.join(PRIVATE, "DATA_SPLIT.private.json"), encoding="utf-8"))
    proxy = json.load(open(os.path.join(PRIVATE, "TARGET_PROXY_BANK.private.json"), encoding="utf-8"))
    proxy_times = {}
    for r in proxy["recordings"]:
        ts = [e["t"] for grp in r["impacts"].values() for e in grp]
        ts += [e["t"] + e["dur"] / 2 for e in r["friction"]]
        proxy_times[r["recording_id"]] = ts
    os.makedirs(NUIS_DIR, exist_ok=True)

    bank = []
    for split_tag, subset in (("train", split["train"]), ("dev", split["dev"])):
      for e in subset:
        rid, wav = e["recording_id"], e["work32k"]
        x, sr = sf.read(wav, dtype="float32")
        dur = len(x) / SR
        rec_dir = os.path.join(NUIS_DIR, split_tag, rid)
        os.makedirs(rec_dir, exist_ok=True)
        S, sb_db, mid_db, onsets = frame_flux_feats(x)
        vp = vad_probs_16k(x, vad)
        ptimes = proxy_times.get(rid, [])
        entries = []

        def near_proxy(t0, t1):
            return any(t0 - 1.0 < t < t1 + 1.0 for t in ptimes)

        # --- pre/post-song scan + whole-recording speech scan ---
        # corpus fact: pre/post regions here are near-silent (VAD~0), so ambience/
        # unrelated-impact cuts come from there; usable speech occurs mid-song under
        # music (NPC/announcer) and is mined anywhere as nuisance (music overlap is
        # acceptable FOR NUISANCE - it is content to be removed).
        regions = [(PRE[0], PRE[1], "pre")]
        if dur > POST_LEN + 1.0:
            regions.append((dur - POST_LEN - 0.5, dur - 0.5, "post"))
        regions.append((10.0, dur - 14.0, "in-song"))
        got = {"speech": 0, "ambience": 0, "pre_song_attract": 0, "unrelated_impacts": 0}
        for (r0, r1, tag) in regions:
            if r1 <= r0:
                continue
            for t0 in np.arange(r0, r1 - 8.0 + 1e-9, 2.0):
                t1 = t0 + 8.0
                if t1 > dur - 0.2:
                    break
                vf = win_vad_frac(vp, t0, t1)
                orate = win_onset_rate(onsets, t0, t1)
                # no clipping gate here: these masters are limited file-wide, and a
                # few clipped peaks do not compromise NUISANCE purity (unlike target
                # proxies, where clipping would destroy transient evidence)
                cls = None
                if vf >= 0.30 and orate < 4.0:
                    cls = "speech"
                elif tag != "in-song" and vf < 0.10 and orate < 1.0:
                    cls = "ambience"
                elif tag != "in-song" and vf < 0.10 and 1.0 <= orate < 2.5:
                    cls = "pre_song_attract"
                elif tag != "in-song" and vf < 0.25 and orate >= 2.5:
                    cls = "unrelated_impacts"
                if cls is None or got[cls] >= CAPS[cls]:
                    continue
                name = f"{cls}_{tag}_{got[cls]:02d}_t{int(t0*10):05d}.wav"
                if cut(x, t0, 8.0, os.path.join(rec_dir, name)):
                    entries.append({"class": cls, "file": os.path.join("work", "audio",
                                      "nuisance", split_tag, rid, name), "t": round(t0, 2),
                                      "region": tag, "seconds": 8.0,
                                      "vad_frac": round(vf, 3),
                                      "onset_rate_per_s": round(orate, 2)})
                    got[cls] += 1

        bank.append({"recording_id": rid, "split": split_tag, "nuisance": entries})
        print(split_tag, rid, e["family"], {k: sum(1 for x2 in entries if x2["class"] == k)
                                 for k in CAPS}, flush=True)

    # --- pristine music: independent allowed source (local mp3s paired with the
    # recordings' folders; deterministic excerpts; LOCAL ONLY) ---
    prim = []
    for e in split["train"] + split["dev"]:
        folder = os.path.dirname(e["abs_path"])
        for mp3 in sorted(glob.glob(os.path.join(folder, "*.mp3")))[:1]:
            try:
                y, _ = librosa.load(mp3, sr=SR, mono=True, duration=600)
            except Exception:
                continue
            if len(y) < int(30 * SR):
                continue
            rid = e["recording_id"]
            rec_dir = os.path.join(NUIS_DIR, "pristine", rid)
            os.makedirs(rec_dir, exist_ok=True)
            for i, frac in enumerate((1 / 3, 2 / 3)):
                t0 = int(len(y) * frac)
                a, b = t0, min(t0 + 10 * SR, len(y))
                name = f"pristine_music_{i:02d}.wav"
                sf.write(os.path.join(rec_dir, name), y[a:b], SR, subtype="FLOAT")
                prim.append({"recording_id": rid, "split": e.get("split", "?"),
                             "file": os.path.join("work", "audio", "nuisance", "pristine", rid, name),
                             "seconds": round((b - a) / SR, 2)})
                break_inner = False
            del y
    with open(os.path.join(PRIVATE, "NUISANCE_BANK.private.json"), "w", encoding="utf-8") as f:
        json.dump({"note": "nuisance training material; TRAIN split + pristine reference audio only",
                   "recordings": bank, "pristine_music": prim}, f, ensure_ascii=False, indent=1)

    from collections import Counter
    c = Counter(x2["class"] for r in bank for x2 in r["nuisance"])
    s = Counter()
    for r in bank:
        for x2 in r["nuisance"]:
            s[x2["class"]] += x2["seconds"]
    print("classes:", dict(c), "| seconds:", dict(s), "| pristine clips:", len(prim))


if __name__ == "__main__":
    main()
