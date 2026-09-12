# S1W step 05 - conservative TARGET_PROXY mining, TRAIN recordings ONLY.
# Audio-only signals (protocol section 19): click-band spectral flux onsets,
# transient-vs-stationary contrast, silero VAD speech rejection, friction
# flatness morphology, clipping rejection. NO vision, NO chart timing, NO
# separator outputs, NO loudest-onset selection (contrast-ranked with
# quiet-context preference instead). All material labeled TARGET_PROXY.
import os
import json
import numpy as np
import soundfile as sf
import librosa
import torch

S1W = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PRIVATE = os.path.join(S1W, "work", "private")
PROXY_DIR = os.path.join(S1W, "work", "audio", "proxy")

SR = 32000
N_FFT, HOP = 1024, 320
PRE_S, POST_S = 0.25, 1.25          # impact cut window
GATE_VAD = 0.40                     # reject windows where clear speech probable
CONTRAST_STRONG, CONTRAST_MIN = 12.0, 6.0
CONTRAST_WEAK_MIN = 3.0
MIN_SPACING_S = 1.5
PER_REC_IMPACT, PER_REC_WEAK, PER_REC_FRICTION = 15, 6, 6


def db(x):
    return 20.0 * np.log10(np.asarray(x) + 1e-12)


def frame_levels(x):
    n = len(x) // HOP
    fr = x[: n * HOP].reshape(n, HOP)
    return np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)


def vad_probs_16k(x, vad):
    import librosa as lr
    x16 = lr.resample(x, orig_sr=SR, target_sr=16000)
    chunks = torch.tensor(x16, dtype=torch.float32).unfold(0, 512, 256)
    outs = []
    with torch.no_grad():
        for i in range(0, chunks.shape[0], 1024):
            outs.append(vad(chunks[i:i + 1024].cuda(), 16000).squeeze(-1).float().cpu())
    return torch.cat(outs).numpy()  # 32 ms grid


def mine_recording(path, vad):
    x, sr = sf.read(path, dtype="float32")
    assert sr == SR
    dur = len(x) / SR

    S = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    click = (freqs >= 1500) & (freqs <= 10000)
    mid = (freqs >= 150) & (freqs <= 2000)
    band16 = (freqs >= 1000) & (freqs <= 6000)

    sb = np.sqrt(S[click].mean(axis=0) + 1e-12)         # band level (mean power/bin)
    sb_db = db(sb)
    mid_db = db(np.sqrt(S[mid].mean(axis=0) + 1e-12))
    flux = np.maximum(np.diff(sb_db, prepend=sb_db[0]), 0.0)

    # adaptive onset picking: local median + margin, absolute floor, min spacing
    k = 31
    pad = np.pad(flux, (k // 2, k // 2), mode="edge")
    loc_med = np.convolve(pad, np.ones(k) / k, mode="valid")
    # gate ALL candidates; spacing is enforced later during selection (gating first
    # prevents cluster-shadowing: the first flux peak of a cluster often fails gates)
    onsets = list(np.where((flux > loc_med + 4.5) & (flux > 3.0) & (sb_db > -70))[0])

    lev_cap = float(np.percentile(mid_db, 85))  # mid-band stationary context cap (excludes loudest 15% contexts)

    flat = np.exp(np.mean(np.log(S[band16] + 1e-12), axis=0)) / np.mean(S[band16] + 1e-12, axis=0)

    # VAD on a 32 ms grid
    vp = vad_probs_16k(x, vad)

    def vad_max(t0, t1):
        i0, i1 = max(0, int(t0 / 0.032)), min(len(vp), int(t1 / 0.032) + 1)
        return float(vp[i0:i1].max()) if i1 > i0 else 0.0

    def band_slice(mask, t0, t1):
        return mask[:, int(t0 * SR / HOP):int(t1 * SR / HOP) + 1]

    impacts = []
    for t in onsets:
        t_s = t * HOP / SR
        if t_s < PRE_S + 0.3 or t_s > dur - POST_S - 0.2:
            continue
        w0, w1 = t_s - PRE_S, t_s + POST_S
        if vad_max(w0 - 0.5, w1 + 0.5) > GATE_VAD:
            continue
        seg = x[int(w0 * SR):int(w1 * SR)]
        if np.max(np.abs(seg)) > 0.98:
            continue
        pre_db = float(np.median(sb_db[int((t_s - 0.25) * SR / HOP):int((t_s - 0.05) * SR / HOP)]))
        peak_db = float(sb_db[int((t_s) * SR / HOP):int((t_s + 0.12) * SR / HOP)].max())
        contrast = peak_db - pre_db
        if contrast < CONTRAST_WEAK_MIN:
            continue
        pre_level_db = float(np.median(mid_db[int((t_s - 0.25) * SR / HOP):int((t_s - 0.05) * SR / HOP)]))
        if pre_level_db > lev_cap:  # avoid dense/hot music contexts
            continue
        # bleed estimate: pre-onset total-band level relative to event peak
        bleed = pre_level_db - peak_db
        kind = ("strong_impact" if contrast >= CONTRAST_STRONG else
                "impact" if contrast >= CONTRAST_MIN else "weak_contact")
        impacts.append({"t": round(t_s, 3), "kind": kind, "contrast_db": round(contrast, 2),
                        "peak_db": round(peak_db, 2), "pre_level_db": round(pre_level_db, 2),
                        "bleed_db": round(bleed, 2)})

    # ranking: prefer strong contrast in quiet context (NOT loudest absolute)
    impacts.sort(key=lambda e: (e["kind"] != "strong_impact", e["kind"] != "impact",
                                -e["contrast_db"], e["pre_level_db"]))
    kept = {"strong_impact": [], "impact": [], "weak_contact": []}
    for e in impacts:
        if len(kept[e["kind"]]) >= (PER_REC_WEAK if e["kind"] == "weak_contact" else PER_REC_IMPACT):
            continue
        if any(abs(e["t"] - k["t"]) < MIN_SPACING_S for k in kept[e["kind"]]):
            continue
        kept[e["kind"]].append(e)

    # friction: long-contact spectral continuity — sustained 1-6 kHz energy with
    # NO sharp click-band onsets (protocol signal: friction morphology + long-contact
    # spectral continuity), low VAD. Absolute spectral flatness alone is dominated
    # by music harmonics and only fires in music gaps; onset-absence generalizes.
    b16_db = db(np.sqrt(S[band16].mean(axis=0) + 1e-12))
    onsets_f = ((flux > loc_med + 4.5) & (flux > 3.0)).astype(np.float64)
    dens = np.convolve(np.pad(onsets_f, (50, 50), mode="edge"),
                       np.ones(100) / 100, mode="valid")[: len(flux)]
    n_frames = len(flux)
    vp_frames = np.interp(np.arange(n_frames) * HOP / SR,
                          np.arange(len(vp)) * 0.032, vp)
    vp_smooth = np.convolve(np.pad(vp_frames, (16, 16), mode="edge"),
                            np.ones(32) / 32, mode="valid")[:n_frames]
    flat_mask = (dens < 0.03) & (vp_smooth < 0.35) & (b16_db > float(np.median(b16_db)))
    gap = int(0.25 * SR / HOP)
    idx = np.where(flat_mask)[0]
    spans, start, prev = [], None, None
    for i in idx:
        if start is None:
            start = prev = i
        elif i - prev <= gap:
            prev = i
        else:
            spans.append((start, prev))
            start = prev = i
    if start is not None:
        spans.append((start, prev))
    spans = [(a, b) for a, b in spans if (b - a) * HOP / SR >= 1.0]

    win_frames = int(2.0 * SR / HOP)
    fric = []
    for s0, s1 in spans:
        # slide a 2 s cut inside the span, keep the position with max coverage
        best, bt0 = -1.0, None
        for f in range(s0, max(s0 + 1, s1 - win_frames + 1), 16):
            t0 = f * HOP / SR
            t1 = t0 + 2.0
            if t1 > dur - 0.2:
                break
            c = float(np.mean(flat_mask[f:f + win_frames]))
            if c > best:
                best, bt0 = c, t0
        if bt0 is None or best < 0.55:
            continue
        t0, t1 = bt0, min(bt0 + 2.0, dur - 0.2)
        if t1 - t0 < 1.5:
            continue
        if vad_max(t0 - 0.3, t1 + 0.3) > GATE_VAD:
            continue
        seg = x[int(t0 * SR):int(t1 * SR)]
        if np.max(np.abs(seg)) > 0.98:
            continue
        mid_lvl = float(np.median(mid_db[int(t0 * SR / HOP):int(t1 * SR / HOP)]))
        if mid_lvl > float(np.percentile(mid_db, 75)):
            continue
        if any(abs(t0 - g["t"]) < 1.0 for g in fric):
            continue
        fric.append({"t": round(t0, 3), "dur": round(t1 - t0, 3),
                     "continuity_coverage": round(best, 3),
                     "mean_flat": round(float(np.mean(flat[int(t0 * SR / HOP):int(t1 * SR / HOP)])), 3),
                     "mid_level_db": round(mid_lvl, 2)})
        if len(fric) >= PER_REC_FRICTION:
            break
    fric = fric[:PER_REC_FRICTION]

    return {"duration_s": round(dur, 2), "impacts": kept, "friction": fric}


def main():
    from silero_vad import load_silero_vad
    vad = load_silero_vad().cuda().eval()
    split = json.load(open(os.path.join(PRIVATE, "DATA_SPLIT.private.json"), encoding="utf-8"))
    os.makedirs(PROXY_DIR, exist_ok=True)

    bank = []
    for e in split["train"]:
        rec_id, wav = e["recording_id"], e["work32k"]
        res = mine_recording(wav, vad)
        rec_dir = os.path.join(PROXY_DIR, rec_id)
        os.makedirs(rec_dir, exist_ok=True)
        x, _ = sf.read(wav, dtype="float32")
        n_ev = 0
        for kind, evs in res["impacts"].items():
            for i, ev in enumerate(evs):
                cut = ev.get("dur", PRE_S + POST_S)
                t0 = ev["t"] - (PRE_S if kind != "friction" else 0.0)
                a, b = int(max(0, t0) * SR), int(min(len(x) / SR, t0 + (1.5 if kind != "friction" else ev["dur"])) * SR)
                name = f"{kind}_{i:02d}_t{int(ev['t']*10):05d}.wav"
                sf.write(os.path.join(rec_dir, name), x[a:b], SR, subtype="FLOAT")
                ev["file"] = os.path.join("work", "audio", "proxy", rec_id, name)
                ev["seconds"] = round((b - a) / SR, 3)
                n_ev += 1
        for i, ev in enumerate(res["friction"]):
            a, b = int(ev["t"] * SR), int((ev["t"] + ev["dur"]) * SR)
            name = f"friction_{i:02d}_t{int(ev['t']*10):05d}.wav"
            sf.write(os.path.join(rec_dir, name), x[a:b], SR, subtype="FLOAT")
            ev["file"] = os.path.join("work", "audio", "proxy", rec_id, name)
            ev["seconds"] = round((b - a) / SR, 3)
            n_ev += 1
        bank.append({"recording_id": rec_id, **res})
        n_imp = sum(len(v) for v in res["impacts"].values())
        print(f"{rec_id} {e['family']:<12} impacts {n_imp:>3} (strong/ord/weak "
              f"{len(res['impacts']['strong_impact'])}/{len(res['impacts']['impact'])}/{len(res['impacts']['weak_contact'])}) "
              f"friction {len(res['friction'])}")

    with open(os.path.join(PRIVATE, "TARGET_PROXY_BANK.private.json"), "w", encoding="utf-8") as f:
        json.dump({"label": "TARGET_PROXY (not ground truth)", "recordings": bank,
                   "gates": {"vad_max": GATE_VAD, "contrast_strong": CONTRAST_STRONG,
                             "contrast_min_impact": CONTRAST_MIN, "contrast_min_weak": CONTRAST_WEAK_MIN,
                             "pre_level_cap": "per-recording frame-level p75 (mid band)",
                             "clipping_reject": 0.98, "min_spacing_s": MIN_SPACING_S}},
                  f, ensure_ascii=False, indent=1)

    n_ev = sum(sum(len(v) for v in b["impacts"].values()) + len(b["friction"]) for b in bank)
    dur_ev = sum(ev["seconds"] for b in bank for grp in list(b["impacts"].values()) + [b["friction"]] for ev in grp)
    recs_with = sum(1 for b in bank if n_ev and (sum(len(v) for v in b["impacts"].values()) + len(b["friction"])) > 0)
    print(f"\nTOTAL events {n_ev}, seconds {dur_ev:.1f}, recordings with events {recs_with}/36")


if __name__ == "__main__":
    main()
