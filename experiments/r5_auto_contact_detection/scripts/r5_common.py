# R5 common helpers: window IO, frame decode, R4-frozen audio refinement, frozen C1 build.
# C1 / refinement parameters are copied verbatim from R4 scripts 10_refine.py / 20_build.py.
import json
import os
import subprocess

import numpy as np
import imageio_ffmpeg
from scipy import signal as sig

R5 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(R5, "work")
OUT = os.path.join(R5, "outputs")
LOG = os.path.join(R5, "logs")
R1_OUT = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs"
R1_WORK = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\work"
R4_OUT = r"D:\Code\RhythmAlign\experiments\r4_contact_reconstruction\outputs"
HANCAM_MP4 = r"D:\Daozh\Videos\舞萌手元\13.2\共感觉\AP\共感怪物AP.mp4"
FF = imageio_ffmpeg.get_ffmpeg_exe()

SR = 48000
W960, H960 = 960, 540

# ---- R4-frozen C1 parameters (20_build.py) - DO NOT MODIFY ----
C1_PRE_S = 0.015
C1_POST_S = 0.150
C1_ATTACK_S = 0.010
C1_RELEASE_S = 0.060


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_json(p, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=float)


def ffmpeg_run(args):
    subprocess.run([FF, "-hide_banner", "-nostdin", "-y"] + args,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def decode_audio(path, wav_out):
    ffmpeg_run(["-i", path, "-vn", "-acodec", "pcm_f32le", wav_out])


def iter_frames(vid, w=W960):
    """Yield (idx, HxWx3 float32) frames, rotated upright, scaled to width w."""
    h = round(w * 540 / 960)  # 16:9 landscape after transpose (1080x1920 source)
    cmd = [FF, "-hide_banner", "-nostdin", "-i", vid,
           "-vf", f"transpose=2,scale={w}:-2",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = w * h * 3
    idx = 0
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        yield idx, np.frombuffer(buf, np.uint8).reshape(h, w, 3).astype(np.float32)
        idx += 1
    proc.wait()


def measure_av_offset_ms(vid, raw_wav, ctx_s=3.0, win_s=8.0):
    """Video-audio vs extracted-raw offset, same method as R1 04_golden_extract.py.
    Returns (offset_ms_video_to_raw, ncc). t_audio_raw = t_video + offset."""
    import soundfile as sf
    tmp = os.path.join(WORK, "_tmp_vidaudio.wav")
    decode_audio(vid, tmp)
    a, sr = sf.read(tmp, dtype="float64", always_2d=True)
    b, sr2 = sf.read(raw_wav, dtype="float64", always_2d=True)
    assert sr == sr2 == SR
    n = int(win_s * sr)
    aa = a[:n].mean(axis=1)
    bb = b[:n].mean(axis=1)
    sos = sig.butter(4, [300, 6000], btype="bandpass", fs=sr, output="sos")
    aa = sig.sosfiltfilt(sos, aa)
    bb = sig.sosfiltfilt(sos, bb)
    c = sig.correlate(bb, aa, mode="full", method="fft")
    k = int(np.argmax(c))
    lag = k - (n - 1)
    ncc = float(c[k] / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12))
    os.remove(tmp)
    return lag / sr * 1000.0, ncc


# ------------------------------------------------------------------
# R4-frozen audio refinement (10_refine.py logic, parameterized window)
# ------------------------------------------------------------------
def comb_envelope(raw_ctx_path, ref_warp_path, ctx_start_s, dur_s):
    """R4 multi-band onset comb envelope, exactly R4's 10_refine.py: analyzes
    [ctx_start_s, ctx_start_s+dur_s] of the ctx-aligned pair; the returned comb
    is indexed in WINDOW time (0 = window start), matching R4's timebase."""
    import librosa
    y, _ = librosa.load(raw_ctx_path, sr=SR, mono=True)
    g = y[int(ctx_start_s * SR):int((ctx_start_s + dur_s) * SR)]
    ref, _ = librosa.load(ref_warp_path, sr=SR, mono=True)
    r = ref[int(ctx_start_s * SR):int((ctx_start_s + dur_s) * SR)]
    gg, rr = g - g.mean(), r - r.mean()
    gain = float(np.dot(gg, rr) / np.dot(rr, rr))
    resid = g - gain * r

    def band_env(x, lo, hi):
        sosb = sig.butter(4, [lo, hi], btype="bandpass", fs=SR, output="sos")
        xb = sig.sosfiltfilt(sosb, x)
        env = np.abs(xb)
        kern = sig.windows.hann(49)
        return sig.convolve(env, kern, mode="same")

    bass = band_env(g, 150, 500)
    mid = band_env(resid, 500, 2000)
    hi1 = band_env(resid, 2000, 6000)
    hi2 = band_env(resid, 6000, 14000)

    def norm(e):
        p99 = np.percentile(e, 99.9)
        return e / (p99 + 1e-9)

    comb = norm(mid) + 1.15 * norm(hi1) + 0.9 * norm(hi2) + 0.45 * norm(bass)
    return comb.astype(np.float32), gain


def refine_candidates(cand_t_video, comb, av_off_ms, dur_s,
                      ids=None, search_ms=40.0):
    """R4 refinement: expected A/V mapping then +/-40 ms comb peak search.
    cand_t_video: video-time candidate times; comb in window time.
    Returns list of dicts with t_audio_refined in window time."""
    out = []
    for i, tv in enumerate(cand_t_video):
        t0 = tv + av_off_ms / 1000.0
        i0 = int(round(t0 * SR))
        lo, hi = max(i0 - int(search_ms / 1000 * SR), 0), min(i0 + int(search_ms / 1000 * SR), len(comb))
        seg = comb[lo:hi]
        if len(seg) == 0:
            out.append({"id": ids[i] if ids else f"c{i:03d}", "t_video": float(tv),
                        "status": "out_of_range"})
            continue
        k = int(np.argmax(seg)) + lo
        t_peak = k / SR
        a, b = max(k - int(0.25 * SR), 0), min(k + int(0.25 * SR), len(comb))
        excl = np.ones(len(comb), bool)
        excl[max(k - int(0.06 * SR), 0):k + int(0.06 * SR)] = False
        sel = np.arange(a, b)
        med = float(np.median(comb[sel[excl[sel]]]))
        ratio = float(seg.max() / (med + 1e-6))
        conf = ("strong" if ratio > 4 and seg.max() > 0.35
                else "present" if ratio > 2.5
                else "weak" if ratio > 1.6 else "no_transient")
        out.append({
            "id": ids[i] if ids else f"c{i:03d}", "t_video": float(tv),
            "t_audio_coarse": round(t0, 4), "t_audio_refined": round(t_peak, 4),
            "refinement_shift_ms": round((t_peak - t0) * 1000, 1),
            "peak": round(float(seg.max()), 3), "local_median": round(med, 3),
            "peak_over_noise": round(ratio, 1), "refined_confidence": conf,
        })
    return out


# ------------------------------------------------------------------
# R4-frozen C1 build (20_build.py C1 path verbatim)
# ------------------------------------------------------------------
def _ramp_up(n):
    return 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, max(n, 2)))


def _ramp_dn(n):
    return 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, max(n, 2)))


def merge_spans(spans):
    spans = sorted(spans)
    out = []
    for s, e in spans:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def envelope_from_spans(n, spans, attack, release):
    env = np.zeros(n)
    A, Rn = max(int(attack * SR), 2), max(int(release * SR), 2)
    for s, e in spans:
        s, e = max(s, 0), min(e, n)
        if e <= s:
            continue
        seg = np.ones(e - s)
        a = min(A, len(seg))
        seg[:a] = np.minimum(seg[:a], _ramp_dn(a))
        r = min(Rn, len(seg))
        seg[len(seg) - r:] = np.minimum(seg[len(seg) - r:], _ramp_up(r))
        env[s:e] = np.maximum(env[s:e], seg)
    return env


def build_c1(raw_wav, ref_aligned, t_refined, out_interaction, out_final_mix):
    """Frozen C1: pre 15 ms / post 150 ms windows, attack 10 ms / release 60 ms,
    overlapping merge by max, interaction = env*raw, final = 0.5*ref + 0.5*interaction.
    raw_wav/ref_aligned: file path or (n,2) array covering [0, dur];
    t_refined in the same time base."""
    import soundfile as sf
    def load(x):
        if isinstance(x, str):
            return sf.read(x, dtype="float64", always_2d=True)[0]
        return np.asarray(x, dtype=np.float64)
    raw = load(raw_wav)
    ref = load(ref_aligned)
    assert len(raw) == len(ref)
    n = len(raw)
    dur = n / SR
    spans = []
    for t in t_refined:
        spans.append((max(t - C1_PRE_S, 0.0), min(t + C1_POST_S, dur)))
    spans = merge_spans([(int(s * SR), int(e * SR)) for s, e in spans])
    env = envelope_from_spans(n, spans, attack=C1_ATTACK_S, release=C1_RELEASE_S)
    c1 = env[:, None] * raw
    sf.write(out_interaction, c1.astype(np.float32), SR, subtype="FLOAT")
    sf.write(out_final_mix, (0.5 * ref + 0.5 * c1).astype(np.float32), SR, subtype="FLOAT")
    return {"n_spans_after_merge": len(spans), "coverage": float(env.mean())}


# ------------------------------------------------------------------
# R4-family audio onset detector (30_evaluate.py) for stem metrics
# ------------------------------------------------------------------
def audio_onsets(y, nf=1536, hop=384):
    import librosa
    S = np.abs(librosa.stft(y.astype(np.float32), n_fft=nf, hop_length=hop)) + 1e-10
    Sdb = librosa.amplitude_to_db(S)
    fr = librosa.fft_frequencies(sr=SR, n_fft=nf)
    dS = np.maximum(0.0, np.diff(Sdb, axis=1))
    fh = dS[(fr >= 2000) & (fr < 12000)].sum(axis=0)
    med = np.median(fh)
    mad = np.median(np.abs(fh - med)) + 1e-9
    pk, _ = sig.find_peaks(fh, height=med + 4.0 * mad,
                           distance=max(1, int(0.06 * SR / hop)))
    return (pk + 1) / (SR / hop) + nf / (2 * SR)


def sinc_warp(ref, a, b, n_out, ref_index0, taps=16, chunk=200_000):
    """R1 05_fine_align.py warp: handcam ctx sample n -> ref-native index."""
    out = np.zeros((n_out, 2))
    valid = np.zeros(n_out, bool)
    n = np.arange(n_out)
    pos = (a * (n / SR) + b) * SR - ref_index0
    ks = np.arange(-taps, taps + 1)
    hwin = np.hanning(2 * taps + 1)
    for c0 in range(0, n_out, chunk):
        c1 = min(c0 + chunk, n_out)
        p = pos[c0:c1]
        base = np.floor(p).astype(np.int64)
        idx = base[:, None] + ks[None, :]
        d = p[:, None] - idx
        w = np.sinc(d) * hwin[None, :]
        inb = (idx >= 0) & (idx < len(ref))
        w *= inb
        out[c0:c1] = np.einsum("mk,mkc->mc", w, ref[np.clip(idx, 0, len(ref) - 1)])
        valid[c0:c1] = inb.all(axis=1)
    return out, valid


def bp48(x, lo, hi):
    sos = sig.butter(4, [lo, hi], btype="bandpass", fs=SR, output="sos")
    return sig.sosfiltfilt(sos, x, axis=0)


# ------------------------------------------------------------------
# Per-frame visual features (shared by dev + holdout so the detector
# runs on byte-identical feature code for both)
# ------------------------------------------------------------------
# judgment text zones in rotated 960x540 coords. CRITICAL PERFECT + FAST/SLOW
# spawn near the judged note: upper band x 240-900 / y 170-300 for most notes,
# plus a lower core (near-side bezel wedge + glass bottom, y 380-500) for
# bottom-button presses (dev audit). Union mask, same feature channels.
JZ_BAND = (250, 165, 900, 305)
JZ_LOW = (400, 380, 700, 500)


def _jz_slices(h, w):
    out = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in (JZ_BAND, JZ_LOW):
        out[y0:y1, x0:x1] = True
    return out
# glass (dark play field) ellipse in 960x540 coords
GLASS_CX, GLASS_CY = 512.0, 272.0
GLASS_A, GLASS_B = 330.0, 155.0
# lower band where hands/arms are visible (bezel + near-side glass)
HAND_Y0 = 330

# judgment text yellow is r~g with low blue; player skin is orange (r>>g),
# so the |r-g| term rejects the arm when it occludes the zone.
def _warm_mask(r, g, b):
    return (r > 150) & (g > 140) & (b < 0.60 * np.minimum(r, g)) & (np.abs(r - g) < 45)


def frame_features(fr, prev):
    """fr: HxWx3 float32 upright frame; prev: previous gray frame or None.
    Returns dict of scalar features."""
    h, w, _ = fr.shape
    gray = fr.mean(axis=2)
    jz = _jz_slices(h, w)
    zr, zg, zb = fr[..., 0], fr[..., 1], fr[..., 2]
    warm = _warm_mask(zr, zg, zb) & jz
    white = (zr > 200) & (zg > 200) & (zb > 180) & jz
    yy, xx = np.mgrid[0:h, 0:w]
    rr = ((xx - GLASS_CX) / GLASS_A) ** 2 + ((yy - GLASS_CY) / GLASS_B) ** 2
    glass = rr < 0.92 ** 2
    bright_glass = glass & (gray > 200)
    f = {
        "jz_warm": float(warm.sum()),
        "jz_white": float(white.sum()),
        "glass_bright": float(bright_glass.sum()),
    }
    if prev is None:
        f["jz_diff"] = 0.0
        f["hand_diff"] = 0.0
    else:
        d = np.abs(gray - prev)
        f["jz_diff"] = float(d[jz].mean())
        f["hand_diff"] = float(d[HAND_Y0:, :].mean())
    return f, gray


def extract_features(vid):
    """Decode all frames once; return (n_frames, dict of feature series)."""
    idxs, feats, prev = [], [], None
    for i, fr in iter_frames(vid):
        f, prev = frame_features(fr, prev)
        idxs.append(i)
        feats.append(f)
    keys = feats[0].keys()
    series = {k: np.array([f[k] for f in feats], np.float64) for k in keys}
    return len(idxs), series


# ------------------------------------------------------------------
# Contact candidate detector (deterministic; parameters in config)
# 4 core features: jz_text (judgment text content), jz_diff (judgment zone
# change), glass_bright (hit-effect flash), hand_diff (hand/arm motion).
# novelty = positive rise of a 3-frame smoothed series over 2 frames,
# normalized by its own p99.5 (clipped at 2.5); weighted fusion;
# candidate = rising edge of the fused score (score above threshold AND
# 2-frame rise above slope gate, with a min-gap between candidates).
# ------------------------------------------------------------------
DEFAULT_CONFIG = {
    "smooth_frames": 3,
    "rise_frames": 2,
    "w_jz_text": 1.2,
    "w_jz_diff": 1.2,
    "w_glass_bright": 1.0,
    "w_hand_diff": 0.5,
    "threshold": 0.55,
    "slope_min": 0.25,        # min 2-frame score rise at the crossing
    "min_dist_frames": 3,
    "norm_pct": 99.5,
    "novelty_clip": 2.5,
}


def _novelty(x, smooth, rise):
    k = np.ones(smooth) / smooth
    s = np.convolve(x, k, mode="same")
    n = np.zeros(len(s))
    n[rise:] = np.maximum(0.0, s[rise:] - s[:-rise])
    return n


def detector_score(series, cfg):
    cfg = {**DEFAULT_CONFIG, **cfg}
    ns = {}
    for name, (key_a, key_b) in {
        "jz_text": ("jz_warm", "jz_white"),
        "jz_diff": ("jz_diff", None),
        "glass_bright": ("glass_bright", None),
        "hand_diff": ("hand_diff", None),
    }.items():
        x = series[key_a] + (series[key_b] if key_b else 0.0)
        ns[name] = _novelty(x, cfg["smooth_frames"], cfg["rise_frames"])
    score = np.zeros(len(next(iter(series.values()))))
    for name, w in [("jz_text", cfg["w_jz_text"]), ("jz_diff", cfg["w_jz_diff"]),
                    ("glass_bright", cfg["w_glass_bright"]),
                    ("hand_diff", cfg["w_hand_diff"])]:
        n = ns[name]
        scale = np.percentile(n, cfg["norm_pct"]) + 1e-9
        score += w * np.minimum(n / scale, cfg["novelty_clip"])
    return score, ns


def detect_candidates(series, fps, cfg=None):
    """Rising-edge peak picking. Candidate time = crossing frame / fps."""
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    score, ns = detector_score(series, cfg)
    rise = np.zeros(len(score))
    rise[cfg["rise_frames"]:] = score[cfg["rise_frames"]:] - score[:-cfg["rise_frames"]]
    cands, last = [], -10 ** 9
    for k in range(1, len(score)):
        if k - last < cfg["min_dist_frames"]:
            continue
        if score[k] >= cfg["threshold"] and rise[k] >= cfg["slope_min"]:
            cands.append({"i": int(k), "t_video": round(k / fps, 4),
                          "score": round(float(score[k]), 3)})
            last = k
    return score, cands
