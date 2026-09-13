# S1R shared library: real-mixture adaptation stage.
#
# Reuses (read-only) the S1W corpus audit/provenance system:
#   - frozen split (work/private/DATA_SPLIT.private.json, seed 20260912)
#   - raw32k 32 kHz mono extractions of ORIGINAL_RAW_HANDCAM recordings
#   - target-proxy bank event times (window tagging only; NOT used as targets)
#
# S1R trains on AUTHENTIC REAL MIXTURES only. No synthetic target+nuisance
# mixtures exist anywhere in this stage (protocol section 8).
import os
import json
import numpy as np

S1R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
S1W = os.path.abspath(os.path.join(S1R, "..", "s1w_existing_corpus_adaptation"))
R3_SCRIPTS = os.path.abspath(os.path.join(S1R, "..", "r3_audio_query", "scripts"))
import sys  # noqa: E402

sys.path.insert(0, R3_SCRIPTS)

S1W_PRIVATE = os.path.join(S1W, "work", "private")
RAW32K_DIR = os.path.join(S1W, "work", "audio", "raw32k")

Q1_WAV = os.path.abspath(os.path.join(S1R, "..", "r3_audio_query", "queries", "query_q1.wav"))
CHUNK = 320000          # official 10 s @ 32 kHz (the trained chunk size)
SR = 32000
CENTRAL = 192000        # 6 s central crop shared by all context views

SEED = 20260914

# Phase 0 sampling policy (TRAIN/DEV only; song-active core approximation)
WIN_LEAD_S = 30         # skip pre-song attract region
WIN_TAIL_S = 15         # skip post-song result screen
WINDOWS_PER_REC = 6
WINDOW_SPAN_S = 14      # material span holding three distinct 10-s context views
LEFT_MARGINS_S = (4, 2, 0)   # early / canonical / late context views around central 6 s

# Anchor acceptance (conservative; protocol section 14 starting criteria)
CORR_MIN = 0.85
RMS_SPREAD_MAX_DB = 2.0
COLLAPSE_MAX_DB = -12.0     # any view below canonical teacher by more than this
AMPLIFY_MAX_DB = 6.0        # any view above canonical teacher by more than this
ONSET_LAG_MAX_MS = 20       # transient timing shift across views
ONSET_CORR_MIN = 0.80       # transient-envelope shape agreement
MRSTFT_MAX = 0.35           # log-mag MR-STFT distance across views (TRAIN-calibrated, documented)
OUT_RAW_MIN_DB = -15.0      # teacher output not near-silent vs raw window
OUT_RAW_MAX_DB = 3.0        # no amplification above the mixture

# Gate (protocol section 15)
GATE_MIN_SECONDS = 120.0
GATE_MIN_RECORDINGS = 8
GATE_MAX_REC_SHARE = 0.20
PILOT_MIN_SECONDS = 60.0
PILOT_MIN_RECORDINGS = 6


def load_split():
    with open(os.path.join(S1W_PRIVATE, "DATA_SPLIT.private.json"), encoding="utf-8") as f:
        raw = json.load(f)
    recs = []
    for key in ("sealed_primary", "test_generalization", "dev", "train"):
        for r in raw[key]:
            r["split"] = r.get("split", key.upper())
            recs.append(r)
    return {"seed": raw["seed"], "recordings": recs}


def load_proxy_bank():
    with open(os.path.join(S1W_PRIVATE, "TARGET_PROXY_BANK.private.json"), encoding="utf-8") as f:
        return json.load(f)


def rec_audio(rec):
    """raw32k mono 32 kHz float32 for a split record (read-only; S1W-derived)."""
    import soundfile as sf

    path = rec.get("work32k")
    if not path:
        stem = rec["family"].replace(" ", "_")
        path = os.path.join(RAW32K_DIR, stem + ".wav")
    if not os.path.exists(path):
        raise FileNotFoundError(f"raw32k audio missing for {rec['family']}: {path}")
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == SR, (path, sr)
    return x


def event_tags_for(rec_id, bank):
    """Proxy event times per kind for one recording (window TAGGING only)."""
    for r in bank["recordings"]:
        if r["recording_id"] == rec_id:
            kinds = {}
            for group in ("impacts", "friction"):
                events = r.get(group) or {}
                if isinstance(events, list):        # friction is a flat list
                    if events:
                        kinds["friction"] = [e["t"] for e in events]
                    continue
                for kind, evs in events.items():
                    kinds[kind] = [e["t"] for e in evs]
            return kinds
    return {}


def window_tags(kinds, w0, w1):
    """Which interaction kinds fall inside [w0, w1] (descriptive tags)."""
    tags = []
    for kind, times in (kinds or {}).items():
        if any(w0 <= t < w1 for t in times):
            tags.append(kind)
    return tags


def build_window_bank(split, bank, splits=("TRAIN", "DEV"), seed=SEED):
    """Deterministic diverse candidate windows per recording.

    Prefers windows containing proxy-tagged interaction of each kind; remaining
    slots are evenly spread + jittered across the song-active core. The proxy
    times only TAG windows; they are never training targets (protocol section 28).
    """
    rng = np.random.default_rng(seed)
    bank_out = []
    for rec in split["recordings"]:
        if rec["split"] not in splits:
            continue
        dur = rec["duration_s"]
        kinds = event_tags_for(rec["recording_id"], bank)
        lo, hi = WIN_LEAD_S, dur - WIN_TAIL_S - WINDOW_SPAN_S
        if hi <= lo:
            hi = max(lo + 1.0, dur - WINDOW_SPAN_S - 1.0)
        cands = []

        # kind-seeking candidates: windows centered so a known event sits mid-window
        for kind, times in (kinds or {}).items():
            for t in times[:4]:
                w0 = t - 5.0
                if lo - 4.0 <= w0 and w0 + WINDOW_SPAN_S <= dur - WIN_TAIL_S + 4.0:
                    cands.append(w0)
        # even spread of remaining slots
        n_seek = len(cands)
        n_spread = max(0, WINDOWS_PER_REC - n_seek)
        if n_spread > 0:
            grid = np.linspace(lo, hi, n_spread + 2)[1:-1]
            for g in grid:
                cands.append(float(g) + float(rng.uniform(-1.5, 1.5)))
        # dedupe (keep order), clamp to material bounds
        seen, final = set(), []
        for w0 in cands:
            w0 = float(min(max(w0, 0.5), dur - WINDOW_SPAN_S - 0.5))
            key = round(w0 / 2.0)
            if key in seen:
                continue
            seen.add(key)
            final.append(w0)
        bank_out.append({
            "recording_id": rec["recording_id"],
            "family": rec["family"],
            "split": rec["split"],
            "duration_s": dur,
            "work32k": rec.get("work32k"),
            "windows": final[:WINDOWS_PER_REC + 2][:WINDOWS_PER_REC + 1],
        })
    return bank_out


def build_views(x, w0):
    """The three 10-s context views + two +-3 dB gain views of one candidate window.

    MUST mirror 01_phase0_audit.audit_window exactly: the window span is 14 s at
    [w0, w0+14); the central 6 s sits at span offset 4 s; context views start at
    span offsets (4 - margin) for margin in (4, 2, 0) s; gain views scale the
    canonical view. Returns aligned central crops of the raw span too.
    """
    sr = SR
    span0 = int(round(w0 * sr))
    span = x[span0:span0 + int(WINDOW_SPAN_S * sr)]
    need = int(WINDOW_SPAN_S * sr)
    if len(span) < need:
        span = np.pad(span, (0, need - len(span)))
    views = {}
    for name, margin_s in zip(("early", "canonical", "late"), LEFT_MARGINS_S):
        s = int((4 - margin_s) * sr)
        views[name] = span[s:s + CHUNK]
    gain_views = {g: views["canonical"] * (10.0 ** (g / 20.0)) for g in (-3.0, 3.0)}
    central = span[int(4 * sr):int(10 * sr)]
    crop_starts = {name: int(m * sr) for name, m in
                   zip(("early", "canonical", "late"), LEFT_MARGINS_S)}
    return {"span": span, "views": views, "gain_views": gain_views,
            "central": central, "crop_starts": crop_starts}


# ---------------- descriptive window-level audio features (32 kHz mono) --------

def rms_db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


def click_band_db(x, sr=SR):
    """2-9 kHz band energy (transient/click retention proxy, S1E convention)."""
    import librosa

    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    band = S[(freqs >= 2000) & (freqs <= 9000)].sum(axis=0)
    return float(10 * np.log10(band.mean() + 1e-12))


def flat_frac(x, sr=SR):
    """Fraction of frames with high spectral flatness (sustained-friction proxy)."""
    import librosa

    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=512))
    fl = librosa.feature.spectral_flatness(S=S + 1e-10)[0]
    return float((fl > 0.25).mean())


def onset_env(x, sr=SR):
    import librosa

    return librosa.onset.onset_strength(y=x, sr=sr, hop_length=320)


def onset_rate_and_level(x, sr=SR):
    import librosa

    env = onset_env(x, sr)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=320, backtrack=False)
    rate = len(frames) / (len(x) / sr)
    levels = []
    for f in frames:
        i = f * 320
        seg = x[i:i + 3200]
        if len(seg):
            levels.append(rms_db(seg))
    return rate, (float(np.median(levels)) if levels else -120.0)


def onset_env_align_lag_corr(env_a, env_b, max_lag_ms=ONSET_LAG_MAX_MS, sr=SR):
    """Best lag (ms) and correlation between two onset envelopes within +-max_lag."""
    hop_ms = 320 / sr * 1000.0
    max_lag = int(round(max_lag_ms / hop_ms))
    a = env_a - env_a.mean()
    b = env_b - env_b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0, 0.0
    best_corr, best_lag = -2.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa, bb = a[-lag:], b[:len(b) + lag]
        elif lag > 0:
            aa, bb = a[:len(a) - lag], b[lag:]
        else:
            aa, bb = a, b
        n = min(len(aa), len(bb))
        if n < 10:
            continue
        c = float(np.dot(aa[:n], bb[:n]) / (na * nb + 1e-12))
        if c > best_corr:
            best_corr, best_lag = c, lag
    return best_lag * hop_ms, best_corr


# ---------------- consistency metrics between teacher/student outputs ----------

def wf_corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mrstft_logmag_dist(a, b, windows=(256, 1024, 2048)):
    """Mean log10-magnitude L1 distance across resolutions (descriptive)."""
    import librosa

    dist = 0.0
    for win in windows:
        hop = win // 4
        pa = np.abs(librosa.stft(a, n_fft=win, hop_length=hop)) + 1e-5
        pb = np.abs(librosa.stft(b, n_fft=win, hop_length=hop)) + 1e-5
        dist += float(np.abs(np.log10(pa) - np.log10(pb)).mean())
    return dist / len(windows)


def spec_logmag_corr(a, b):
    """Mean Pearson correlation of log-magnitude spectra across time frames."""
    import librosa

    S = np.log10(np.abs(librosa.stft(a, n_fft=1024, hop_length=512)) + 1e-5)
    T = np.log10(np.abs(librosa.stft(b, n_fft=1024, hop_length=512)) + 1e-5)
    n = min(S.shape[1], T.shape[1])
    S, T = S[:, :n], T[:, :n]
    S = S - S.mean(axis=1, keepdims=True)
    T = T - T.mean(axis=1, keepdims=True)
    num = (S * T).sum(axis=1)
    den = np.sqrt((S ** 2).sum(axis=1) * (T ** 2).sum(axis=1)) + 1e-12
    return float(np.mean(num / den))


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
