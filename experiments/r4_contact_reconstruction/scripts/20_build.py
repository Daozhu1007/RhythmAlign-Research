# R4 Phase 3 - C0..C3 construction. Fixed gains, float WAV, no limiter/compressor,
# no normalization of event levels. All interaction audio cut from golden_raw.wav.
import json
import os

import numpy as np
import soundfile as sf
from scipy import signal as sig
from scipy import ndimage as ndi

R4 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(R4, "work")
OUT = os.path.join(R4, "outputs")
R1_OUT = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs"
R1_WORK = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\work"
SR = 48000
GOLDEN_S0 = int(3.0 * SR)   # golden window inside ctx-aligned files
N = 15 * SR

def ramp_up(n):
    return 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, max(n, 2)))

def ramp_dn(n):
    return 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, max(n, 2)))

def main():
    raw, _ = sf.read(os.path.join(R1_OUT, "golden_raw.wav"), dtype="float64", always_2d=True)
    ref, _ = sf.read(os.path.join(R1_WORK, "ref_warp_fixed.wav"), dtype="float64", always_2d=True)
    ref = ref[GOLDEN_S0:GOLDEN_S0 + N]
    assert len(raw) == len(ref) == N

    R = json.load(open(os.path.join(OUT, "contacts_refined.json")))
    events = [e for e in R["events"]
              if e["refined_confidence"] != "no_transient"
              and e["type"] != "none"]
    tref = [e["t_audio_refined"] for e in events]
    print("events used:", len(tref))

    comb = np.load(os.path.join(WORK, "comb_env.npy"))

    # body envelope: cabinet/button resonance clock (150-2000 Hz of the raw mix)
    sosb = sig.butter(4, [150, 2000], btype="bandpass", fs=SR, output="sos")
    xb = sig.sosfiltfilt(sosb, raw.mean(axis=1))
    body = sig.convolve(np.abs(xb), sig.windows.hann(int(0.025 * SR)), mode="same")
    body /= np.percentile(body, 99.9) + 1e-9

    # ---------------- C0 ----------------
    c0 = 0.5 * ref
    sf.write(os.path.join(OUT, "C0_clean_music.wav"), c0.astype(np.float32), SR, subtype="FLOAT")

    # ---------------- C1: fixed windows ----------------
    PRE, POST = 0.015, 0.150
    spans = []
    for t in tref:
        spans.append((max(t - PRE, 0.0), min(t + POST, 15.0)))
    spans = merge_spans([(int(s * SR), int(e * SR)) for s, e in spans])
    env1 = envelope_from_spans(N, spans, attack=0.010, release=0.060)
    c1 = env1[:, None] * raw
    sf.write(os.path.join(OUT, "C1_interaction.wav"), c1.astype(np.float32), SR, subtype="FLOAT")
    sf.write(os.path.join(OUT, "C1_final_mix.wav"), (0.5 * ref + 0.5 * c1).astype(np.float32), SR, subtype="FLOAT")

    # ---------------- C2: C1 windows + percussive emphasis ----------------
    perc = percussive_blend(raw)  # per-sample gain mask applied to raw
    c2 = env1[:, None] * perc
    sf.write(os.path.join(OUT, "C2_interaction.wav"), c2.astype(np.float32), SR, subtype="FLOAT")
    sf.write(os.path.join(OUT, "C2_final_mix.wav"), (0.5 * ref + 0.5 * c2).astype(np.float32), SR, subtype="FLOAT")

    # ---------------- C3: event-adaptive envelopes ----------------
    spans3, ev3 = adaptive_spans(events, comb, body)
    env3 = envelope_from_spans(N, spans3, attack=0.008, release=0.045)
    c3 = env3[:, None] * raw
    sf.write(os.path.join(OUT, "C3_interaction.wav"), c3.astype(np.float32), SR, subtype="FLOAT")
    sf.write(os.path.join(OUT, "C3_final_mix.wav"), (0.5 * ref + 0.5 * c3).astype(np.float32), SR, subtype="FLOAT")

    # ---------------- diagnostics ----------------
    diag = {
        "events_used": len(tref),
        "c1": {"pre_ms": PRE * 1000, "post_ms": POST * 1000,
               "n_spans_after_merge": len(spans),
               "coverage": float(env1.mean())},
        "c2": {"recipe": "HPSS median-filter percussive soft mask, out = raw*(0.35+0.65*M^0.8)"},
        "c3": {"n_spans_after_merge": len(spans3), "coverage": float(env3.mean()),
               "events": ev3},
    }
    json.dump(diag, open(os.path.join(WORK, "build_diag.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in diag.items() if k != "c3"}, indent=1))
    tails = [e["tail_ms"] for e in ev3]
    print("c3 tail ms: median", np.median(tails := np.array(tails)), "p90", np.percentile(tails, 90))

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
        # window core raised to 1, with attack ramp ending at s+A and release starting at e-R
        seg = np.ones(e - s)
        a = min(A, len(seg))
        seg[:a] = np.minimum(seg[:a], ramp_dn(a))
        r = min(Rn, len(seg))
        seg[len(seg) - r:] = np.minimum(seg[len(seg) - r:], ramp_up(r))
        env[s:e] = np.maximum(env[s:e], seg)
    return env

def percussive_blend(raw, n_fft=2048, hop=512, floor=0.35, mask_pow=0.8, emph=0.65):
    mono = raw.mean(axis=1)
    f, t, Z = sig.stft(mono, fs=SR, nperseg=n_fft, noverlap=n_fft - hop)
    Zh = ndi.median_filter(np.abs(Z), size=(31, 1), mode="reflect")
    Zp = ndi.median_filter(np.abs(Z), size=(1, 31), mode="reflect")
    M = Zp / (Zh + Zp + 1e-9)
    g = floor + emph * np.power(M, mask_pow)
    g = np.clip(g, floor, 1.0)
    # invert mask to time domain by applying to STFT of each channel
    out = np.zeros_like(raw)
    for c in range(raw.shape[1]):
        fc, tc, Zc = sig.stft(raw[:, c], fs=SR, nperseg=n_fft, noverlap=n_fft - hop)
        Yc = Zc * g
        _, xr = sig.istft(Yc, fs=SR, nperseg=n_fft, noverlap=n_fft - hop)
        out[: min(len(xr), len(raw)), c] = xr[: len(raw)]
    return out

def adaptive_spans(events, comb, body, cap=0.260):
    spans, ev = [], []
    for e in events:
        t = e["t_audio_refined"]
        i = int(t * SR)
        w0, w1 = max(i - int(0.012 * SR), 0), min(i + int(0.030 * SR), len(comb) - 1)
        seg = comb[w0:w1]
        if len(seg) == 0:
            continue
        k = int(np.argmax(seg)) + w0
        peak = float(comb[k])
        # attack start: last point before peak below 18% of peak (sharp comb clock)
        thr = 0.18 * peak
        j = k
        while j > max(w0, k - int(0.030 * SR)) and comb[j] > thr:
            j -= 1
        attack_start = j / SR
        # decay measured on the body envelope (150-2000 Hz, ~25 ms smoothing):
        # cabinet/button body resonance outlives the click; stop under 30% of body peak
        b0 = max(k - int(0.005 * SR), 0)
        b1 = min(k + int(cap * SR), len(body) - 1)
        bpk = float(body[b0:b1].max())
        m = k
        while m < b1 and body[m] > 0.30 * bpk:
            m += 1
        tail_end = m / SR
        tail_ms = (tail_end - t) * 1000
        # start 4 ms before the measured attack crossing; 4 ms attack ramp ends there
        spans.append([int((attack_start - 0.004) * SR), int((tail_end + 0.045) * SR)])
        ev.append({"id": e["id"], "t_ref": t, "attack_start": round(attack_start, 4),
                   "peak_t": round(k / SR, 4), "peak": round(peak, 3), "tail_ms": round(tail_ms, 1)})
    spans = merge_spans(spans)
    return spans, ev

if __name__ == "__main__":
    main()
