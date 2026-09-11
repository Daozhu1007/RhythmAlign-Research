# R3 Phase 1 - build audio queries from VISUALLY-CONFIRMED player contact events.
#
# Evidence chain per event: R1 stereo-gate click candidate -> handcam frame strip
# (scripts/01_select_events.py, 5 frames -80..+80 ms @ 60 fps) reviewed by eye.
# No synthetic sounds: every query slice is golden_raw.wav copied as-is
# (except 5 ms boundary fades so concatenation does not add clicks).
import json
import os

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

R3 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R1_OUT = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs"
QDIR = os.path.join(R3, "queries")
LOGS = os.path.join(R3, "logs")
SR = 48000
PRE, POST = 0.020, 0.150   # window around R1 candidate time, s
FADE = 0.005
GAP = 0.030                # silence between concatenated Q2 events

# Visual verdicts from frame-strip review (work/frames/strip_*.jpg).
# t = R1 gate click time (s, golden_raw timeline).
EVENTS = [
    {"t": 0.389,  "type": "button_hit", "confidence": "medium",
     "notes": "glove at 6 o'clock ring edge in frames; clearest audio isolation (snr +0.8 dB), single hit"},
    {"t": 1.237,  "type": "button_hit", "confidence": "high",
     "notes": "right glove strikes 2 o'clock button, clear swing-in then contact; flux 4961"},
    {"t": 2.437,  "type": "palm_tap",   "confidence": "high",
     "notes": "left palm slaps inner 8 o'clock; very clear hand contact; crowd 1"},
    {"t": 3.477,  "type": "button_hit", "confidence": "high",
     "notes": "both hands on ring (left 8 o'clock + right 12-1 o'clock), contact then retract"},
    {"t": 4.485,  "type": "button_hit", "confidence": "high",
     "notes": "right glove presses 1-2 o'clock button, CRITICAL BREAK judgment visible"},
    {"t": 10.187, "type": "palm_slap",  "confidence": "high",
     "notes": "left palm slap on 10-11 o'clock button, hand visibly on button at t=0 frame"},
    {"t": 12.960, "type": "button_hit", "confidence": "high",
     "notes": "both hands press simultaneously, PERFECT judgment; zero other candidates within 0.35 s"},
    {"t": 14.571, "type": "button_hit", "confidence": "high",
     "notes": "right hand retracts as left glove lands on 4-5 o'clock button (contact ~+40 ms)"},
    {"t": 14.907, "type": "button_hit", "confidence": "high",
     "notes": "both hands press upper ring, PERFECT visible; lowest local music level (snr +3.0 dB)"},
]

Q1_T = 14.907   # cleanest background + unambiguous two-hand button press


def slice_at(y, t):
    a = max(0, int(round((t - PRE) * SR)))
    b = min(len(y[0]), int(round((t + POST) * SR)))
    seg = np.stack([ch[a:b] for ch in y], axis=1).astype(np.float32)
    n = len(seg)
    f = int(FADE * SR)
    env = np.ones(n, dtype=np.float32)
    env[:f] = np.linspace(0, 1, f)
    env[-f:] = np.linspace(1, 0, f)
    return seg * env[:, None]


def spec_png(path, *segs_labels):
    fig, axes = plt.subplots(len(segs_labels), 1, figsize=(8, 2.2 * len(segs_labels)), squeeze=False)
    for ax, (seg, lab) in zip(axes[:, 0], segs_labels):
        mono = seg.mean(axis=1)
        S = librosa.amplitude_to_db(np.abs(librosa.stft(mono, n_fft=1024, hop_length=256)) + 1e-10)
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma", extent=[0, len(mono) / SR, 0, SR // 2])
        ax.set_title(lab, fontsize=9)
        ax.set_ylabel("Hz")
    axes[-1, 0].set_xlabel("s")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    os.makedirs(QDIR, exist_ok=True)
    y, sr = librosa.load(os.path.join(R1_OUT, "golden_raw.wav"), sr=SR, mono=False)
    assert sr == SR and y.shape[1] == 720000, (y.shape, sr)

    events = []
    for ev in EVENTS:
        a = max(0, (ev["t"] - PRE))
        b = min(15.0, (ev["t"] + POST))
        events.append({**ev, "start": round(a, 4), "end": round(b, 4),
                       "timestamp": round(ev["t"], 4)})

    # Q1: single cleanest event (raw slice)
    q1 = slice_at(y, Q1_T)
    sf.write(os.path.join(QDIR, "query_q1.wav"), q1, SR, subtype="FLOAT")

    # Q2: confirmed events concatenated with 30 ms gaps (raw slices, faded edges)
    parts, cat = [], np.zeros((int(GAP * SR), y.shape[0]), dtype=np.float32)
    for ev in events:
        parts.append(slice_at(y, ev["t"]))
        parts.append(cat)
    q2 = np.concatenate(parts[:-1], axis=0)
    sf.write(os.path.join(QDIR, "query_q2.wav"), q2, SR, subtype="FLOAT")

    for name, seg in (("q1", q1), ("q2", q2)):
        rms = float(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12))
        print(f"{name}: {len(seg)/SR:.3f}s peak {np.max(np.abs(seg)):.3f} rms {rms:.1f} dBFS")

    out = {
        "source": os.path.join(R1_OUT, "golden_raw.wav"),
        "window": {"pre_s": PRE, "post_s": POST, "fade_s": FADE, "gap_s": GAP},
        "q1_event": next(e for e in events if e["t"] == Q1_T),
        "q2_events": events,
        "rejected_examples": {
            "5.131/5.488/7.733 region": "music too loud locally (snr -5..-11 dB), 5-9 candidates within 0.35 s",
        },
        "visual_evidence": "work/frames/strip_*.jpg (5 frames -80..+80 ms per candidate, 60 fps handcam)",
    }
    with open(os.path.join(QDIR, "query_events.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    spec_png(os.path.join(LOGS, "01_queries_spec.png"),
             (q1, "Q1 single event @ 14.907 s"), (q2, "Q2 nine confirmed events"))
    print("saved queries/query_q{1,2}.wav + query_events.json")


if __name__ == "__main__":
    main()
