"""S1a 90 — synthetic fixture generator (tooling validation ONLY, no real acoustics).

Generates fixtures/session0001/ mimicking the raw/ layout so the full ingest ->
QC -> sync -> mixtures -> oracles -> evaluation -> listening-pack path can be
validated before real capture exists. Every artifact is labeled SYNTHETIC_FIXTURE.

Usage: python 90_generate_synthetic_fixtures.py
"""
import csv
import os
import sys

import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

SR = 48000  # native fixture rate (exercises the 48k -> 32k extraction path)
SEED = 20260910

# ----------------------------------------------------------------- sources
def env_exp(t, t0, a_fast, tau):
    e = np.zeros_like(t)
    m = t >= t0
    e[m] = np.exp(-(t[m] - t0) / tau)
    r = m & (t < t0 + a_fast)
    e[r] *= (t[r] - t0) / a_fast
    return e


def tap(t, t0, f0, amp, tau=0.03):
    rng = np.random.default_rng(int(t0 * 1000) % (2 ** 31))
    burst = rng.standard_normal(len(t)) * 0.4
    burst = np.convolve(burst, np.ones(8) / 8, mode="same")  # slight band-limit
    tone = np.sin(2 * np.pi * f0 * t)
    return amp * env_exp(t, t0, 0.002, tau) * (tone * 0.8 + burst * 0.5)


def palm(t, t0, amp):
    return tap(t, t0, 140.0, amp, tau=0.08)


def friction(t, t0, t1, amp, fc=1800.0, slow=True):
    rng = np.random.default_rng(int(t0 * 100) % (2 ** 31))
    n = rng.standard_normal(len(t))
    # simple band-pass via FFT
    X = np.fft.rfft(n)
    fr = np.fft.rfftfreq(len(t), 1.0 / SR)
    X[(fr < fc * 0.5) | (fr > fc * 2.5)] = 0.0
    n = np.fft.irfft(X, len(t))
    env = np.zeros_like(t)
    if slow:
        env += np.clip((t - t0) / 0.08, 0, 1) * np.clip((t1 - t) / 0.08, 0, 1)
    else:
        env += np.abs(np.sin(2 * np.pi * 9 * (t - t0))) * \
            np.clip((t - t0) / 0.02, 0, 1) * np.clip((t1 - t) / 0.02, 0, 1)
    m = (t >= t0) & (t <= t1)
    return np.where(m, amp * env * n * 2.0, 0.0)


def build_target_take(dur_s=36.0):
    """T1-like quiet interaction take with a scripted event schedule + rest."""
    t = np.arange(int(dur_s * SR)) / SR
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(len(t)) * 3e-4  # room noise floor
    events, friction_iv = [], []
    plan = [
        ("tap_weak", 1.5, 800.0, 0.02), ("tap", 3.0, 900.0, 0.12),
        ("tap", 4.5, 1100.0, 0.15), ("button", 6.0, 500.0, 0.2),
        ("palm", 7.5, None, 0.3),
        ("friction_slow", 9.0, 11.5, 0.05), ("friction_fast", 12.0, 13.2, 0.12),
        ("tap", 14.5, 950.0, 0.1), ("button", 15.3, 520.0, 0.18),
        ("tap_weak", 16.2, 820.0, 0.025), ("palm", 17.5, None, 0.28),
        ("rest", 19.0, 22.5, 0.0),
        ("tap", 23.0, 1050.0, 0.14), ("button", 23.6, 510.0, 0.19),
        ("tap", 24.2, 980.0, 0.13), ("button", 24.8, 530.0, 0.2),
        ("friction_slow", 26.0, 29.5, 0.045),
        ("tap_weak", 30.5, 780.0, 0.022), ("palm", 31.8, None, 0.25),
        ("rest", 33.5, 35.5, 0.0),
    ]
    for item in plan:
        kind, a, b, c = item
        if kind == "tap":
            x += tap(t, a, b, c); events.append({"t0": a, "t1": a + 0.12, "label": kind})
        elif kind == "tap_weak":
            x += tap(t, a, b, c, tau=0.015); events.append({"t0": a, "t1": a + 0.06, "label": kind})
        elif kind == "button":
            x += tap(t, a, b, c, tau=0.05); events.append({"t0": a, "t1": a + 0.15, "label": kind})
        elif kind == "palm":
            x += palm(t, a, c); events.append({"t0": a, "t1": a + 0.25, "label": kind})
        elif kind == "friction_slow":
            x += friction(t, a, b, c, slow=True); friction_iv.append({"t0": a, "t1": b})
            events.append({"t0": a, "t1": b, "label": kind})
        elif kind == "friction_fast":
            x += friction(t, a, b, c, slow=False); friction_iv.append({"t0": a, "t1": b})
            events.append({"t0": a, "t1": b, "label": kind})
        elif kind == "rest":
            pass  # deliberate silence (room noise only)
    return x, events, friction_iv


def build_pristine_music(dur_s=40.0):
    """Pristine 'maimai-like' nuisance: harmonic stacks + melody envelope + kick."""
    t = np.arange(int(dur_s * SR)) / SR
    rng = np.random.default_rng(SEED + 1)
    x = np.zeros_like(t)
    melody = [(220.0, 0.5), (277.2, 0.5), (329.6, 0.5), (440.0, 0.5)]
    step = 0.5
    for i in range(int(dur_s / step)):
        f = melody[i % len(melody)][0]
        t0 = i * step
        for h, w in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.2), (5, 0.1)):
            x += w * 0.12 * np.sin(2 * np.pi * f * h * t) * \
                env_exp(t, t0, 0.01, 0.3)
    for k in range(int(dur_s / 0.5)):  # kick
        x += 0.35 * np.sin(2 * np.pi * 55 * t) * env_exp(t, k * 0.5, 0.002, 0.09)
    x += rng.standard_normal(len(t)) * 5e-4
    return x


def build_speech_like(dur_s=30.0):
    """Announcer/NPC-like nuisance: formant-modulated voiced segments + pauses."""
    t = np.arange(int(dur_s * SR)) / SR
    rng = np.random.default_rng(SEED + 2)
    x = np.zeros_like(t)
    f0 = 140.0
    for k in range(int(dur_s / 1.2)):
        t0 = k * 1.2
        seg = (t >= t0) & (t < t0 + 0.7)
        vibr = f0 * (1 + 0.03 * np.sin(2 * np.pi * 4 * t))
        form = (np.sin(2 * np.pi * vibr * t) * 0.5
                + np.sin(2 * np.pi * 3 * vibr * t) * 0.3
                + np.sin(2 * np.pi * 5 * vibr * t) * 0.15)
        x[seg] = 0.25 * form[seg]
    return x


def build_ambience(dur_s=30.0):
    t = np.arange(int(dur_s * SR)) / SR
    rng = np.random.default_rng(SEED + 3)
    n = rng.standard_normal(len(t))
    X = np.fft.rfft(n)
    fr = np.fft.rfftfreq(len(t), 1.0 / SR)
    X[(fr > 2000)] = 0.0  # rumble-ish crowd
    x = np.fft.irfft(X, len(t))
    return 0.06 * x / (np.std(x) + 1e-12)


def build_unrelated_impacts(dur_s=30.0):
    t = np.arange(int(dur_s * SR)) / SR
    x = np.zeros_like(t)
    rng = np.random.default_rng(SEED + 4)
    for t0 in np.arange(1.0, dur_s - 1.0, 2.3):
        f0 = float(rng.uniform(300, 700))
        x += tap(t, t0, f0, 0.15, tau=0.04)  # similar-but-unrelated impacts
    return x


def build_rir():
    rng = np.random.default_rng(SEED + 5)
    n = int(0.05 * SR)
    rir = np.zeros(n)
    rir[0] = 1.0
    for d in (37, 97, 181, 313, 520, 700):
        rir[d] = rng.uniform(-0.3, 0.3) * np.exp(-d / 400.0)
    rir += rng.standard_normal(n) * 0.01 * np.exp(-np.arange(n) / 250.0)
    return rir / (np.max(np.abs(rir)) + 1e-12)


def sync_clap(t, t0, amp=0.9):
    return tap(t, t0, 2500.0, amp, tau=0.008) + tap(t, t0 + 0.004, 300.0, amp * 0.7, tau=0.05)


# -------------------------------------------------------------------- main
def main() -> None:
    sess = os.path.join(C.FIXTURE_DIR, "session0001")
    for sub in ("phone", "aux", "pristine"):
        os.makedirs(os.path.join(sess, sub), exist_ok=True)

    t_clap0, t_clap1 = 0.5, 35.2
    # --- phone channel (48 kHz, T1-like quiet take + sync claps + faint hum)
    target, events, friction_iv = build_target_take()
    t = np.arange(len(target)) / SR
    phone = target + sync_clap(t, t_clap0) + sync_clap(t, t_clap1) \
        + 0.005 * np.sin(2 * np.pi * 60 * t)  # faint mains hum to exercise QC bands
    C.write_wav(os.path.join(sess, "phone", "phone_audio.wav"), phone, SR)

    # --- close mic: target with different timbre + music leakage (-20 dB)
    music = build_pristine_music()[: len(phone)]
    close = 0.7 * target + 0.1 * music + sync_clap(t, t_clap0) * 1.2 \
        + sync_clap(t, t_clap1) * 1.2
    C.write_wav(os.path.join(sess, "aux", "closemic.wav"), close, SR)

    # --- contact channel: impulsive event evidence (no airborne timbre), leads air
    contact = np.zeros_like(phone)
    for ev in events:
        if ev["label"] in ("rest", "friction_slow", "friction_fast"):
            continue
        i0 = int(max(0, (ev["t0"] - 0.0008) * SR))  # 0.8 ms sensor lead
        rng = np.random.default_rng(int(ev["t0"] * 1e5) % (2 ** 31))
        ln = int(0.02 * SR)
        contact[i0:i0 + ln] += 0.3 * rng.standard_normal(ln) * \
            np.exp(-np.arange(ln) / (0.004 * SR))
    contact += sync_clap(t, t_clap0) + sync_clap(t, t_clap1)
    C.write_wav(os.path.join(sess, "aux", "contact.wav"), contact, SR)

    # --- playback-only + ordinary noisy gameplay (T4-like)
    C.write_wav(os.path.join(sess, "phone", "phone_playback_only.wav"),
                0.25 * music[: int(30 * SR)], SR)
    noisy = 0.25 * music[: int(30 * SR)] + target[: int(30 * SR)] * 0.8 \
        + 0.2 * build_speech_like()[: int(30 * SR)]
    C.write_wav(os.path.join(sess, "phone", "phone_noisy_gameplay.wav"), noisy, SR)

    # --- pristine music + RIR (documented path transform for mixtures)
    C.write_wav(os.path.join(sess, "pristine", "fixture_track_a.wav"),
                build_pristine_music(40.0), SR)
    rir = build_rir()
    C.write_wav(os.path.join(sess, "pristine", "fixture_rir_a.wav"), rir, SR)

    # --- anchors.csv (hints only; ingest locates exact onsets)
    with open(os.path.join(sess, "anchors.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["channel", "event", "approx_time_s", "notes"])
        for ch in ("phone", "closemic", "contact"):
            w.writerow([ch, "start_clap", t_clap0, ""])
            w.writerow([ch, "end_clap", t_clap1, ""])

    # --- session_notes.txt
    with open(os.path.join(sess, "session_notes.txt"), "w", encoding="utf-8") as f:
        f.write(
            "session_id      : session0001\n"
            "date / time     : 2026-09-10 (SYNTHETIC FIXTURE)\n"
            "phone device    : synthetic 48 kHz generator\n"
            "aux devices     : synthetic closemic + contact\n"
            "sync truth      : aux channels share phone clock in fixtures; the sync\n"
            "                  channel entries exercise the onset picker only\n")

    # --- mixture recipe (12 T2 fixture cases per S1a Part 7 structure)
    phone32 = os.path.join(sess, "phone", "phone_audio.wav")
    pristine = os.path.join(sess, "pristine", "fixture_track_a.wav")
    rir_path = os.path.join(sess, "pristine", "fixture_rir_a.wav")

    def seg(t0, t1):
        return {"path": phone32, "offset_s": t0, "duration_s": t1 - t0}

    # events/friction in segment-local time, for the evaluator
    def local(t0, t1):
        le = [{"t0": e["t0"] - t0, "t1": min(e["t1"], t1) - t0, "label": e["label"]}
              for e in events if e["t1"] > t0 + 0.01 and e["t0"] < t1 - 0.01]
        lf = [{"t0": e["t0"] - t0, "t1": min(e["t1"], t1) - t0}
              for e in friction_iv if e["t1"] > t0 + 0.05 and e["t0"] < t1 - 0.05]
        return le, lf

    cases = []
    spec = [
        # id, target window, family plan, levels
        ("case01_tap_music", (2.6, 7.6), [("music", -10), ], None),
        ("case02_tap_music", (2.6, 7.6), [("music", 0)], None),
        ("case03_tap_music", (2.6, 7.6), [("music", +10)], None),
        ("case04_weak_music", (1.0, 6.0), [("music", -20)], None),
        ("case05_palm_speech", (7.0, 12.0), [("speech", -10)], None),
        ("case06_friction_music", (8.6, 13.6), [("music", 0)], None),
        ("case07_friction_music", (25.6, 30.6), [("music", -10)], None),
        ("case08_dense_music", (22.6, 27.6), [("music", +10)], None),
        ("case09_weak_speech", (15.8, 20.8), [("speech", 0)], None),
        ("case10_button_impact", (5.6, 10.6), [("impact", 0)], None),
        ("case11_composite", (2.6, 7.6), [("music", 0), ("ambience", 0)], None),
        ("case12_composite_weak", (15.8, 20.8),
         [("music", -10), ("speech", 0), ("impact", 0)], None),
    ]
    offsets = {"music": 2.0, "speech": 1.0, "ambience": 0.5, "impact": 1.5}
    # speech/ambience/impact nuisance stems are written once here:
    C.write_wav(os.path.join(C.FIXTURE_DIR, "nuisance_speech.wav"),
                build_speech_like(30.0), SR)
    C.write_wav(os.path.join(C.FIXTURE_DIR, "nuisance_ambience.wav"),
                build_ambience(30.0), SR)
    C.write_wav(os.path.join(C.FIXTURE_DIR, "nuisance_impacts.wav"),
                build_unrelated_impacts(30.0), SR)
    src_of = {
        "music": pristine,
        "speech": os.path.join(C.FIXTURE_DIR, "nuisance_speech.wav"),
        "ambience": os.path.join(C.FIXTURE_DIR, "nuisance_ambience.wav"),
        "impact": os.path.join(C.FIXTURE_DIR, "nuisance_impacts.wav"),
    }
    for cid, (t0, t1), fams, _x in spec:
        le, lf = local(t0, t1)
        nuisances = []
        for fam, lvl in fams:
            nuisances.append({
                "path": os.path.abspath(src_of[fam]),
                "offset_s": offsets[fam], "level_db": lvl, "family": fam,
                "path_transform": ({"rir_wav": os.path.abspath(rir_path),
                                    "rir_norm_peak": True} if fam == "music" else None),
            })
        cases.append({
            "id": cid,
            "target": {"path": os.path.abspath(phone32), "offset_s": t0,
                       "duration_s": t1 - t0, "events": le,
                       "friction_intervals": lf},
            "nuisances": nuisances,
        })
    C.save_json({"sr": C.WORK_SR, "data_class": "SYNTHETIC_FIXTURE",
                 "cases": cases},
                os.path.join(C.FIXTURE_DIR, "fixture_mixtures_recipe.json"))
    C.log(f"fixtures written under {os.path.relpath(sess, C.S1A_DIR)} "
          f"({len(cases)} mixture cases)")


if __name__ == "__main__":
    main()
