# Phase 2 - automatic golden sample selection from the full handcam.
# Heuristic event model:
#   handcam transients = music onsets (explained via aligned reference) + player clicks
#                        (unexplained high-band) + taiko (unexplained low-band boom).
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import librosa
from scipy import signal as sig

from r1_common import HANCAM_MP4, WORK_DIR, load_wav, load_json, save_json, db

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 48000
NFFT, HOP = 1024, 256
FR = SR / HOP  # frame rate 187.5 Hz

# ---- load ----
y, sr = load_wav(os.path.join(WORK_DIR, "handcam_native.wav"))
assert sr == SR
mid = (y[:, 0] + y[:, 1]) * 0.5
r, _ = load_wav(os.path.join(WORK_DIR, "ref_native.wav"))
rmid = (r[:, 0] + r[:, 1]) * 0.5
coarse = load_json(os.path.join(WORK_DIR, "coarse_offset.json"))["hybrid"]["offset"]

# ---- handcam STFT features ----
S = np.abs(librosa.stft(mid.astype(np.float32), n_fft=NFFT, hop_length=HOP)) ** 2
S = librosa.power_to_db(S + 1e-10)
freqs = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
dS = np.maximum(0.0, np.diff(S, axis=1))  # positive flux, (bins, frames-1)
bands = {
    "low": (freqs >= 40) & (freqs < 300),
    "mid": (freqs >= 300) & (freqs < 2000),
    "high": (freqs >= 2000) & (freqs < 12000),
}
flux = {k: dS[b].sum(axis=0) for k, b in bands.items()}
flux["broad"] = dS.sum(axis=0)

# ---- reference onsets mapped to handcam timeline ----
on_ref = librosa.onset.onset_strength(y=rmid.astype(np.float32), sr=SR, hop_length=HOP)
ref_peaks_t = librosa.onset.onset_detect(
    onset_envelope=on_ref, sr=SR, hop_length=HOP, backtrack=False, units="time")
ref_peaks_hand = ref_peaks_t + coarse  # t_hand = t_ref + coarse (t_ref = t_hand - coarse)
# reference RMS envelope (music presence), 0.1 s resolution
hop_rms = 2048
rms_ref = librosa.feature.rms(y=rmid.astype(np.float32), frame_length=hop_rms, hop_length=hop_rms)[0]
t_rms_ref = (np.arange(len(rms_ref)) * hop_rms + hop_rms / 2) / SR + coarse
music_present_t = t_rms_ref
music_present_v = rms_ref > np.max(rms_ref) * 0.02


def music_at(t):
    idx = np.searchsorted(music_present_t, t)
    idx = np.clip(idx, 0, len(music_present_v) - 1)
    return music_present_v[idx]


# ---- peak picking per band ----
def pick_peaks(f, min_sep_s=0.06, k=4.0):
    med = np.median(f)
    mad = np.median(np.abs(f - med)) + 1e-9
    pk, props = sig.find_peaks(f, height=med + k * mad, distance=max(1, int(min_sep_s * FR)))
    return pk, props["peak_heights"]


events = {}
for band in ("low", "high", "broad"):
    pk, h = pick_peaks(flux[band])
    t = (pk + 1) / FR + NFFT / (2 * SR)  # frame -> time
    if len(t):
        # nearest aligned reference onset
        d = np.min(np.abs(t[:, None] - ref_peaks_hand[None, :])) if len(ref_peaks_hand) else 9e9
        d = np.array([np.min(np.abs(tt - ref_peaks_hand)) if len(ref_peaks_hand) else 9e9 for tt in t])
        explained = d < 0.06
    else:
        explained = np.zeros(0, bool)
    events[band] = {"t": t, "h": h, "explained_music": explained}

clicks = events["high"]["t"][~events["high"]["explained_music"]]
click_h = events["high"]["h"][~events["high"]["explained_music"]]
taiko = events["low"]["t"][~events["low"]["explained_music"]]
taiko_h = events["low"]["h"][~events["low"]["explained_music"]]
strong_click = click_h > np.median(click_h) + 1.5 * (np.median(np.abs(click_h - np.median(click_h))) + 1e-9)

n = len(mid) / SR
# ---- per-second table ----
sec = np.arange(0, int(n) + 1)
in_track = np.array([music_at(s + 0.5) for s in sec[:-1]], bool)
c_cnt = np.array([np.sum((clicks >= s) & (clicks < s + 1)) for s in sec[:-1]])
t_cnt = np.array([np.sum((taiko >= s) & (taiko < s + 1)) for s in sec[:-1]])
weak_cnt = np.array([np.sum((~strong_click) & (clicks >= s) & (clicks < s + 1)) for s in sec[:-1]])

print(f"events: clicks(unexpl high)={len(clicks)} taiko(unexpl low)={len(taiko)} "
      f"music-onsets in handcam timeline={int(np.sum((ref_peaks_hand>=0)&(ref_peaks_hand<n)))}")
print(f"in-track seconds: {in_track.sum()}/{len(in_track)}  track approx [{np.argmax(in_track)} .. {len(in_track)-1-np.argmax(in_track[::-1])}]s")

# ---- window scoring ----
def window_score(a, b, track_start, track_end):
    m = (sec[:-1] >= a) & (sec[:-1] < b)
    if not m.any():
        return -1e9, {}
    taiko_s, click_s, dense_s = t_cnt[m] >= 1, c_cnt[m] >= 1, c_cnt[m] >= 2
    sparse_s = (c_cnt[m] == 0) & in_track[m]
    both_s = taiko_s & click_s
    taiko_only = taiko_s & ~click_s
    track_frac = in_track[m].mean()
    if track_frac < 0.98:
        return -1e9, {}
    if (a < track_start + 3) or (b > track_end - 4):
        return -1e9, {}
    f = {
        "taiko_sec": min(1.0, taiko_s.sum() / 4),
        "click_sec": min(1.0, click_s.sum() / 6),
        "dense_sec": min(1.0, dense_s.sum() / 4),
        "sparse_sec": min(1.0, sparse_s.sum() / 2),
        "overlap_sec": min(1.0, both_s.sum() / 2),
        "taiko_only_sec": min(1.0, taiko_only.sum() / 1),
        "dyn_range": min(1.0, (c_cnt[m].max() - c_cnt[m].min()) / 4),
        "weak_clicks": min(1.0, weak_cnt[m].sum() / 8),
        "click_total": min(1.0, c_cnt[m].sum() / 20),
    }
    w = {"taiko_sec": 2.0, "click_sec": 1.5, "dense_sec": 1.5, "sparse_sec": 1.5,
         "overlap_sec": 2.0, "taiko_only_sec": 2.0, "dyn_range": 1.0,
         "weak_clicks": 1.0, "click_total": 1.0}
    return sum(w[k] * v for k, v in f.items()), {k: round(float(v), 2) for k, v in f.items()} | {"counts": {
        "taiko_s": int(taiko_s.sum()), "click_s": int(click_s.sum()), "dense_s": int(dense_s.sum()),
        "sparse_s": int(sparse_s.sum()), "both_s": int(both_s.sum()), "taiko_only_s": int(taiko_only.sum()),
        "clicks": int(c_cnt[m].sum()), "taiko": int(t_cnt[m].sum())}}


track_start = float(np.argmax(in_track))
track_end = float(len(in_track) - np.argmax(in_track[::-1]))
cands = []
for dur in (15, 16, 17, 18, 19, 20):
    for a in np.arange(track_start + 3, track_end - 4 - dur, 0.5):
        sc, det = window_score(a, a + dur, track_start, track_end)
        if sc > -1e8:
            cands.append((sc, float(a), dur, det))
cands.sort(key=lambda x: -x[0])
print("\ntop-5 candidate windows:")
for sc, a, dur, det in cands[:5]:
    print(f"  [{a:7.1f} .. {a+dur:7.1f}] score={sc:6.2f} {det['counts']}")

sc, start, dur, det = cands[0]
end = start + dur
reason = (
    f"score {sc:.2f}; window contains {det['counts']['taiko']} unexplained low-band (taiko) events in "
    f"{det['counts']['taiko_s']}s, {det['counts']['clicks']} unexplained high-band (player-click) events in "
    f"{det['counts']['click_s']}s; dense-operation seconds={det['counts']['dense_s']}, "
    f"operation-free in-track seconds={det['counts']['sparse_s']}, taiko+player-overlap seconds={det['counts']['both_s']}, "
    f"taiko-without-player seconds={det['counts']['taiko_only_s']}; fully inside the track portion "
    f"[{track_start:.0f},{track_end:.0f}]s, clear of track start/end."
)
sel = {
    "start": start, "end": end, "duration": dur, "selection_reason": reason,
    "score_detail": det, "coarse_offset_s": coarse,
    "track_portion_handcam": [track_start, track_end],
    "candidates": [{"start": a, "dur": d, "score": s, "detail": dt} for s, a, d, dt in cands[:10]],
}
save_json(os.path.join(WORK_DIR, "golden_selection.json"), sel)
print(f"\nSELECTED: [{start:.1f}, {end:.1f}] ({dur}s)\nreason: {reason}")

# ---- diagnostic plot ----
fig, ax = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
img = librosa.display.specshow if False else None
tt = np.arange(S.shape[1]) / FR + NFFT / (2 * SR)
ax[0].pcolormesh(tt, freqs, 10 * np.log10(S + 1e-10), shading="auto", vmin=-100, vmax=-20, cmap="magma")
ax[0].axvspan(start, end, color="lime", alpha=0.25)
ax[0].set_ylabel("Hz"); ax[0].set_yscale("log"); ax[0].set_ylim(40, 20000)
ax[0].set_title(f"handcam mid-channel spectrogram, selected golden window {start:.1f}-{end:.1f}s")
ax[1].plot(sec[:-1] + 0.5, c_cnt, label="player-click events/s", color="tab:blue")
ax[1].plot(sec[:-1] + 0.5, t_cnt, label="taiko events/s", color="tab:red")
ax[1].plot(sec[:-1] + 0.5, weak_cnt, label="weak clicks/s", color="tab:cyan", alpha=0.6)
ax[1].axvspan(start, end, color="lime", alpha=0.2)
ax[1].set_xlabel("handcam time [s]"); ax[1].legend(); ax[1].set_title("unexplained transient events per second")
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "golden_selection.png"), dpi=110)
print("plot saved")
