# R4.5 - conservative contact window reconstruction. D0..D3 stems + final mixes.
# Frozen vs R4: oracle contact identity, A/V offset, refined audio onsets, onset
# detector, clean music alignment, final mix gain convention (0.5*ref + 0.5*stem).
# Only the interaction extraction envelope / window policy changes.
# Fixed gains, float WAV, no limiter/compressor, no normalization of event levels.
# All interaction audio cut from golden_raw.wav.
import json
import os

import numpy as np
import soundfile as sf

R4 = r"D:\Code\RhythmAlign\experiments\r4_contact_reconstruction"
R45 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(R45, "outputs")
WORK = os.path.join(R45, "work")
R1_OUT = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs"
R1_WORK = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\work"
SR = 48000
GOLDEN_S0 = int(3.0 * SR)   # golden window inside ctx-aligned files
N = 15 * SR

# C1 baseline (frozen): pre 15 ms / post 150 ms, smooth attack/release
PRE, POST = 0.015, 0.150
ATTACK, RELEASE = 0.010, 0.060
# D1: one conservative longer tail (no grid search)
POST_D1 = 0.210
# D2: type-aware fixed windows, one reasonable choice each (no sweep)
WIN_D2 = {
    "press":   (0.015, 0.180),
    "touch":   (0.015, 0.140),
    "rest":    (0.015, 0.140),   # uncertain-type transient -> touch-like
    "slide":   (0.015, 0.250),   # continuous gesture: longer window, no fragmentation
}
RELEASE_POST = 0.080          # release merges into previous event tail
RELEASE_MERGE_GAP = 0.150     # only merge if previous span ends within this of the release
# D3: bridge only very short gaps between adjacent windows (single threshold)
BRIDGE_GAP = 0.070
# D3 region policy: oracle continuous regions short enough to keep open with
# natural fades. Long / dense / uncertain regions stay on discrete windows + bridge.
REGION_OPEN = {"R1", "R3", "R6", "R8"}
REGION_OPEN_MAX_S = 1.5       # sanity: only regions shorter than this may be fully opened
# Region -> audio offset: contacts_refined timestamps were refined on golden_raw.wav
# and empirically sit at t_video + ~0.3 ms (the 17.625 ms A/V residual is already
# absorbed by the refinement). Regions share the video-frame clock, so use the
# same empirically calibrated offset as the event windows.
AV_OFFSET = None              # resolved at runtime: median(t_audio_refined - t_video)


def ramp_up(n):
    return 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, max(n, 2)))

def ramp_dn(n):
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
        seg[:a] = np.minimum(seg[:a], ramp_dn(a))
        r = min(Rn, len(seg))
        seg[len(seg) - r:] = np.minimum(seg[len(seg) - r:], ramp_up(r))
        env[s:e] = np.maximum(env[s:e], seg)
    return env

def fixed_spans(tref, pre, post):
    """C1-style fixed windows around every event, merged; overlap = max, no added gain."""
    spans = []
    for t in tref:
        spans.append((max(t - pre, 0.0), min(t + post, 15.0)))
    return merge_spans([(int(s * SR), int(e * SR)) for s, e in spans])

def typed_spans(events):
    """D2: fixed conservative windows per event type; releases merge into the
    previous event's tail instead of opening their own gate pulse."""
    spans = []
    merged_release = 0
    skipped_release = []
    prev = None   # index into spans of the previous non-release event
    for ev in sorted(events, key=lambda e: e["t_audio_refined"]):
        t = ev["t_audio_refined"]
        if ev["type"] == "release":
            if prev is not None and t + 0.0 <= (prev[1] / SR + RELEASE_MERGE_GAP):
                prev[1] = max(prev[1], int((t + RELEASE_POST) * SR))
                merged_release += 1
            else:
                skipped_release.append(ev["id"])
            continue
        pre, post = WIN_D2.get(ev["type"], WIN_D2["touch"])
        s0, e0 = int(max(t - pre, 0.0) * SR), int(min(t + post, 15.0) * SR)
        spans.append([s0, e0])
        prev = spans[-1]
    return spans, merged_release, skipped_release

def bridge_spans(spans, max_gap):
    """Fill only very short gaps between adjacent merged spans (natural continuity)."""
    spans = merge_spans(spans)
    out = [spans[0][:]] if spans else []
    bridged = 0
    for s, e in spans[1:]:
        if (s - out[-1][1]) / SR <= max_gap:
            out[-1][1] = max(out[-1][1], e)
            bridged += 1
        else:
            out.append([s, e])
    return out, bridged

def main():
    raw, _ = sf.read(os.path.join(R1_OUT, "golden_raw.wav"), dtype="float64", always_2d=True)
    ref, _ = sf.read(os.path.join(R1_WORK, "ref_warp_fixed.wav"), dtype="float64", always_2d=True)
    ref = ref[GOLDEN_S0:GOLDEN_S0 + N]
    assert len(raw) == len(ref) == N

    R = json.load(open(os.path.join(R4, "outputs", "contacts_refined.json")))
    O = json.load(open(os.path.join(R4, "outputs", "contacts_oracle.json")))
    events = [e for e in R["events"]
              if e["refined_confidence"] != "no_transient"
              and e["type"] != "none"]
    tref = [e["t_audio_refined"] for e in events]
    dv = np.array([e["t_audio_refined"] - e["t_video"] for e in events])
    av_offset = float(np.median(dv))
    print(f"events used: {len(tref)} | t_audio - t_video median {av_offset*1000:.3f} ms "
          f"(empirical region offset; 17.625 ms A/V residual absorbed by refinement)")

    versions = {}

    # ---------------- D0: exact C1 reproduction ----------------
    spans0 = fixed_spans(tref, PRE, POST)
    env0 = envelope_from_spans(N, spans0, attack=ATTACK, release=RELEASE)
    d0 = env0[:, None] * raw
    versions["D0"] = {"env": env0, "stereo": d0, "spans": spans0}

    # ---------------- D1: longer fixed tail ----------------
    spans1 = fixed_spans(tref, PRE, POST_D1)
    env1 = envelope_from_spans(N, spans1, attack=ATTACK, release=RELEASE)
    versions["D1"] = {"env": env1, "stereo": env1[:, None] * raw, "spans": spans1}

    # ---------------- D2: type-aware fixed windows ----------------
    spans2_raw, n_merged_rel, skipped_rel = typed_spans(events)
    spans2 = merge_spans(spans2_raw)
    env2 = envelope_from_spans(N, spans2, attack=ATTACK, release=RELEASE)
    versions["D2"] = {"env": env2, "stereo": env2[:, None] * raw, "spans": spans2}

    # ---------------- D3: region-aware conservative gating ----------------
    regions = {r["id"]: r for r in O["regions"]}
    spans3 = [sp[:] for sp in spans2_raw]
    region_note = {}
    for rid in sorted(regions):
        rg = regions[rid]
        if rid.startswith("G"):       # confirmed hands-off gap: never opened
            region_note[rid] = "gap, kept closed"
            continue
        dur = rg["t1"] - rg["t0"]
        if rid in REGION_OPEN and dur <= REGION_OPEN_MAX_S:
            s0 = int(max(rg["t0"] + av_offset, 0.0) * SR)
            e0 = int(min(rg["t1"] + av_offset, 15.0) * SR)
            spans3.append([s0, e0])
            region_note[rid] = f"opened continuously ({dur:.2f} s, natural fades)"
        else:
            region_note[rid] = (f"discrete windows + bridge only "
                                f"({dur:.2f} s, long/dense/uncertain)")
    n_pre_bridge = len(merge_spans([sp[:] for sp in spans3]))
    spans3, n_bridged = bridge_spans(spans3, BRIDGE_GAP)
    env3 = envelope_from_spans(N, spans3, attack=ATTACK, release=RELEASE)
    versions["D3"] = {"env": env3, "stereo": env3[:, None] * raw, "spans": spans3}

    # ---------------- write stems, final mixes, envelopes ----------------
    diag = {"region_offset_ms": av_offset * 1000, "events_used": len(tref),
            "frozen": ["contact identity", "A/V offset", "refined onsets", "onset detector",
                       "music alignment", "final mix gain convention"],
            "win_d2_ms": {k: [v[0] * 1000, v[1] * 1000] for k, v in WIN_D2.items()},
            "d2": {"release_merged": n_merged_rel, "release_skipped": skipped_rel},
            "d3": {"bridge_gap_ms": BRIDGE_GAP * 1000, "spans_before_bridge": n_pre_bridge,
                   "spans_after_bridge": len(spans3), "bridges_added": n_bridged,
                   "regions": region_note},
            "versions": {}}
    for name, v in versions.items():
        stem = v["stereo"]
        mix = 0.5 * ref + 0.5 * stem
        sf.write(os.path.join(OUT, f"{name}_interaction.wav"), stem.astype(np.float32),
                 SR, subtype="FLOAT")
        sf.write(os.path.join(OUT, f"{name}_final_mix.wav"), mix.astype(np.float32),
                 SR, subtype="FLOAT")
        np.save(os.path.join(WORK, f"env_{name}.npy"), v["env"].astype(np.float64))
        diag["versions"][name] = {
            "n_spans_after_merge": len(v["spans"]),
            "coverage": float(v["env"].mean()),
        }
        print(f"{name}: spans {len(v['spans'])} | coverage {float(v['env'].mean()):.3f}")

    # D1 must be a pointwise superset of D0's envelope (longer tail, same logic)
    assert np.all(env1 >= env0 - 1e-12), "D1 envelope must dominate D0"
    json.dump(diag, open(os.path.join(WORK, "build_diag.json"), "w"), indent=1, default=float)
    print("saved stems, final mixes, envelopes, work/build_diag.json")

if __name__ == "__main__":
    main()
