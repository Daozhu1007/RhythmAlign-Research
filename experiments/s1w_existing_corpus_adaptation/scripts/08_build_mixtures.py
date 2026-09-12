# S1W step 08 - weakly supervised constructed mixtures (protocol section 22).
# y = s_proxy + g*n with EXACT bookkeeping: gains solved for the requested TNR,
# measured TNR + every component + placement + scale stored in the recipe.
# Distribution: 50% mix / 20% target-only / 20% nuisance-only / 10% hard negatives.
# TRAIN examples sample TRAIN banks only; DEV examples sample DEV banks only.
import os
import json
import numpy as np
import soundfile as sf

S1W = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PRIVATE = os.path.join(S1W, "work", "private")
MIX_DIR = os.path.join(S1W, "work", "audio", "mixtures")

SR = 32000
T = 10
N = T * SR
SEED = 20260912
N_TRAIN, N_DEV = 2400, 160
TNR_MIX = [-20.0, -10.0, 0.0, 10.0]
TNR_HARD = [-30.0, -25.0, -20.0]

rng = np.random.default_rng(SEED)


def load_bank(path):
    b = json.load(open(os.path.join(PRIVATE, path), encoding="utf-8"))
    ev = []
    for r in b["recordings"]:
        for kind, lst in r["impacts"].items():
            ev += [{"rid": r["recording_id"], "kind": kind, **e} for e in lst]
        for e in r["friction"]:
            ev += [{"rid": r["recording_id"], "kind": "friction", **e}]
    return ev


def load_nuis(bank):
    out = {}
    for r in bank["recordings"]:
        for e in r["nuisance"]:
            out.setdefault(e["class"], []).append(e)
    out["pristine_music"] = [{"file": p["file"], "seconds": p["seconds"]}
                             for p in bank["pristine_music"]]
    return out


def read32(path):
    x, sr = sf.read(os.path.join(S1W, path), dtype="float32")
    assert sr == SR
    return x


def build_target(stratum, events, by_rid):
    """Place event clips on a silent 10 s canvas. Returns (track, active_spans, recipe)."""
    canvas = np.zeros(N, dtype=np.float32)
    spans, recipe = [], []

    def pick(cls, k):
        got = []
        tries = 0
        while len(got) < k and tries < 60:
            tries += 1
            e = events[rng.integers(len(events))]
            if e["kind"] != cls:
                continue
            if any(e["file"] == g["file"] for g in got):
                continue
            got.append(e)
        return got

    if stratum == "dense":
        items = pick("strong_impact", int(rng.integers(1, 4))) + \
                pick("impact", int(rng.integers(7, 11)))
    elif stratum == "sparse":
        items = pick("impact", int(rng.integers(2, 5)))
    elif stratum == "weak":
        items = pick("weak_contact", int(rng.integers(3, 9))) + \
                pick("impact", int(rng.integers(0, 3)))
    else:  # friction
        items = pick("friction", int(rng.integers(1, 3))) + \
                pick("impact", int(rng.integers(2, 7)))
    if not items:
        items = pick("impact", 3)
    rng.shuffle(items)

    cursor = 0.0
    for e in items:
        x = read32(e["file"])
        dur = len(x) / SR
        start = cursor + float(rng.uniform(0.15, 0.6))
        if start + dur > T - 0.1:
            break
        a, b = int(start * SR), int(start * SR) + len(x)
        canvas[a:b] += x[: max(0, N - a)]
        spans.append((a / SR, b / SR))
        recipe.append({"file": e["file"], "kind": e["kind"], "rid": e["rid"],
                       "t_start": round(start, 3), "seconds": round(dur, 3)})
        cursor = b / SR
    peak = float(np.max(np.abs(canvas)))
    scale = 1.0
    if peak > 0.9:
        scale = 0.9 / peak
        canvas *= scale
    return canvas, spans, {"events": recipe, "stratum": stratum, "target_scale": round(scale, 6)}


def build_nuisance(nuis, force_attract=False):
    """Assemble a 10 s nuisance track from 1-3 clips with 50 ms crossfades."""
    if force_attract and nuis.get("pre_song_attract"):
        base = nuis["pre_song_attract"][rng.integers(len(nuis["pre_song_attract"]))]
        others = []
    else:
        prefs = [("pristine_music", 0.5), ("pre_song_attract", 0.25), ("ambience", 0.25)]
        avail = [(c, w) for c, w in prefs if nuis.get(c)]
        tot = sum(w for _, w in avail)
        r = rng.random() * tot
        acc = 0.0
        base_cls = avail[-1][0]
        for c, w in avail:
            acc += w
            if r <= acc:
                base_cls = c
                break
        pool = nuis[base_cls]
        base = pool[rng.integers(len(pool))]
        others = []
        if rng.random() < 0.35:
            cls = "ambience" if rng.random() < 0.6 else "unrelated_impacts"
            if nuis.get(cls):
                e = nuis[cls][rng.integers(len(nuis[cls]))]
                others.append((e, float(rng.uniform(0.5, 1.0))))
    recipe = []
    track = np.zeros(N, dtype=np.float32)
    weight = np.zeros(N, dtype=np.float32)

    def add(path, gain):
        x = read32(path)
        recipe.append({"file": path, "gain": round(gain, 6)})
        pos = 0
        while pos < N:
            seg = x[: min(N - pos, len(x))] * gain
            track[pos:pos + len(seg)] += seg
            weight[pos:pos + len(seg)] += 1.0
            if pos + len(seg) >= N:
                break
            ov = min(int(0.05 * SR), len(seg) - 1)
            pos += len(seg) - ov

    add(base["file"], 1.0)
    for e, g in others:
        add(e["file"], g)
    weight[weight == 0] = 1.0
    track /= weight  # normalize crossfade regions (plain average where overlapped)
    peak = float(np.max(np.abs(track)))
    if peak > 0.9:
        track *= 0.9 / peak
    return track, {"components": recipe}


def stage_tnr(s, n, tnr_db, spans):
    e_s = sum(float(np.sum(s[int(a * SR):int(b * SR)] ** 2)) for a, b in spans) + 1e-12
    e_n = float(np.sum(n ** 2)) + 1e-12
    g = float(np.sqrt(e_s / e_n / (10 ** (tnr_db / 10))))
    y = s + g * n
    guard = 1.0
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        guard = 0.9 / peak
        y = y * guard
    tnr_meas = 10 * np.log10(e_s / e_n + 1e-12)
    return y, g, guard, float(tnr_meas)


def main():
    train_events = load_bank("TARGET_PROXY_BANK.private.json")
    dev_events = load_bank("TARGET_PROXY_BANK_DEV.private.json")
    nuis_bank = json.load(open(os.path.join(PRIVATE, "NUISANCE_BANK.private.json"), encoding="utf-8"))
    # strict split hygiene: TRAIN mixtures use TRAIN-recording acoustic nuisance and
    # TRAIN-paired pristine music; DEV mixtures use DEV-recording acoustic nuisance
    # (+ pristine music, an independent source, allowed anywhere)
    train_nuis = load_nuis({
        "recordings": [r for r in nuis_bank["recordings"] if r["split"] == "train"],
        "pristine_music": [p for p in nuis_bank["pristine_music"] if p.get("split") == "TRAIN"]})
    dev_only = {"recordings": [r for r in nuis_bank["recordings"] if r["split"] == "dev"],
                "pristine_music": nuis_bank["pristine_music"]}
    dev_nuis = load_nuis(dev_only)

    plan = []
    for i in range(N_TRAIN):
        r = rng.random()
        typ = "MIX" if r < 0.5 else ("T_ONLY" if r < 0.7 else ("N_ONLY" if r < 0.9 else "HARD"))
        plan.append(("train", i, typ))
    for i in range(N_DEV):
        r = rng.random()
        typ = "MIX" if r < 0.5 else ("T_ONLY" if r < 0.7 else ("N_ONLY" if r < 0.9 else "HARD"))
        plan.append(("dev", i, typ))

    os.makedirs(os.path.join(MIX_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(MIX_DIR, "dev"), exist_ok=True)
    recipes = []
    for split_name, idx, typ in plan:
        if split_name == "train":
            events, nuis = train_events, train_nuis
        else:
            events, nuis = dev_events, dev_nuis
        out = {"id": f"{split_name}_{idx:05d}", "split": split_name, "type": typ}
        if typ in ("MIX", "T_ONLY", "HARD"):
            stratum = rng.choice(["dense", "sparse", "weak", "friction"], p=[0.3, 0.25, 0.25, 0.2])
            s, spans, srec = build_target(stratum, events, None)
            out["target"] = srec
        if typ in ("MIX", "N_ONLY", "HARD"):
            force = typ == "HARD" and rng.random() < 0.5
            n, nrec = build_nuisance(nuis, force_attract=force)
            out["nuisance"] = nrec
        out["y_file"] = f"{out['id']}_y.wav"
        if typ in ("MIX", "T_ONLY", "HARD"):
            out["s_file"] = f"{out['id']}_s.wav"
        if typ in ("MIX", "N_ONLY", "HARD"):
            out["n_file"] = f"{out['id']}_n.wav"
        d = os.path.join(MIX_DIR, split_name)
        if typ == "MIX":
            tnr = float(rng.choice(TNR_MIX))
            y, g, guard, tm = stage_tnr(s, n, tnr, spans)
            out.update({"tnr_requested_db": tnr, "nuisance_gain": round(g, 6),
                        "peak_guard_scale": round(guard, 6), "tnr_measured_db": round(tm, 2)})
            sf.write(os.path.join(d, out["y_file"]), y, SR, subtype="FLOAT")
            sf.write(os.path.join(d, out["s_file"]), (s * guard).astype(np.float32), SR, subtype="FLOAT")
            sf.write(os.path.join(d, out["n_file"]), (n * g * guard).astype(np.float32), SR, subtype="FLOAT")
        elif typ == "HARD":
            tnr = float(rng.choice(TNR_HARD))
            y, g, guard, tm = stage_tnr(s, n, tnr, spans)
            out.update({"tnr_requested_db": tnr, "nuisance_gain": round(g, 6),
                        "peak_guard_scale": round(guard, 6), "tnr_measured_db": round(tm, 2),
                        "hard": True})
            sf.write(os.path.join(d, out["y_file"]), y, SR, subtype="FLOAT")
            sf.write(os.path.join(d, out["s_file"]), (s * guard).astype(np.float32), SR, subtype="FLOAT")
            sf.write(os.path.join(d, out["n_file"]), (n * g * guard).astype(np.float32), SR, subtype="FLOAT")
        elif typ == "T_ONLY":
            out["peak_guard_scale"] = out["target"]["target_scale"]
            sf.write(os.path.join(d, out["y_file"]), s, SR, subtype="FLOAT")
            sf.write(os.path.join(d, out["s_file"]), s, SR, subtype="FLOAT")
        else:  # N_ONLY
            sf.write(os.path.join(d, out["y_file"]), n, SR, subtype="FLOAT")
            sf.write(os.path.join(d, out["n_file"]), n, SR, subtype="FLOAT")
        recipes.append(out)
        if (idx + 1) % 400 == 0:
            print(f"{split_name} {idx+1} done", flush=True)

    with open(os.path.join(PRIVATE, "MIXTURES.private.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "note": "exact bookkeeping: y = guard*(s + gain*n); "
                   "TNR measured on target-active spans; wavs are exact stored components",
                   "examples": recipes}, f, ensure_ascii=False, indent=1)
    from collections import Counter
    c = Counter((r["split"], r["type"]) for r in recipes)
    print(dict(c))


if __name__ == "__main__":
    main()
