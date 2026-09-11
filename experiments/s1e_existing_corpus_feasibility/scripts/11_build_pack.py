# S1E step 11 - blind listening pack assembly.
# Rules (disclosed):
#   - items are anonymized with a deterministic shuffle (seed 20260910) so the key is
#     reproducible but the owner cannot infer identity from ordering;
#   - EVERY item (including raw) is peak-normalized to -3 dBFS for the pack ONLY, so
#     loudness differences cannot dominate blind preference; raw unmodified outputs
#     remain in outputs/audio for verification. Peak norm does not hide deletion
#     (silence stays silence); applied peak gains are recorded in listening_key.json;
#   - all items mono (methods are mono by design; raw is downmixed).
import os
import sys
import json
import hashlib
import random

import numpy as np
import soundfile as sf
import resampy

S1E = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(S1E, "scripts"))
from s1e_common import CLIP_DIR, OUT_DIR, LOG_DIR  # noqa: E402

PACK = os.path.join(S1E, "listening_pack", "audio")
SEED = 20260910
TARGET_PEAK = 10 ** (-3.0 / 20)

manifest = json.load(open(os.path.join(CLIP_DIR, "clip_manifest.json"), encoding="utf-8"))

CONDITIONS = {
    "raw": ("clips/audio/{cid}_48k_stereo.wav", 48000),
    "clapsep": ("outputs/audio/clapsep_aq_{cid}_target.wav", 32000),
    "audiosep": ("outputs/audio/audiosep_p2_{cid}_target.wav", 32000),
    "soloaudio": ("outputs/audio/soloaudio_aq_{cid}_target.wav", 24000),
    "specialist": ("outputs/audio/specialist_{cid}_target.wav", 48000),
}
AUDIOSEP_CLIPS = {"c1_speech_npc_a", "c2_speech_npc_b", "c3_taiko_prompt", "c6_dense_golden"}


def main():
    os.makedirs(PACK, exist_ok=True)
    items = []
    for clip in manifest["clips"]:
        cid = clip["id"]
        for meth, (rel, srt) in CONDITIONS.items():
            if meth == "audiosep" and cid not in AUDIOSEP_CLIPS:
                continue
            path = os.path.join(S1E, rel.format(cid=cid))
            x, sr = sf.read(path, dtype="float64", always_2d=True)
            x = np.mean(x, axis=1)
            if sr != srt:
                x = resampy.resample(x.astype(np.float32), sr, srt)
                sr = srt
            peak = float(np.max(np.abs(x)))
            gain = TARGET_PEAK / peak if peak > 1e-10 else 1.0
            if peak <= 1e-10:
                print(f"WARNING near-digital-silence item: {meth}/{cid} peak={peak:.2e}")
            x = x * gain
            items.append({"clip": cid, "method": meth, "sr": sr,
                          "peak_gain_applied": gain, "orig_peak": peak})
    rng = random.Random(SEED)
    order = list(range(len(items)))
    rng.shuffle(order)
    key = []
    for anon_idx, item_idx in enumerate(order, start=1):
        it = items[item_idx]
        anon = f"item_{anon_idx:02d}.wav"
        # reload at native rate for writing
        rel, srt = CONDITIONS[it["method"]]
        path = os.path.join(S1E, rel.format(cid=it["clip"]))
        x, sr = sf.read(path, dtype="float64", always_2d=True)
        x = np.mean(x, axis=1)
        if sr != srt:
            import resampy as _r
            x = _r.resample(x.astype(np.float32), sr, srt)
            sr = srt
        x = x * it["peak_gain_applied"]
        sf.write(os.path.join(PACK, anon), x.astype(np.float32), sr, subtype="FLOAT")
        key.append({"item": anon, "clip": it["clip"], "method": it["method"],
                    "sr": sr, "peak_gain_applied": it["peak_gain_applied"],
                    "orig_peak": it["orig_peak"]})
    with open(os.path.join(S1E, "listening_pack", "listening_key.json"), "w",
              encoding="utf-8") as f:
        json.dump({"seed": SEED, "note": "Do NOT open before rating. Deterministic "
                                           "shuffle; peak-normalized to -3 dBFS for "
                                           "blind comparability only.",
                   "mapping": key}, f, ensure_ascii=False, indent=2)
    # playback manifest (public): clip group + duration only, no method identity
    pub = [{"item": k["item"], "clip": k["clip"], "duration_s": None, "sr": k["sr"]}
           for k in key]
    for p in pub:
        p["duration_s"] = round(next(c["duration_s"] for c in manifest["clips"]
                                     if c["id"] == p["clip"]), 1)
    json.dump(pub, open(os.path.join(S1E, "listening_pack", "playback_order.json"), "w",
                        encoding="utf-8"), indent=2)
    print(f"assembled {len(key)} items in {PACK}")


if __name__ == "__main__":
    main()
