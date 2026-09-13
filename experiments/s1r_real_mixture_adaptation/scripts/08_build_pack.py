# S1R step 08 - FIXED-GAIN BLIND LISTENING PACK (protocol sections 37-39, 49).
#
# 6 groups x 3 methods (RAW / ZERO-SHOT / S1R) = 18 anonymous items.
# Per-group ONE common playback gain derived from the group's RAW item ONLY,
# applied unchanged to all three items of that group. No per-item peak/RMS/
# loudness normalization, no method-specific gain (the S1W v1 mistake is not
# repeated). Relative levels preserved and audited to <= 0.05 dB.
# The blind key is PRIVATE; PLAYBACK_ORDER.json publishes anonymous names only.
import os
import sys
import json
import hashlib
import numpy as np
import soundfile as sf

S1R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(S1R, "scripts"))
import s1r_common as rc  # noqa: E402

SEALED = os.path.join(S1R, "work", "audio", "sealed")
PACK_DIR = os.path.join(S1R, "listening_pack", "wavs")
SEED = 20260914
METHODS = ("raw", "zeroshot", "s1r")
GAIN_TARGET_PEAK = 0.7          # ~ -3.1 dBFS headroom, derived from RAW only


def rms_db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    import json

    os.makedirs(PACK_DIR, exist_ok=True)
    groups = ["c1_speech_npc_a", "c2_speech_npc_b", "c3_taiko_prompt",
              "c6_dense_golden", "c7_weak_taps", "c8_slide_friction"]

    rng = np.random.default_rng(SEED)
    items = []          # (group, method, wav path)
    for g in groups:
        srcs = {}
        for m in METHODS:
            path = os.path.join(SEALED, f"{g}_{m}.wav")
            x, sr = sf.read(path, dtype="float32")
            assert sr == rc.SR
            srcs[m] = x
        # ONE common gain for the group, from RAW only
        peak = float(np.max(np.abs(srcs["raw"])))
        gain = GAIN_TARGET_PEAK / (peak + 1e-12)
        for m in METHODS:
            y = srcs[m] * gain
            name = f"{g}_{m}_pack.wav"
            dst = os.path.join(PACK_DIR, name)
            sf.write(dst, y, rc.SR)
            items.append({"group": g, "method": m, "file": name, "gain": float(gain)})

    # blinded anonymous items + randomized playback order
    order = rng.permutation(len(items))
    key, public_order = [], []
    for n, idx in enumerate(order, start=1):
        it = items[idx]
        anon = f"item_{n:02d}"
        src = os.path.join(PACK_DIR, it["file"])
        dst = os.path.join(PACK_DIR, anon + ".wav")
        os.replace(src, dst)
        key.append({"item": anon, "group": it["group"], "method": it["method"],
                    "group_gain": it["gain"], "sha256": sha256(dst)})
        public_order.append({"item": anon})

    # gain audit: relative levels inside each group preserved exactly
    audit_rows, max_err = [], 0.0
    for k in key:
        x, _ = sf.read(os.path.join(PACK_DIR, k["item"] + ".wav"), dtype="float32")
        packed_db = rms_db(x)
        # expected: source rms + 20log10(group gain) - recompute from sealed source
        src_path = os.path.join(SEALED, f"{k['group']}_{k['method']}.wav")
        s, _ = sf.read(src_path, dtype="float32")
        expected_db = rms_db(s) + 20 * np.log10(k["group_gain"])
        err = abs(packed_db - expected_db)
        max_err = max(max_err, err)
        audit_rows.append({"item": k["item"], "group": k["group"],
                           "method": k["method"], "packed_rms_db": round(packed_db, 4),
                           "expected_rms_db": round(expected_db, 4),
                           "abs_error_db": round(err, 5)})

    rc.save_json(os.path.join(S1R, "work", "private", "listening_key.private.json"),
                 {"seed": SEED, "note": "PRIVATE blind key - never publish",
                  "items": key})
    rc.save_json(os.path.join(S1R, "listening_pack", "PLAYBACK_ORDER.json"),
                 {"stage": "S1R", "n_items": len(key),
                  "instruction": "listen once, in order; free-form notes only",
                  "order": public_order})
    rc.save_json(os.path.join(S1R, "work", "private", "PACK_GAIN_AUDIT.private.json"),
                 {"max_abs_error_db": round(max_err, 5), "rows": audit_rows})
    print(json.dumps({"items": len(key), "max_gain_error_db": round(max_err, 5)}))


if __name__ == "__main__":
    main()
