# S1W step 13b - gain-corrected blind listening pack regeneration (pack v2).
#
# WHY: pack v1 (13_build_listening_pack.py) applied PER-ITEM -3 dBFS peak
# normalization, violating the preregistered S1W gain rule (no per-output /
# per-method / per-item normalization; a common declared input-derived gain
# convention only). Per-item normalization can amplify near-silent residue and
# hide the actual magnitude of target deletion (adapted outputs are ~28-33 dB
# below RAW RMS on five of six groups).
#
# v2 PROTOCOL (this script):
#   * Same frozen inference outputs, same seed 20260912 -> identical anonymous
#     method assignment and playback order as v1 (verified programmatically
#     against the previous sealed key; the mapping itself is NEVER printed,
#     published, or exposed in any output).
#   * Exactly ONE common gain per group, derived from the RAW source only:
#         g = min(1.0, 10**(TARGET_DBFS/20) / peak(raw_32k_mono))
#     (native scale kept when the raw peak is already below target; the raw
#     48 kHz stereo -> mono -> 32 kHz transform is identical to v1). The SAME g
#     is applied to the raw, zero-shot and adapted items of that group. No
#     per-item peak/RMS/loudness normalization, no method-specific gain, ever.
#   * Validation: relative RMS (dB) among RAW / ZERO-SHOT / ADAPTED per group is
#     measured before packing (frozen float domain) and after packing (PCM_16
#     read back from disk); every pairwise difference must be preserved within
#     TOL_DB (quantization/encoding tolerance only). Packaging must not erase
#     suppression/deletion magnitude.
#   * The public audit (listening_pack/LISTENING_PACK_GAIN_AUDIT.md) is
#     role-keyed only: it contains NO item<->method correspondence. Item-level
#     detail stays in the sealed local key under work/private/.
import os
import json
import shutil
import random
import hashlib
import numpy as np
import soundfile as sf
import librosa

S1W = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(S1W, "outputs", "audio")
PACK = os.path.join(S1W, "listening_pack")
PRIVATE = os.path.join(S1W, "work", "private")

GROUPS = ["c1_speech_npc_a", "c2_speech_npc_b", "c3_taiko_prompt",
          "c6_dense_golden", "c7_weak_taps", "c8_slide_friction"]
METHODS = ["raw", "zeroshot", "adapted"]
SEED = 20260912
TARGET_DBFS = -1.0          # raw-peak headroom target for the common group gain
TOL_DB = 0.05               # allowed relative-RMS drift (resample/PCM_16 encoding only)
OLD_KEY = os.path.join(PRIVATE, "listening_key.private.json")
OLD_KEY_BAK = os.path.join(PRIVATE, "listening_key.invalid_pack_v1.private.json.bak")
PRIVATE_AUDIT = os.path.join(PRIVATE, "pack_gain_audit.private.json")
PUBLIC_AUDIT = os.path.join(PACK, "LISTENING_PACK_GAIN_AUDIT.md")
GAIN_RULE = ("one common gain per group derived from the RAW source peak only "
             "(headroom target -1 dBFS; native scale kept if raw peak is already "
             "below target), applied identically to raw/zero-shot/adapted; "
             "no per-item normalization of any kind")


def rms_db(x):
    r = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))
    return 20.0 * np.log10(max(r, 1e-12))


def peak(x):
    return float(np.max(np.abs(x))) if len(x) else 0.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assignment_hash(assign):
    h = hashlib.sha256()
    for iid in sorted(assign):
        h.update(f"{iid}|{assign[iid]['group']}|{assign[iid]['method']}".encode())
    return h.hexdigest()


def main():
    os.makedirs(PACK, exist_ok=True)

    # deterministic anonymous assignment: identical RNG consumption as v1 (same
    # seed, same loop order) -> same mapping and sheet order; never printed
    rng = random.Random(SEED)
    assign = {}
    n = 0
    for gid in GROUPS:
        order = METHODS[:]
        rng.shuffle(order)
        for method in order:
            n += 1
            assign[f"item_{n:02d}"] = {"group": gid, "method": method}

    # verify assignment is unchanged vs the sealed v1 key (programmatic access
    # only; key contents are never printed, published, or exposed)
    mapping_unchanged = None
    if os.path.exists(OLD_KEY):
        with open(OLD_KEY, encoding="utf-8") as f:
            old_hash = assignment_hash(json.load(f))
        mapping_unchanged = (old_hash == assignment_hash(assign))

    # load frozen inputs, measure pre-pack levels, derive ONE gain per group from RAW
    xs_group, pre, gains_db, frozen_sha = {}, {}, {}, {}
    for gid in GROUPS:
        raw48, _ = sf.read(os.path.join(OUT, f"raw_{gid}_48k.wav"), dtype="float32")
        mono = raw48.mean(axis=1) if raw48.ndim > 1 else raw48
        x_raw = librosa.resample(mono, orig_sr=48000, target_sr=32000).astype(np.float32)
        xs = {"raw": x_raw}
        for m in ("zeroshot", "adapted"):
            xs[m], _ = sf.read(os.path.join(OUT, f"clapsep_{m}_{gid}_32k.wav"),
                               dtype="float32")
        xs_group[gid] = xs
        pre[gid] = {m: rms_db(xs[m]) for m in METHODS}
        rp = peak(x_raw)
        g = min(1.0, (10 ** (TARGET_DBFS / 20.0)) / rp) if rp > 0 else 1.0
        gains_db[gid] = 20.0 * np.log10(g)
        for m in METHODS:
            frozen_sha[f"{m}_{gid}"] = sha256_file(
                os.path.join(OUT, f"raw_{gid}_48k.wav" if m == "raw"
                             else f"clapsep_{m}_{gid}_32k.wav"))

    # write the pack: same gain for all three items of a group, PCM_16 as v1
    written = []  # internal only: (item_id, group, method, length); never printed
    for gid in GROUPS:
        g = float(10 ** (gains_db[gid] / 20.0))
        for iid, a in assign.items():
            if a["group"] != gid:
                continue
            x = (xs_group[gid][a["method"]] * g).astype(np.float32)
            if peak(x) > 1.0 + 1e-6:
                raise SystemExit(f"ABORT: common gain would clip {gid}; "
                                 "per-item fixes are not allowed by protocol")
            path = os.path.join(PACK, f"{iid}.wav")
            sf.write(path, x, 32000, subtype="PCM_16")
            written.append((iid, gid, a["method"], len(x)))

    # read the pack back and validate relative-RMS preservation
    post, pair_errs = {gid: {} for gid in GROUPS}, {}
    for iid, gid, m, length in written:
        y, sr = sf.read(os.path.join(PACK, f"{iid}.wav"), dtype="float32")
        if sr != 32000 or len(y) != length:
            raise SystemExit(f"ABORT: pack readback mismatch for {gid}/{m}")
        post[gid][m] = rms_db(y)
    max_err = 0.0
    for gid in GROUPS:
        pair_errs[gid] = {}
        for a, b in (("zeroshot", "raw"), ("adapted", "raw"), ("adapted", "zeroshot")):
            before = pre[gid][a] - pre[gid][b]
            after = post[gid][a] - post[gid][b]
            err = abs(after - before)
            pair_errs[gid][f"{a}-minus-{b}"] = {
                "before_db": round(before, 3), "after_db": round(after, 3),
                "abs_err_db": round(err, 4)}
            max_err = max(max_err, err)
    preserved = bool(max_err <= TOL_DB)

    # public playback sheet: identical generation as v1 -> byte-identical file
    items = [{"item_id": iid, "challenge_group": assign[iid]["group"],
              "duration_s": round(length / 32000, 2)}
             for iid, _, _, length in sorted(written)]
    random.Random(SEED).shuffle(items)
    json.dump({"seed": SEED,
               "note": "method identities are SEALED in a private local key; "
                       "this public file lists playback order, groups and durations only",
               "items": items},
              open(os.path.join(PACK, "PLAYBACK_ORDER.json"), "w", encoding="utf-8"),
              indent=1)

    # sealed key update: same mapping, per-group common gains; v1 key backed up first
    key = {iid: {"group": assign[iid]["group"], "method": assign[iid]["method"],
                 "pack_gain_db": round(gains_db[assign[iid]["group"]], 2),
                 "gain_rule": GAIN_RULE}
           for iid in sorted(assign)}
    if os.path.exists(OLD_KEY) and not os.path.exists(OLD_KEY_BAK):
        shutil.copy2(OLD_KEY, OLD_KEY_BAK)
    json.dump(key, open(OLD_KEY, "w", encoding="utf-8"), indent=1)

    # private audit detail (local only; contains the mapping, never published)
    json.dump({"pack_revision": "v2_gain_corrected",
               "gain_rule": GAIN_RULE, "seed": SEED,
               "target_raw_peak_dbfs": TARGET_DBFS, "tolerance_db": TOL_DB,
               "mapping_unchanged_vs_v1": mapping_unchanged,
               "max_rel_rms_abs_err_db": round(max_err, 4),
               "relative_rms_preserved": preserved,
               "per_group": {gid: {"common_gain_db": round(gains_db[gid], 3),
                                   "pre_rms_dbfs": {m: round(pre[gid][m], 3)
                                                    for m in METHODS},
                                   "post_rms_dbfs": {m: round(post[gid][m], 3)
                                                     for m in METHODS},
                                   "pairs": pair_errs[gid]}
                             for gid in GROUPS},
               "frozen_input_sha256": frozen_sha},
              open(PRIVATE_AUDIT, "w", encoding="utf-8"), indent=1)

    # public audit: role-keyed only, no item<->method correspondence anywhere
    lines = [
        "# S1W Listening Pack — Gain Protocol Audit (pack v2)",
        "",
        "Pack v1 was **invalid**: it peak-normalized every item independently to −3 dBFS,",
        "violating the preregistered S1W rule (no per-item normalization; a common",
        "input-derived gain convention only). Independent normalization amplifies",
        "near-silent residue and hides the magnitude of target deletion. The historical",
        "v1 mistake is deliberately recorded here and in `LISTENING_INSTRUCTIONS.md`.",
        "",
        "## v2 gain rule (regenerated from the same frozen outputs)",
        "",
        f"- `{GAIN_RULE}`",
        "- Same seed (20260912) and generation loop as v1 → the anonymous method",
        "  assignment and playback order are unchanged (verified programmatically",
        f"  against the sealed v1 key: unchanged = {mapping_unchanged}). The",
        "  item↔method mapping itself is never printed or exposed; this audit is",
        "  role-keyed (RAW / ZERO-SHOT / ADAPTED) only and is safe to read while blind.",
        "- No retraining, no checkpoint change, no inference change: the 12 frozen",
        "  input files are hashed below and were consumed read-only.",
        "",
        "## Relative RMS preservation (before packing vs after packing)",
        "",
        "Values are RMS differences in dB within the same challenge group, measured on",
        "the frozen float outputs (before) and read back from the packed PCM_16 WAVs",
        f"(after). Tolerance: {TOL_DB} dB (resampling/encoding only).",
        "",
        "| Group | Common gain | ZERO-SHOT − RAW before → after (dB) | ADAPTED − RAW before → after (dB) | Max abs err (dB) |",
        "|---|---|---|---|---|",
    ]
    for gid in GROUPS:
        zr, ar = pair_errs[gid]["zeroshot-minus-raw"], pair_errs[gid]["adapted-minus-raw"]
        me = max(p["abs_err_db"] for p in pair_errs[gid].values())
        lines.append(
            f"| {gid} | {gains_db[gid]:+.2f} dB "
            f"| {zr['before_db']:+.2f} → {zr['after_db']:+.2f} "
            f"| {ar['before_db']:+.2f} → {ar['after_db']:+.2f} | {me:.3f} |")
    lines += [
        "",
        f"**Result: relative gain preservation {'PASS' if preserved else 'FAIL'} "
        f"(max |err| = {max_err:.4f} dB ≤ {TOL_DB} dB).**",
        "",
        "Large loudness loss between items of the same group (e.g. ADAPTED tens of dB",
        "below RAW) is genuine model behavior — suppression/deletion magnitude — and is",
        "preserved exactly by packaging; it is itself decision evidence.",
        "",
        "## Frozen input integrity (read-only, unchanged)",
        "",
        "| Frozen input | sha256 (first 16) |",
        "|---|---|",
    ]
    for name in sorted(frozen_sha):
        lines.append(f"| {name} | {frozen_sha[name][:16]}… |")
    lines.append("")
    with open(PUBLIC_AUDIT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # console summary: anonymous-safe (role-keyed, no item mapping)
    print(f"pack v2 written: {len(written)} items; common per-group gains from RAW only")
    for gid in GROUPS:
        print(f"  {gid}: gain {gains_db[gid]:+.2f} dB | "
              f"ZS-RAW {pair_errs[gid]['zeroshot-minus-raw']['before_db']:+.2f} -> "
              f"{pair_errs[gid]['zeroshot-minus-raw']['after_db']:+.2f} dB | "
              f"AD-RAW {pair_errs[gid]['adapted-minus-raw']['before_db']:+.2f} -> "
              f"{pair_errs[gid]['adapted-minus-raw']['after_db']:+.2f} dB")
    print(f"mapping unchanged vs v1: {mapping_unchanged}; "
          f"relative RMS preservation: {'PASS' if preserved else 'FAIL'} "
          f"(max err {max_err:.4f} dB); key remains sealed")


if __name__ == "__main__":
    main()
