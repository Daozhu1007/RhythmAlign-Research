# S2A step 10 - human listening pack (protocol sections 30-35).
#
# ONLY called if the machine promotion gate passes. At most 6 A/B pairs:
# S1R (frozen baseline) vs S2A CORRECT-REFERENCE output. One common gain per
# pair derived from the RAW source only, applied to BOTH methods; relative gain
# validated numerically. A/B side randomized per pair; the mapping is sealed in
# work/private/listening_key.private.json. WAVs stay local (gitignored).
#
# Documented deviation from section 32: the historical sealed recording's only
# candidate pristine reference FAILED alignment (song identity), so reference-
# conditioned S2A cannot run on it and the recommended c3/c6/c7 sentinels are
# unavailable. The two continuity-sensitive pairs instead come from the 5th
# held-out recording (different windows), keeping the <= 2-sentinel and
# no-recording->1/3 rules intact.
import os
import sys
import json
import soundfile as sf
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import s2a_common as sc  # noqa: E402
import s1r_common as rc  # noqa: E402
import s1r_model as s1m  # noqa: E402

PAIR_S = 7.0
TARGET_RMS_DB = -20.0
SEED = 20260915

OUT_WAV = os.path.join(sc.S2A, "listening_pack", "wavs")


def main():
    import torch
    device = torch.device("cuda")
    test_log = json.load(open(os.path.join(sc.S2A, "logs", "09_real_test.json"),
                              encoding="utf-8"))
    rows = test_log["rows"]
    pairs = json.load(open(os.path.join(sc.S2A_PRIVATE, "S2A_SPLIT.private.json"),
                           encoding="utf-8"))["pairs"]
    pair_of = {p["recording_id"]: p for p in pairs}
    split = sc.load_split()
    rec_of = {r["recording_id"]: r for r in split["recordings"]}

    # best-advantage window per recording; sentinels = two windows of the 5th
    by_rec = {}
    for r in rows:
        adv = r["suppression_adv_vs_s1r_db"]
        by_rec.setdefault(r["recording_id"], []).append((adv, r))
    ranked = sorted(by_rec.items(),
                    key=lambda kv: max(a for a, _ in kv[1]), reverse=True)
    chosen = []
    for rid, lst in ranked[:4]:
        chosen.append(max(lst, key=lambda t: t[0])[1])
    fifth = ranked[4][0]
    fifth_rows = sorted(by_rec[fifth], key=lambda t: t[0], reverse=True)
    chosen.extend([fifth_rows[0][1], fifth_rows[1][1]])

    tmp = s1m.load_model(device)
    e_pos = s1m.embed_q1(tmp)
    del tmp
    torch.cuda.empty_cache()
    e_neg = np.zeros((1, 512), dtype=np.float32)
    s2a = smod_load(device, test_log)
    ck = torch.load(os.path.join(sc.S2A, "checkpoints", test_log["freeze"]["checkpoint"]),
                    map_location=device, weights_only=False)
    s2a.load_state_dict(ck["model_state"], strict=True)
    s1r = s2a.base_model  # frozen S1R weights inside the same object
    import s1r_infer as si

    os.makedirs(OUT_WAV, exist_ok=True)
    rng = np.random.default_rng(SEED)
    key = {"seed": SEED, "note": "PRIVATE blind key - never publish", "items": []}
    audit = []
    for i, r in enumerate(chosen, start=1):
        rid, w0 = r["recording_id"], r["w0_s"]
        entry = pair_of[rid]
        x = rc.rec_audio(rec_of[rid])
        ref32k = sc.ref_audio32k(entry)
        a = int(w0 * rc.SR)
        b = int((w0 + 10.0) * rc.SR)
        center = (a + b) // 2
        half = int(PAIR_S * rc.SR / 2)
        a2, b2 = center - half, center + half
        raw = x[a2:b2]

        y_s1r = si.ola_separate(s1r, x, a2, b2, e_pos, e_neg, device, student=True)
        y_s2a = _ola_s2a(s2a, x, a2, b2, entry, ref32k, e_pos, e_neg, device)

        gain = 10 ** ((TARGET_RMS_DB - rc.rms_db(raw)) / 20.0)
        s1r_g = y_s1r * gain
        s2a_g = y_s2a * gain
        sf.write(os.path.join(OUT_WAV, "_s1r_%02d.wav" % i), s1r_g, rc.SR)
        sf.write(os.path.join(OUT_WAV, "_s2a_%02d.wav" % i), s2a_g, rc.SR)

        side = "A" if rng.random() < 0.5 else "B"
        mapping = {"s1r": "A", "s2a": "B"} if side == "A" else {"s1r": "B", "s2a": "A"}
        for meth, s in mapping.items():
            src = os.path.join(OUT_WAV, f"_{meth}_%02d.wav" % i)
            sf.write(os.path.join(OUT_WAV, f"pair_{i:02d}_{s}.wav"),
                     sf.read(src, dtype="float32")[0], rc.SR)
            os.remove(src)
        key["items"].append({"pair": f"pair_{i:02d}", "recording_id": rid,
                             "w0_s": w0, "s1r_side": mapping["s1r"],
                             "s2a_side": mapping["s2a"],
                             "raw_gain": gain})
        audit.append({"pair": f"pair_{i:02d}", "recording_id": rid,
                      "gain_from": "RAW only", "gain": gain,
                      "rms_s1r_db": round(rc.rms_db(s1r_g), 3),
                      "rms_s2a_db": round(rc.rms_db(s2a_g), 3),
                      "abs_gain_error_db": round(abs(
                          rc.rms_db(s1r_g) - rc.rms_db(s2a_g)), 5)})
        print(f"pair_{i:02d}: {rid} w0={w0} s1r->{mapping['s1r']} "
              f"adv={r['suppression_adv_vs_s1r_db']} dB")

    sc.save_json(os.path.join(sc.S2A_PRIVATE, "listening_key.private.json"), key)
    sc.save_json(os.path.join(sc.S2A_PRIVATE, "PACK_GAIN_AUDIT.private.json"),
                 {"rows": audit, "max_abs_error_db":
                     max(a["abs_gain_error_db"] for a in audit)})
    pub_rows = [{"pair": a["pair"], "recording_id": a["recording_id"],
                 "common_gain_derived_from": "RAW only",
                 "relative_gain_error_db": a["abs_gain_error_db"]}
                for a in audit]
    sc.save_json(os.path.join(sc.S2A, "listening_pack", "PACK_GAIN_AUDIT.public.json"),
                 {"note": "role-keyed public audit; method identity sealed",
                  "rows": pub_rows})
    print("PACK DONE (wavs local only)")


def smod_load(device, test_log):
    import s2a_model as smod
    return smod.build_model(device)


def _ola_s2a(model, x, a, b, entry, ref_full, e_pos, e_neg, device):
    """Identical to scripts/09_real_test.py::_ola_s2a (single source of truth)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s09", os.path.join(sc.S2A, "scripts", "09_real_test.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._ola_s2a(model, x, a, b, entry, ref_full, e_pos, e_neg, device)


if __name__ == "__main__":
    main()
