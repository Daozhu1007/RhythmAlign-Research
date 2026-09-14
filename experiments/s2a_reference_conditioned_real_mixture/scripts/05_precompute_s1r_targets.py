# S2A step 5: S2A training window bank + frozen S1R targets (protocol sections
# 13/20). Windows come from the S1R phase-0 candidate bank (same conventions),
# restricted to recordings whose pristine reference passed the alignment gate,
# with reference coverage >= 0.9 of the 14-s window span. The anchor target for
# S2A is the FROZEN S1R OUTPUT (pseudo-label evidence, never truth).
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import s2a_common as sc  # noqa: E402
import s1r_common as rc  # noqa: E402
import s1r_model as smod_base  # noqa: E402
import s1r_infer as si  # noqa: E402

COVERAGE_MIN = 0.9
WIN_TAIL_S = 15.0
SPAN_S = 14.0


def main():
    device = torch.device("cuda")
    zeros = np.zeros((1, 512), dtype=np.float32)
    zeroshot = smod_base.load_model(device)
    emb = smod_base.embed_q1(zeroshot)
    del zeroshot
    torch.cuda.empty_cache()
    s1r = smod_base.load_student_from_zero_shot(device, sc.S1R_CKPT)

    split = sc.load_split()
    rec_of = {r["recording_id"]: r for r in split["recordings"]}
    with open(os.path.join(sc.S1R, "work", "private", "phase0_windows.private.json"),
              encoding="utf-8") as f:
        phase0 = json.load(f)

    pairs = json.load(open(os.path.join(sc.S2A_PRIVATE, "S2A_SPLIT.private.json"),
                           encoding="utf-8"))["pairs"]
    ref_recs = [p for p in pairs if p["s2a_split"] in ("TRAIN", "DEV")]

    bank, targets = [], {}
    n_drop_cov = 0
    for p in ref_recs:
        rid = p["recording_id"]
        rec = rec_of[rid]
        x = rc.rec_audio(rec)
        ref32k = sc.ref_audio32k(p)
        wins = [w for w in phase0 if w["recording_id"] == rid and "error" not in w]
        for w in wins:
            w0 = w["w0_s"]
            # reference coverage over the 14-s span
            _, cov = sc.ref_segment(p, ref32k, w0, w0 + SPAN_S)
            if cov < COVERAGE_MIN:
                n_drop_cov += 1
                continue
            t_views = {}
            for name, margin_s in zip(("early", "canonical", "late"),
                                      rc.LEFT_MARGINS_S):
                v0 = w0 + (4.0 - margin_s)          # view start in handcam time
                y = si.ola_separate(s1r, x, int(v0 * rc.SR),
                                    int((v0 + 10.0) * rc.SR),
                                    emb, zeros, device, student=True)
                central = y[margin_s * rc.SR:(margin_s * rc.SR) + rc.CENTRAL]
                d = os.path.join(sc.S2A, "work", "audio", "s1r_targets", rid)
                os.makedirs(d, exist_ok=True)
                fn = "w0_%06.2f_%s.npy" % (w0, name)
                np.save(os.path.join(d, fn), central.astype(np.float32))
                t_views[name] = fn
            bank.append({
                "recording_id": rid, "family": rec["family"],
                "s2a_split": p["s2a_split"], "w0_s": w0,
                "labels": w.get("interaction_labels", []),
                "anchor_was_accepted": bool(w.get("accepted")),
                "ref_id": p["ref_id"], "song_id": p["song_id"],
                "files": t_views,
            })
        print(f"{rid} {rec['family']:12s} {p['s2a_split']:5s}: "
              f"{sum(1 for b in bank if b['recording_id'] == rid)} windows")

    counts = {}
    for b in bank:
        counts.setdefault(b["s2a_split"], set()).add(b["recording_id"])
    summary = {
        "n_windows": len(bank),
        "n_windows_by_split": {k: sum(1 for b in bank if b["s2a_split"] == k)
                               for k in ("TRAIN", "DEV")},
        "n_recordings_by_split": {k: len(v) for k, v in counts.items()},
        "n_dropped_low_ref_coverage": n_drop_cov,
        "seconds_by_split": {k: round(sum(1 for b in bank if b["s2a_split"] == k) * 6.0, 1)
                             for k in ("TRAIN", "DEV")},
    }
    sc.save_json(os.path.join(sc.S2A_PRIVATE, "TRAIN_WINDOW_BANK.private.json"),
                 {"seed": rc.SEED, "windows": bank, "summary": summary})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
