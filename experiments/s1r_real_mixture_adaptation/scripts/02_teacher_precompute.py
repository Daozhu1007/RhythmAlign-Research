# S1R step 02 - precompute FROZEN TEACHER anchor outputs.
#
# For every HIGH_CONFIDENCE_TEACHER_ANCHOR (Phase 0 accepted TRAIN + DEV windows)
# run the frozen zero-shot teacher on the same five views used by the audit and
# store only the aligned central 6 s crops. These are PSEUDO-LABEL / ANCHOR
# EVIDENCE for conservative distillation - never ground truth (protocol section 9).
# Teacher inference happens exactly once here; training never touches the teacher.
import os
import sys
import time
import numpy as np

S1R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(S1R, "scripts"))
import s1r_common as rc  # noqa: E402

OUT_DIR = os.path.join(S1R, "work", "audio", "teacher")
VIEWS = ("early", "canonical", "late")
GAINS = (-3.0, 3.0)


def main():
    import torch  # noqa: E402
    import clapsep_lib  # noqa: E402

    t00 = time.time()
    with open(os.path.join(S1R, "work", "private", "phase0_windows.private.json"),
              encoding="utf-8") as f:
        results = json_load_results()

    anchors = [r for r in results if r.get("accepted")]
    print(f"precomputing teacher crops for {len(anchors)} anchors "
          f"({sum(1 for a in anchors if a['split'] == 'TRAIN')} TRAIN / "
          f"{sum(1 for a in anchors if a['split'] == 'DEV')} DEV)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = clapsep_lib.load_clapsep(device)
    emb = clapsep_lib.embed_audio_query(model, rc.Q1_WAV)
    zeros = np.zeros((1, 512), dtype=np.float32)

    # group anchors per recording to load raw audio once
    by_rec = {}
    for a in anchors:
        by_rec.setdefault((a["recording_id"], a["split"]), []).append(a)

    manifest = []
    n_done = 0
    for (rec_id, split), items in sorted(by_rec.items()):
        wd = {"family": None, "work32k": items[0].get("work32k")}
        # family needed for fallback path; recover from split records
        if not wd["work32k"]:
            split_all = rc.load_split()
            for r in split_all["recordings"]:
                if r["recording_id"] == rec_id:
                    wd["family"] = r["family"]
                    wd["work32k"] = r.get("work32k")
                    break
        x = rc.rec_audio(wd)
        rec_dir = os.path.join(OUT_DIR, rec_id)
        os.makedirs(rec_dir, exist_ok=True)
        for a in items:
            v = rc.build_views(x, a["w0_s"])
            files = {}
            cons = {}
            for name in VIEWS:
                import clapsep_lib
                out = clapsep_lib.separate(model, v["views"][name], emb, zeros, device)
                cs = v["crop_starts"][name]
                crop = out[cs:cs + rc.CENTRAL]
                fn = f"w{a['w0_s']}_{name}.npy"
                np.save(os.path.join(rec_dir, fn), crop.astype(np.float32))
                files[name] = fn
            for g in GAINS:
                import clapsep_lib
                out = clapsep_lib.separate(model, v["gain_views"][g], emb, zeros, device)
                corrected = (out / (10.0 ** (g / 20.0)))[int(2 * rc.SR):int(2 * rc.SR) + rc.CENTRAL]
                fn = f"w{a['w0_s']}_gain{int(g)}.npy"
                np.save(os.path.join(rec_dir, fn), corrected.astype(np.float32))
                files[f"gain{int(g)}"] = fn
            # consistency of the stored crops (recomputed from what training will see)
            ce, cc, cl = (np.load(os.path.join(rec_dir, files[n])) for n in VIEWS)
            cons = {
                "wf_corr_min": round(min(rc.wf_corr(ce, cc), rc.wf_corr(cl, cc),
                                         rc.wf_corr(ce, cl)), 5),
                "gain_gain-3_vs_canon_corr": round(rc.wf_corr(np.load(
                    os.path.join(rec_dir, files["gain-3"])), cc), 5),
            }
            manifest.append({"recording_id": rec_id, "split": split, "w0_s": a["w0_s"],
                             "files": files, "central_seconds": rc.CENTRAL / rc.SR,
                             "stored_crop_consistency": cons})
            n_done += 1
        del x
        print(f"[{n_done}/{len(anchors)} anchors] {rec_id} ({time.time() - t00:.0f}s)",
              flush=True)

    rc.save_json(os.path.join(S1R, "work", "private", "teacher_precompute.private.json"),
                 {"n_anchors": len(manifest), "teacher": "frozen zero-shot CLAPSep "
                  "(parity-verified, logs/00_parity_check.json)", "anchors": manifest})
    print("DONE", round(time.time() - t00, 1), "s")


def json_load_results():
    import json
    with open(os.path.join(S1R, "work", "private", "phase0_windows.private.json"),
              encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
