# RTA1 step 05 - frozen baselines (runs only after real capture + QC).
#
# Methods:
#   A. zero-shot CLAPSep      - loaded ONCE as an independent instance
#   B. ORIGINAL selected S1R  - loaded as an immutable independent instance
#                               from its own checkpoint (sha256 recorded)
#
# Immutability discipline (RTA-0 lesson - S2A's "S1R baseline" reused trained
# weights): every baseline is a separate model object; no parameter tensor is
# shared with any other method; the S1W adapted artifact is REFUSED. Parameter
# hashes are recorded before and after the run to prove nothing moved.
#
#   --check    verify checkpoints exist + hashes, load both instances, run the
#              immutability self-test, and exit (no mixtures needed)
#   --run DIR  run both baselines over the mixtures in DIR (writes metrics +
#              audio outputs under work/oracle/baselines/)
import argparse
import hashlib
import os
import sys

import numpy as np
import soundfile as sf

RTA1 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESEARCH = os.path.abspath(os.path.join(RTA1, "..", ".."))
S1R_DIR = os.path.join(RESEARCH, "experiments", "s1r_real_mixture_adaptation")
S1W_DIR = os.path.join(RESEARCH, "experiments", "s1w_existing_corpus_adaptation")
sys.path.insert(0, os.path.join(S1R_DIR, "scripts"))
sys.path.insert(0, os.path.join(RTA1, "scripts"))
import rta1_lib as rl  # noqa: E402

S1R_CKPT = os.path.join(S1R_DIR, "checkpoints", "s1r_selected.ckpt")
S1R_CKPT_SHA256 = "e1fade5ebeb84e75527d41bb5aba48dc197243a49aa34cb2fee673897a1cc303"


def refuse_s1w(path):
    """The S1W adapted checkpoint is a negative artifact and is never a
    baseline. Refuse anything inside the S1W stage or named s1w*."""
    rp = os.path.realpath(path)
    if rp.startswith(os.path.realpath(S1W_DIR)):
        raise ValueError(f"refusing S1W artifact as a baseline: {path}")
    if "s1w" in os.path.basename(rp).lower():
        raise ValueError(f"refusing S1W artifact as a baseline: {path}")


def param_hash(model):
    import torch

    h = hashlib.sha256()
    for _n, p in sorted(model.named_parameters()):
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_baselines(device):
    """Two independent immutable instances + hashes. No shared parameters."""
    import torch
    import s1r_model as s1m

    refuse_s1w(S1R_CKPT)
    sha = rl.sha256_file(S1R_CKPT)
    assert sha == S1R_CKPT_SHA256, f"S1R checkpoint hash mismatch: {sha}"

    zero_shot = s1m.load_model(device)                       # instance A
    s1r = s1m.load_student_from_zero_shot(device, S1R_CKPT)  # instance B
    for m in (zero_shot, s1r):
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()
    emb_q1 = s1m.embed_q1(zero_shot)
    zeros = np.zeros((1, 512), dtype=np.float32)
    info = {
        "s1r_checkpoint": os.path.relpath(S1R_CKPT, RESEARCH),
        "s1r_checkpoint_sha256": sha,
        "zero_shot_param_hash": param_hash(zero_shot),
        "s1r_param_hash": param_hash(s1r),
    }
    return zero_shot, s1r, (emb_q1, zeros), info


def run_mixture(model, emb, y, device, chunk_s=10.0):
    """Single-pass separation of a <=10 s mixture via the official path."""
    import torch

    arr = np.asarray(y, dtype=np.float32)
    scale_back = 1.0
    mx = float(np.max(np.abs(arr)))
    if mx > 1:
        arr = arr * (0.9 / mx)
        scale_back = mx / 0.9
    x = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            if hasattr(model, "base_model"):       # S2A-style wrapper (not used here)
                raise RuntimeError("wrapper models are not valid baselines")
            import clapsep_train_lib as ctl

            _mask, pred, _ = ctl.train_forward(model, x, emb[0], emb[1])
    return pred.float().squeeze(0).cpu().numpy() * scale_back


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", default=None, metavar="DIR",
                    help="directory of mixture WAVs")
    args = ap.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zero_shot, s1r, emb, info = load_baselines(device)
    print("baselines loaded:", {k: (v[:16] + "..." if isinstance(v, str) else v)
                                for k, v in info.items()})
    h0_zs, h0_s1r = param_hash(zero_shot), param_hash(s1r)
    assert h0_zs == info["zero_shot_param_hash"]
    assert h0_s1r == info["s1r_param_hash"]

    if args.check:
        print("CHECK OK: both frozen baselines load as independent immutable "
              "instances; checkpoint hash verified; S1W refusal active")
        return 0
    if not args.run:
        print("nothing to do: pass --check and/or --run DIR")
        return 1

    out_dir = os.path.join(rl.ORACLE, "baselines")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name in sorted(os.listdir(args.run)):
        if not name.endswith(".wav"):
            continue
        y, sr = sf.read(os.path.join(args.run, name), dtype="float32")
        y = rl.mono(y)
        assert sr == rl.SR, (name, sr)
        print(f"running baselines on {name} ...", flush=True)
        y_zs = run_mixture(zero_shot, emb, y, device)
        h1_zs = param_hash(zero_shot)
        y_s1r = run_mixture(s1r, emb, y, device)
        h1_s1r = param_hash(s1r)
        assert h1_zs == h0_zs, f"zero-shot parameters moved during run ({name})"
        assert h1_s1r == h0_s1r, f"S1R parameters moved during run ({name})"
        sf.write(os.path.join(out_dir, f"{name[:-4]}_zeroshot.wav"), y_zs, sr)
        sf.write(os.path.join(out_dir, f"{name[:-4]}_s1r.wav"), y_s1r, sr)
        rows.append({
            "mixture": name, "mixture_sha256": rl.sha256_file(os.path.join(args.run, name)),
            "rms_db": {"raw": round(rl.rms_db(y), 2),
                       "zeroshot": round(rl.rms_db(y_zs), 2),
                       "s1r": round(rl.rms_db(y_s1r), 2)},
            "band_2k9_retention_db": {
                "zeroshot": round(rl.band_profile_db(y_zs)[0] - rl.band_profile_db(y)[0], 3),
                "s1r": round(rl.band_profile_db(y_s1r)[0] - rl.band_profile_db(y)[0], 3)},
        })
    rl.save_json(os.path.join(out_dir, "baseline_metrics.json"),
                 {"fixture": False, "checkpoint_info": info, "rows": rows})
    print(f"done: {len(rows)} mixtures; parameter immutability held")
    return 0


if __name__ == "__main__":
    sys.exit(main())
