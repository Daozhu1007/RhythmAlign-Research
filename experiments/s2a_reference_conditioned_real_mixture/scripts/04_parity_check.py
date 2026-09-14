# S2A step 4: S1R base parity + initialization checks (protocol sections 13-14, 23).
#
# 1. S1R INFERENCE PARITY: the frozen s1r_selected.ckpt (sha-verified) reproduces
#    the STORED sealed-group S1R outputs from the S1R stage (waveform-level match).
# 2. INITIAL PARITY: at initialization, S2A(x, correct_ref) == S1R(x) and
#    S2A(x, zero_ref) == S1R(x) and S2A(x, wrong_ref) == S1R(x) (gate=0, head=0
#    by construction; verified numerically on real DEV windows).
# 3. ADAPTER GRADIENT: the reference adapter actually receives gradient through
#    the injection-invariance objective on a controlled perturbation.
import os
import sys
import json
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(__file__))
import s2a_common as sc  # noqa: E402
import s2a_model as smod  # noqa: E402
import s1r_model as smod_base  # noqa: E402
import s1r_common as rc  # noqa: E402
import s1r_infer as si  # noqa: E402

SEALED_DIR = os.path.join(sc.S1R, "work", "audio", "sealed")
STORED = os.path.join(sc.S1R, "logs", "07_sealed_test.json")


def main():
    device = torch.device("cuda")
    import hashlib

    sha = hashlib.sha256(open(sc.S1R_CKPT, "rb").read()).hexdigest()
    assert sha == sc.S1R_CKPT_SHA256, "S1R checkpoint hash mismatch"

    zeros = np.zeros((1, 512), dtype=np.float32)
    zeroshot = smod_base.load_model(device)
    emb_base = smod_base.embed_q1(zeroshot)
    del zeroshot
    torch.cuda.empty_cache()

    # ---------- 1. S1R inference parity vs stored sealed outputs ----------
    # Mirrors S1R's 07_sealed_test invocation exactly: full-recording mix +
    # group bounds + exact 10-s chunk overlap-add. In-process recomputation is
    # bit-identical; the stored Sep-14 outputs differ by <=~0.08 dB due to
    # cross-process kernel-selection drift (driver/cuDNN), so acceptance is
    # metric-level: |dRMS| < 0.1 dB and waveform corr >= 0.999.
    s1r = smod_base.load_student_from_zero_shot(device, sc.S1R_CKPT)
    stored = json.load(open(STORED, encoding="utf-8"))["rows"]
    mix, _sr = sf.read(os.path.join(SEALED_DIR, "sealed_32k_mono.wav"), dtype="float32")
    rows = []
    for row in stored:
        gid = row["group"]
        t0, t1 = row["t"]
        a, b = int(t0 * rc.SR), int(t1 * rc.SR)
        st_new = si.ola_separate(s1r, mix, a, b, emb_base, zeros, device, student=True)
        st_ref = sf.read(os.path.join(SEALED_DIR, f"{gid}_s1r.wav"), dtype="float32")[0]
        rms_new, rms_ref = rc.rms_db(st_new), rc.rms_db(st_ref)
        n = min(len(st_new), len(st_ref))
        corr = rc.wf_corr(st_new[:n], st_ref[:n])
        rows.append({"group": gid, "rms_ref_db": round(rms_ref, 3),
                     "rms_new_db": round(rms_new, 3),
                     "rms_abs_diff_db": round(abs(rms_ref - rms_new), 4),
                     "waveform_corr": round(corr, 6),
                     "pass": bool(abs(rms_ref - rms_new) < 0.1 and corr >= 0.999)})
        print(f'parity {gid}: rms {rms_ref:.3f} -> {rms_new:.3f} corr {corr:.5f} '
              f'{"PASS" if rows[-1]["pass"] else "FAIL"}')
    parity_ok = all(r["pass"] for r in rows)
    del s1r, mix
    torch.cuda.empty_cache()

    # ---------- 2/3. initial parity + adapter gradient ----------
    model = smod.build_model(device)
    n_train = sum(p.numel() for p in smod.trainable_parameters(model))
    n_total = sum(p.numel() for p in model.parameters())

    split = sc.load_split()
    pairs = json.load(open(os.path.join(sc.S2A_PRIVATE, "S2A_SPLIT.private.json"),
                           encoding="utf-8"))["pairs"]
    rmap = {p["recording_id"]: p for p in pairs}
    dev_ref = [r for r in rmap.values() if r["s2a_split"] == "DEV"]
    rec = next(r for r in split["recordings"] if r["recording_id"] == dev_ref[0]["recording_id"])
    x = rc.rec_audio(rec)
    w0 = 45.0
    v = rc.build_views(x, w0)
    chunk = v["views"]["canonical"]
    entry = dev_ref[0]
    ref32k = sc.ref_audio32k(entry)
    m_seg, cov = sc.ref_segment(entry, ref32k, w0 + 4.0, w0 + 14.0)
    m_seg = m_seg[:rc.CHUNK]
    wrong_entry = next(r for r in rmap.values()
                       if r["song_id"] != entry["song_id"] and r["s2a_split"] == "TEST")
    m_wrong, _ = sc.ref_segment(wrong_entry, sc.ref_audio32k(wrong_entry),
                                0.0, 10.0)
    m_wrong = m_wrong[:rc.CHUNK]

    smod.set_training_mode(model, training=False)
    e_pos = emb_base
    e_neg = zeros

    def fwd(ref):
        arr = np.asarray(chunk, dtype=np.float32)
        mx = float(np.max(np.abs(arr)))
        scale_back = 1.0
        xt = torch.tensor(arr, device=device).unsqueeze(0)
        rt = torch.tensor(np.asarray(ref, dtype=np.float32), device=device).unsqueeze(0)
        if mx > 1:
            xt = xt * (0.9 / mx)
            scale_back = mx / 0.9
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                _mask, pred = smod.train_forward(model, xt, rt, e_pos, e_neg)
        return (pred.float().detach().squeeze(0) * scale_back).cpu().numpy()

    # S1R single-chunk forward through the OFFICIAL wrapper on the same weights
    # (model.base_model IS the verified S1R checkpoint; in-process equivalence
    # of my replicated decoder path vs this wrapper is exact).
    arr = np.asarray(chunk, dtype=np.float32)
    mx = float(np.max(np.abs(arr)))
    xt0 = torch.tensor(arr, device=device).unsqueeze(0)
    scale_back = 1.0
    if mx > 1:
        xt0 = xt0 * (0.9 / mx)
        scale_back = mx / 0.9
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            _m, _p, _ = __import__("clapsep_train_lib").train_forward(
                model.base_model, xt0, emb_base, zeros)
    s1r_out = (_p.float().detach().squeeze(0) * scale_back).cpu().numpy()

    outs = {"correct": fwd(m_seg), "zero": fwd(np.zeros(rc.CHUNK, dtype=np.float32)),
            "wrong": fwd(m_wrong)}
    init = {}
    for k, y in outs.items():
        n = min(len(y), len(s1r_out))
        d = float(np.max(np.abs(y[:n] - s1r_out[:n])))
        init[k] = {"max_abs_diff_vs_s1r": d, "pass": bool(d < 1e-3)}
        print(f'init-parity {k}: max|d|={d:.2e} {"PASS" if d < 1e-3 else "FAIL"}')

    # gradient: injection-invariance path must reach the adapter gate
    model.train()
    smod.set_training_mode(model, training=True)
    arr = np.asarray(chunk, dtype=np.float32)
    mx = float(np.max(np.abs(arr)))
    xt = torch.tensor(arr, device=device).unsqueeze(0)
    if mx > 1:
        xt = xt * (0.9 / mx)
    mt = torch.tensor(m_seg, device=device).unsqueeze(0)
    x_aug = xt + 0.25 * mt
    _mask, pred_aug = smod.train_forward(model, x_aug, mt, e_pos, e_neg)
    loss = torch.nn.functional.l1_loss(pred_aug.squeeze(0), xt.squeeze(0))
    loss.backward()
    g_out = float(model.adapter.out.weight.grad.abs().sum())
    g_conv1 = float(model.adapter.conv1.weight.grad.abs().sum())
    g_head = float(next(model.base_model.decoder_model.mask_net.parameters()).grad.abs().sum())
    frozen_ok = True
    for n_, p in model.named_parameters():
        if not n_.startswith(smod.TRAINABLE_PREFIXES):
            frozen_ok = frozen_ok and (p.grad is None or float(p.grad.abs().sum()) == 0.0)
        else:
            frozen_ok = frozen_ok and (p.grad is not None)
    print(f"grads: out_head={g_out:.3e} conv1={g_conv1:.3e} mask_head={g_head:.3e} "
          f"frozen_clean={frozen_ok}")

    out = {
        "s1r_ckpt_sha256": sha,
        "sealed_parity": {"pass": parity_ok, "rows": rows},
        "init_parity_vs_s1r": init,
        "adapter_gradient": {"out_head_abs_sum": g_out, "conv1_abs_sum": g_conv1,
                             "mask_head_abs_sum": g_head, "frozen_clean": bool(frozen_ok)},
        "trainable_params": n_train, "total_params": n_total,
        "trainable_percent": round(100.0 * n_train / n_total, 2),
        "all_pass": bool(parity_ok and all(v["pass"] for v in init.values())
                         and g_out > 0 and frozen_ok),
    }
    sc.save_json(os.path.join(sc.S2A, "logs", "04_parity_check.json"), out)
    print("PARITY CHECK:", "PASS" if out["all_pass"] else "FAIL")


if __name__ == "__main__":
    main()
