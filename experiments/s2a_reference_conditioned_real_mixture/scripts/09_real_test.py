# S2A step 9 - HELD-OUT TEST (protocol sections 27-29).
#
# Runs ONLY after checkpoint/protocol freeze (08_selection). Five song-disjoint
# held-out recordings, 4 deterministic windows each. Methods: RAW, ZERO-SHOT,
# S1R (frozen base), S2A CORRECT_REFERENCE, S2A ZERO_REFERENCE, S2A
# WRONG_REFERENCE (wrong ref from a DIFFERENT TEST song than the one evaluated).
# Descriptive proxies only - no stems exist; causal ablations reported separately
# from quality claims.
import os
import sys
import json
import hashlib
import zlib
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(__file__))
import s2a_common as sc  # noqa: E402
import s2a_model as smod  # noqa: E402
import s2a_augment as sa  # noqa: E402
import s1r_common as rc  # noqa: E402
import s1r_model as s1m  # noqa: E402
import s1r_infer as si  # noqa: E402

N_WIN = 4
ALPHA_EVAL = 0.30
LEAD_S, TAIL_S, SPAN_S = 30.0, 15.0, 10.0


def pick_windows(entry, dur):
    """Deterministic spread of windows inside the REFERENCE-COVERED span."""
    ref32k = sc.ref_audio32k(entry)
    off = entry["alignment"]["offset_s"]
    dur_ref = len(ref32k) / rc.SR
    lo = max(0.0, off, LEAD_S)
    hi = min(dur - TAIL_S - SPAN_S, off + dur_ref - SPAN_S)
    assert hi > lo, f"no reference-covered span for {entry['recording_id']}"
    wins = []
    for frac in np.linspace(0.05, 0.95, N_WIN):
        w0 = lo + frac * (hi - lo)
        _, cov = sc.ref_segment(entry, ref32k, w0, w0 + SPAN_S)
        if cov < 0.9:
            raise AssertionError(f"low ref coverage at w0={w0} "
                                 f"({entry['recording_id']})")
        wins.append(round(float(w0), 2))
    return sorted(set(wins))[:N_WIN]


def band_db(x, lo, hi, sr=rc.SR):
    import librosa
    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    band = S[(freqs >= lo) & (freqs <= hi)].sum(axis=0)
    return float(10 * np.log10(band.mean() + 1e-12))


def ref_active_attenuation(raw, out, ref_seg, sr=rc.SR):
    """Reference-correlated spectral energy change (section 29): attenuation in
    ref-active TF cells vs ref-inactive cells, output relative to raw."""
    import librosa

    def cells(y):
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
        return S
    R = cells(ref_seg)
    if not np.any(R > 0):
        return 0.0, 0.0
    thr = np.quantile(R, 0.75)
    active = R >= thr
    A_raw, A_out = cells(raw), cells(out)
    n = min(A_raw.shape[1], A_out.shape[1], active.shape[1])
    act = active[:, :n]
    A_raw = A_raw[:, :n] ** 2
    A_out = A_out[:, :n] ** 2
    e_raw_a = A_raw[act].mean()
    e_out_a = A_out[act].mean()
    inact = ~act
    if not inact.any():
        return (round(float(10 * np.log10((e_out_a + 1e-12) / (e_raw_a + 1e-12))), 3),
                0.0)
    e_raw_i = A_raw[inact].mean()
    e_out_i = A_out[inact].mean()
    att_active = 10 * np.log10((e_out_a + 1e-12) / (e_raw_a + 1e-12))
    att_inactive = 10 * np.log10((e_out_i + 1e-12) / (e_raw_i + 1e-12))
    return round(float(att_active), 3), round(float(att_inactive), 3)


def main():
    device = torch.device("cuda")
    sel = json.load(open(os.path.join(sc.S2A, "logs", "08_selection.json"),
                         encoding="utf-8"))
    # Negative path: section-25 selection may legitimately fail (section-26 gate
    # REFERENCE_ADAPTER_IGNORED); the documented evaluation checkpoint is then
    # chosen by the recorded rationale, never by test-set tuning.
    ckpt_name = sel["selected"] or sel["evaluation_checkpoint"]
    assert ckpt_name, "no checkpoint available"
    ckpt_path = os.path.join(sc.S2A, "checkpoints", ckpt_name)
    sha = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()

    zeros = np.zeros((1, 512), dtype=np.float32)
    tmp = s1m.load_model(device)
    e_pos = s1m.embed_q1(tmp)
    del tmp
    torch.cuda.empty_cache()
    e_neg = zeros

    zeroshot = s1m.load_model(device)
    s2a = smod.build_model(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    s2a.load_state_dict(ck["model_state"], strict=True)
    smod.set_training_mode(s2a, training=False)

    pairs = json.load(open(os.path.join(sc.S2A_PRIVATE, "S2A_SPLIT.private.json"),
                           encoding="utf-8"))["pairs"]
    test = [p for p in pairs if p["s2a_split"] == "TEST"]
    split = sc.load_split()
    rec_of = {r["recording_id"]: r for r in split["recordings"]}

    rows = []
    for ti, entry in enumerate(test):
        rec = rec_of[entry["recording_id"]]
        x = rc.rec_audio(rec)
        ref32k = sc.ref_audio32k(entry)
        dur = rec["duration_s"]
        wrong = test[(ti + 1) % len(test)]
        ref_wrong = sc.ref_audio32k(wrong)
        for w0 in pick_windows(entry, dur):
            a, b = int(w0 * rc.SR), int((w0 + SPAN_S) * rc.SR)
            raw = x[a:b]
            m_can = sc.ref_segment(entry, ref32k, w0, w0 + SPAN_S)[0][:rc.CHUNK]
            m_wrong = sc.ref_segment(wrong, ref_wrong, 0.0, SPAN_S)[0][:rc.CHUNK]

            def s2a_ola(ref_seg_full):
                return _ola_s2a(s2a, x, a, b, entry, ref_seg_full, e_pos, e_neg,
                                device)

            y_corr = s2a_ola(ref32k)
            y_zero = _ola_s2a(s2a, x, a, b, None, None, e_pos, e_neg, device)
            y_wrong = _ola_s2a(s2a, x, a, b, wrong, ref_wrong, e_pos, e_neg, device)
            y_s1r = si.ola_separate(s2a.base_model, x, a, b, e_pos, e_neg, device,
                                    student=True)
            y_zs = si.ola_separate(zeroshot, x, a, b, e_pos, e_neg, device,
                                   student=False)

            # controlled injection triad (quantified causal evidence, section 26
            # logic on held-out data): ALL methods run the exact OLA protocol on
            # the same augmented region; target = S1R output on the UNMODIFIED
            # region (y_s1r, computed above).
            seed = zlib.crc32(f'{entry["recording_id"]}|{w0:.2f}'.encode())
            rng = np.random.default_rng(seed)
            x_aug_region, alpha = sa.make_augmented(raw, m_can, rng)
            xa_full = x.copy()
            xa_full[a:b] = x_aug_region
            t = y_s1r
            s1r_aug = si.ola_separate(s2a.base_model, xa_full, a, b, e_pos, e_neg,
                                      device, student=True)

            def resid(y, target):
                return round(float(rc.rms_db(y - target)), 3)

            r_s1r = resid(s1r_aug, t)
            r_corr = resid(_ola_s2a(s2a, xa_full, a, b, entry, ref32k,
                                    e_pos, e_neg, device), t)
            r_zero = resid(_ola_s2a(s2a, xa_full, a, b, None, None,
                                    e_pos, e_neg, device), t)
            r_wrong = resid(_ola_s2a(s2a, xa_full, a, b, wrong, ref_wrong,
                                     e_pos, e_neg, device), t)

            row = {
                "recording_id": entry["recording_id"], "song_id": entry["song_id"],
                "w0_s": w0, "alpha": round(float(alpha), 4),
                "rms_db": {"raw": round(rc.rms_db(raw), 2),
                           "zeroshot": round(rc.rms_db(y_zs), 2),
                           "s1r": round(rc.rms_db(y_s1r), 2),
                           "s2a_correct": round(rc.rms_db(y_corr), 2),
                           "s2a_zero": round(rc.rms_db(y_zero), 2),
                           "s2a_wrong": round(rc.rms_db(y_wrong), 2)},
                "click_ret_vs_raw_db": {
                    "s1r": round(band_db(y_s1r, 2000, 9000) - band_db(raw, 2000, 9000), 3),
                    "s2a_correct": round(band_db(y_corr, 2000, 9000) - band_db(raw, 2000, 9000), 3)},
                "ref_attenuation": {
                    "s2a_correct": ref_active_attenuation(raw, y_corr, m_can),
                    "s2a_zero": ref_active_attenuation(raw, y_zero, m_can),
                    "s2a_wrong": ref_active_attenuation(raw, y_wrong, m_can)},
                "injection_residual_db": {
                    "s1r_on_augmented": r_s1r,
                    "s2a_correct": r_corr, "s2a_zero": r_zero, "s2a_wrong": r_wrong},
                "suppression_adv_vs_s1r_db": round(r_s1r - r_corr, 3),
                "causal_gap_zero_db": round(r_zero - r_corr, 3),
                "causal_gap_wrong_db": round(r_wrong - r_corr, 3),
                "collapse_flags": {
                    "s2a_correct_below_s1r_6db": bool(rc.rms_db(y_corr) < rc.rms_db(y_s1r) - 6.0),
                    "s2a_wrong_below_s1r_6db": bool(rc.rms_db(y_wrong) < rc.rms_db(y_s1r) - 6.0)},
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        del ref32k

    # freeze record
    freeze = {
        "checkpoint": os.path.basename(ckpt_path), "checkpoint_sha256": sha,
        "selected_from": ckpt_name,
        "selection_status": "SECTION25_SELECTION_FAILED_SECTION26_REFERENCE_ADAPTER_IGNORED"
                            if not sel["selected"] else "selected",
        "protocol": {"chunk": "exact 10 s @ 32 kHz OLA hann 0.5",
                     "reference_alignment": "accepted per-recording offsets (03_data_gate)",
                     "reference_preprocessing": "frozen STFT mag, per-chunk standardize",
                     "wrong_reference": "different held-out TEST song than evaluated"},
        "no_tuning_after_freeze": True,
    }
    sc.save_json(os.path.join(sc.S2A, "logs", "09_real_test.json"),
                 {"freeze": freeze, "rows": rows})
    print("HELD-OUT TEST DONE")


def _ola_s2a(model, x, a, b, entry, ref_full, e_pos, e_neg, device):
    """OLA of the S2A model with per-chunk aligned reference (or zero/wrong)."""
    hop = rc.CHUNK // 2
    s = max(0, a)
    n = b - a
    total = ((n + hop - 1) // hop) * hop + hop
    e = min(len(x), s + total)
    while e - s < n:
        s = max(0, s - hop)
    region = x[s:e]
    window = np.hanning(rc.CHUNK)
    acc = np.zeros(len(region), dtype=np.float64)
    wsum = np.zeros(len(region), dtype=np.float64)
    for i in range(0, len(region) - rc.CHUNK + 1, hop):
        chunk = region[i:i + rc.CHUNK]
        t0 = (s + i) / rc.SR
        if entry is None:
            ref = np.zeros(rc.CHUNK, dtype=np.float32)
        else:
            ref, _cov = sc.ref_segment(entry, ref_full, t0, t0 + 10.0)
            ref = ref[:rc.CHUNK]
        arr = np.asarray(chunk, dtype=np.float32)
        mx = float(np.max(np.abs(arr)))
        scale_back = 1.0
        xt = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
        rt = torch.tensor(ref, dtype=torch.float32, device=device).unsqueeze(0)
        if mx > 1:
            xt = xt * (0.9 / mx)
            scale_back = mx / 0.9
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                _mask, pred = smod.train_forward(model, xt, rt, e_pos, e_neg)
        y = (pred.float().detach().squeeze(0) * scale_back).cpu().numpy()
        acc[i:i + rc.CHUNK] += y * window
        wsum[i:i + rc.CHUNK] += window
    wsum[wsum < 1e-6] = 1.0
    out = (acc / wsum).astype(np.float32)
    return out[(a - s):(a - s) + n]


if __name__ == "__main__":
    main()
