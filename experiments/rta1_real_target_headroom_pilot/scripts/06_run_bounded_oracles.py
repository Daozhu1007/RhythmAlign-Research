# RTA1 step 06 - bounded real-mask oracles + fixed-mask source-path analysis.
#
# The oracle is NOT a deployable model. It receives the true target offline
# and measures representational headroom of the bounded real-mask family:
#   1. BINWISE bounded real magnitude mask  M = clamp(|S_p|/(|S_y|+eps), 0, 1)
#   2. WAVEFORM-OPTIMIZED bounded real mask (projected gradient on the mask
#      parameters, bounds [0,1], waveform-SNR objective, fixed seed)
#   4. target roundtrip reconstruction ISTFT(STFT(p)) as the lossless floor
#
# For every mask, the SAME fixed mask is applied separately to p and n
# (source-path analysis): target damage vs nuisance leakage. A nonlinear
# separator is never rerun independently on the paths.
#
# --selftest validates the implementation on internally synthesized material
# (software validation only, never scientific evidence).
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
import rta1_lib as rl  # noqa: E402

N_FFT = 1024
HOP = 256
OPT_ITERS = 300
OPT_LR = 0.01
SEED = 20260915


# ------------------------------------------------------------ mask oracles --

def binwise_bounded_mask(p, y, n_fft=N_FFT, hop=HOP):
    """Classical bounded magnitude oracle from perfect target knowledge."""
    Sp = rl.stft(p, n_fft, hop)
    Sy = rl.stft(y, n_fft, hop)
    M = torch_clamp(Sp.abs() / (Sy.abs() + 1e-10), 0.0, 1.0)
    return M


def waveform_optimized_mask(y, p, n_fft=N_FFT, hop=HOP,
                            iters=OPT_ITERS, lr=OPT_LR, seed=SEED):
    """Free real mask (one parameter per TF bin, clamped to [0,1]) optimized
    directly for waveform SNR of the reconstruction. Bounds are enforced by
    clamping inside the objective (projected gradient)."""
    import torch

    torch.manual_seed(seed)
    Sy = rl.stft(y, n_fft, hop)
    M0 = binwise_bounded_mask(p, y, n_fft, hop)
    param = torch.logit(torch_clamp(M0, 0.02, 0.98)).clone().requires_grad_(True)
    pt = torch.as_tensor(p, dtype=torch.float32)
    opt = torch.optim.Adam([param], lr=lr)
    history = []
    for i in range(iters):
        opt.zero_grad()
        M = torch.clamp(param, 0.0, 1.0)
        y_hat = rl.istft(M * Sy, n_fft, hop, length=len(p))
        loss = ((y_hat - pt) ** 2).mean() / (pt ** 2).mean().clamp_min(1e-12)
        loss.backward()
        opt.step()
        if i % 50 == 0 or i == iters - 1:
            history.append({"iter": i, "rel_mse": float(loss.detach())})
    M = torch.clamp(param.detach(), 0.0, 1.0)
    assert float(M.min()) >= 0.0 and float(M.max()) <= 1.0
    return M, history


def torch_clamp(x, lo, hi):
    import torch

    return torch.clamp(torch.as_tensor(x, dtype=torch.float32), lo, hi)


# ------------------------------------------------- source-path decomposition --

def source_path_analysis(M, p, n, n_fft=N_FFT, hop=HOP):
    """Apply the SAME fixed mask separately to the target and nuisance paths.

    Returns p_hat_M, n_hat_M plus a linearity residual proving
    M*(p+n) == Mp + Mn to numeric tolerance.
    """
    Sp, Sn = rl.stft(p, n_fft, hop), rl.stft(n, n_fft, hop)
    Sy = Sp + Sn
    p_hat = rl.istft(M * Sp, n_fft, hop, length=len(p)).numpy()
    n_hat = rl.istft(M * Sn, n_fft, hop, length=len(n)).numpy()
    y_hat = rl.istft(M * Sy, n_fft, hop, length=len(p)).numpy()
    lin_res = float(np.max(np.abs((p_hat + n_hat) - y_hat))
                    / (np.max(np.abs(y_hat)) + 1e-12))
    return p_hat, n_hat, lin_res


# ------------------------------------------------------------------ metrics --

def snr_db(ref, est):
    ref, est = np.asarray(ref, np.float64), np.asarray(est, np.float64)
    return round(10.0 * np.log10((ref ** 2).sum()
                                 / (((ref - est) ** 2).sum() + 1e-20) + 1e-12), 3)


def nmse_db(ref, est):
    ref, est = np.asarray(ref, np.float64), np.asarray(est, np.float64)
    return round(10.0 * np.log10((((ref - est) ** 2).sum())
                                 / ((ref ** 2).sum() + 1e-20) + 1e-12), 3)


def nuisance_attenuation_db(n, n_hat):
    return round(rl.rms_db(np.asarray(n)) - rl.rms_db(np.asarray(n_hat)), 3)


def fixed_gain_no_clip(y_hat, ref, full_scale=1.0):
    """Single fixed global gain, chosen to maximize SNR subject to no clipping.
    Returns (gained signal, gain)."""
    denom = float((y_hat ** 2).sum())
    g = float((y_hat @ ref) / denom) if denom > 0 else 1.0
    peak = float(np.max(np.abs(y_hat))) + 1e-12
    g = min(g, full_scale / peak)
    if abs(g - 1.0) < 1e-6:
        return y_hat, 1.0
    gained = y_hat * g
    if snr_db(ref, gained) >= snr_db(ref, y_hat):
        return gained, round(g, 6)
    return y_hat, 1.0


def evaluate_oracle(name, M, p, n, y, n_fft=N_FFT, hop=HOP):
    y_hat = rl.istft(M * rl.stft(y, n_fft, hop), n_fft, hop,
                     length=len(p)).numpy()
    y_hat, gain = fixed_gain_no_clip(y_hat, p)
    p_hat, n_hat, lin_res = source_path_analysis(M, p, n, n_fft, hop)
    row = {
        "oracle": name,
        "recon_snr_db": snr_db(p, y_hat),
        "target_nmse_db": nmse_db(p, p_hat),
        "nuisance_attenuation_db": nuisance_attenuation_db(n, n_hat),
        "fixed_gain": gain,
        "mask_min": round(float(M.min()), 5),
        "mask_max": round(float(M.max()), 5),
        "mask_in_bounds": bool(float(M.min()) >= -1e-6
                               and float(M.max()) <= 1.0 + 1e-6),
        "sourcepath_linearity_maxrel": round(lin_res, 8),
    }
    assert row["mask_in_bounds"], row
    assert lin_res < 1e-4, f"fixed-mask linearity violated: {lin_res}"
    return row


def roundtrip_floor(p, n_fft=N_FFT, hop=HOP):
    p_rt = rl.istft(rl.stft(p, n_fft, hop), n_fft, hop, length=len(p)).numpy()
    return {"oracle": "target_roundtrip",
            "recon_snr_db": snr_db(p, p_rt),
            "target_nmse_db": nmse_db(p, p_rt)}


# ------------------------------------------------------------------- selftest --

def selftest():
    """Synthesize a known target + nuisance and verify the oracle family.
    TOOLING VALIDATION ONLY - not scientific evidence."""
    rng = np.random.default_rng(SEED)
    sr = rl.SR
    t = np.arange(int(4.0 * sr)) / sr
    # target: sparse 3 kHz transients + friction-like 5 kHz band noise
    p = np.zeros_like(t)
    for t0 in (0.3, 1.1, 1.9, 2.7, 3.4):
        idx = int(t0 * sr)
        env = np.exp(-np.arange(3000) / 800.0)
        p[idx:idx + 3000] += env * np.sin(2 * np.pi * 3000 * np.arange(3000) / sr)
    p[int(2.0 * sr):int(3.5 * sr)] += 0.02 * rng.standard_normal(int(1.5 * sr)) \
        * (1 + np.sin(2 * np.pi * 2.0 * t[int(2.0 * sr):int(3.5 * sr)]))
    # nuisance: low-frequency "music" (bass line + slow chord)
    n = (0.25 * np.sin(2 * np.pi * 110 * t)
         + 0.2 * np.sin(2 * np.pi * 220 * t + 0.7)
         + 0.05 * np.sin(2 * np.pi * 55 * t))
    y = p + n
    p, n, y = p.astype(np.float32), n.astype(np.float32), y.astype(np.float32)

    rows = [roundtrip_floor(p)]
    M1 = binwise_bounded_mask(p, y)
    rows.append(evaluate_oracle("binwise_bounded", M1, p, n, y))
    M2, hist = waveform_optimized_mask(y, p)
    rows.append(evaluate_oracle("waveform_optimized_bounded", M2, p, n, y))

    print(json.dumps(rows, indent=1))
    assert rows[0]["recon_snr_db"] > 40, "roundtrip must be near-lossless"
    assert rows[1]["recon_snr_db"] > 10, "binwise oracle should isolate LF music"
    assert rows[2]["recon_snr_db"] >= rows[1]["recon_snr_db"] - 0.1, \
        "optimized oracle should not lose to binwise init"
    assert rows[1]["nuisance_attenuation_db"] > 6
    print("SELFTEST OK (tooling validation only)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None, metavar="DIR",
                    help="directory of mixture wavs + p/n folders from recipes")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.run:
        print("nothing to do: pass --selftest and/or --run DIR")
        return 1
    print("real runs require captured material + mixture recipes (04); "
          "see ORACLE_PROTOCOL.md")
    return 1


if __name__ == "__main__":
    sys.exit(main())
