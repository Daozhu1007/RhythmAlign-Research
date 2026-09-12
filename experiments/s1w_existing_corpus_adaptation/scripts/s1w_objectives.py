# S1W shared losses + DEV metrics (declared a priori; protocol section 26/31).
import torch
import numpy as np

PREEMPH = 0.95
MR_WINDOWS = [256, 1024, 2048]  # short: attack/transients; long: friction/decay texture
W_WAVE, W_MRSTFT, W_HF = 1.00, 0.50, 0.25


def preemph(x):
    return x - PREEMPH * torch.roll(x, 1, dims=-1)


def mr_stft_loss(pred, target):
    """Multi-resolution log-magnitude distance.

    Log scale is REQUIRED here: linear ratio normalization explodes on
    near-silent targets (hard negatives at -30 dB TNR have |s| rms ~0.01,
    giving linear-ratio losses in the hundreds that dominate and derail
    optimization). log10 with 1e-5 floor keeps all target levels comparable.
    """
    loss = 0.0
    for win in MR_WINDOWS:
        hop = win // 4
        window = torch.hann_window(win, device=pred.device)
        P = torch.stft(pred.squeeze(1), win, hop, window=window, return_complex=True).abs()
        T = torch.stft(target.squeeze(1), win, hop, window=window, return_complex=True).abs()
        loss = loss + (torch.log10(P + 1e-5) - torch.log10(T + 1e-5)).abs().mean()
    return loss / len(MR_WINDOWS)


def recon_loss(pred, target):
    """Preservation-aware loss (protocol section 26): waveform + MR-STFT + HF pre-emph."""
    p = pred.squeeze(1)
    t = target.squeeze(1)
    l_wave = (p - t).abs().mean()
    l_mr = mr_stft_loss(pred, target)
    l_hf = (preemph(p) - preemph(t)).abs().mean()
    return W_WAVE * l_wave + W_MRSTFT * l_mr + W_HF * l_hf, {
        "l_wave": l_wave.detach(), "l_mr": l_mr.detach(), "l_hf": l_hf.detach()}


def absent_loss(pred):
    """Target-absent: strongly penalize output energy."""
    p = pred.squeeze(1)
    l1 = p.abs().mean()
    l2 = (p ** 2).mean()
    return (l1 + l2).squeeze(), {"l_abs1": l1.detach(), "l_abs2": l2.detach()}


# ---------------- fixed-gain evaluation metrics (constructed mixtures: exact) ----

def rms(x, dim=-1):
    return torch.sqrt(torch.mean(x ** 2, dim=dim) + 1e-12)


@torch.no_grad()
def dev_metrics(pred, s, n=None, tnr_gain=None, spans=None):
    """Returns per-example metric dict. Everything fixed-gain; no normalization.

    target_distortion_db: 10log10(E(pred-s) on target spans / E(s) on target spans)
    leakage_ratio_db:     10log10(E(pred-s) everywhere / (gain^2 * E(n)))  [needs exact n]
                          0 dB = nuisance untouched; negative = suppressed
    hf_retention_db:      pre-emphasized energy ratio pred/s on target spans
    """
    p = pred.squeeze(1).float().cpu()
    t = s.squeeze(1).float().cpu()
    out = {}
    if spans:
        mask = torch.zeros_like(t)
        for a, b in spans:
            mask[..., int(a * 32000):int(b * 32000)] = 1.0
        if mask.sum() > 0:
            e_err = float(((p - t) ** 2 * mask).sum())
            e_sig = float(((t ** 2) * mask).sum() + 1e-12)
            out["target_distortion_db"] = 10 * np.log10(e_err / e_sig + 1e-12)
            e_hp_err = float(((preemph(p) - preemph(t)) ** 2 * mask).sum())
            e_hp_sig = float(((preemph(t)) ** 2 * mask).sum() + 1e-12)
            # 0 dB = crisp-content perfectly kept; negative = HF loss (muffling proxy)
            out["hf_retention_db"] = -10 * np.log10(e_hp_err / e_hp_sig + 1e-12)
    if n is not None:
        nn = n.squeeze(1).float().cpu()
        if tnr_gain is not None:
            e_err = float(((p - t) ** 2).sum())
            e_n = float(((nn * tnr_gain) ** 2).sum() + 1e-12)
            out["leakage_ratio_db"] = 10 * np.log10(e_err / e_n + 1e-12)
        else:
            out["output_rms_db"] = float(20 * np.log10(float(rms(p)) + 1e-12))
            out["input_rms_db"] = float(20 * np.log10(float(rms(nn)) + 1e-12))
    out["recon_l1"] = float((p - t).abs().mean())
    return out
