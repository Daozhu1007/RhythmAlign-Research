# S1R losses (protocol sections 21-27). Declared a priori.
#
# L = 1.00*teacher_anchor + 0.75*real_context_consistency + 0.25*scale_equivariance
#   + 1.00*anti_collapse + 0.10*pretrained_weight_anchor
#
# All audio-domain terms operate on the shared aligned central 6 s real crop.
# The teacher is FROZEN; its outputs are pseudo-label/anchor evidence, never
# ground truth (never called true stems).
import torch

W_ANCHOR = 1.00
W_CONSIST = 0.75
W_SCALE = 0.25
W_COLLAPSE = 1.00
W_WEIGHT = 0.10

MR_WINDOWS = (256, 1024, 2048)

# anti-collapse guard (protocol section 24)
COLLAPSE_HINGE_DB = -6.0      # student may sit at most 6 dB under teacher before penalty
COLLAPSE_PENALTY_DB = 1.0     # quadratic-in-dB weight beyond the hinge
TRANSIENT_GUARD_DB = -4.5     # transient-band guard midpoint of the 3-6 dB suggestion
TRANSIENT_FRAC = 0.20         # top-energy teacher frames count as transient neighborhoods


def mr_stft_logmag(pred, target):
    """Multi-resolution log10-magnitude L1 (S1W-verified formulation)."""
    loss = 0.0
    for win in MR_WINDOWS:
        hop = win // 4
        window = torch.hann_window(win, device=pred.device)
        P = torch.stft(pred, win, hop, window=window, return_complex=True).abs()
        T = torch.stft(target, win, hop, window=window, return_complex=True).abs()
        loss = loss + (torch.log10(P + 1e-5) - torch.log10(T + 1e-5)).abs().mean()
    return loss / len(MR_WINDOWS)


def distill(pred, target):
    """Preservation-aware teacher-anchor distance: waveform L1 + MR-STFT log-mag."""
    l_wave = (pred - target).abs().mean()
    l_mr = mr_stft_logmag(pred, target)
    return l_wave + 0.5 * l_mr, {"l_wave": l_wave.detach(), "l_mr": l_mr.detach()}


def consistency(pred_a, pred_b):
    """Cross-view student consistency: waveform L1 + MR-STFT log-mag."""
    l_wave = (pred_a - pred_b).abs().mean()
    l_mr = mr_stft_logmag(pred_a, pred_b)
    return l_wave + 0.5 * l_mr, {"l_wave": l_wave.detach(), "l_mr": l_mr.detach()}


def frame_band_energy(x, n_fft=1024, hop=320):
    """Per-frame 2-9 kHz band energy (transient/click band)."""
    S = torch.stft(x, n_fft, hop, window=torch.hann_window(n_fft, device=x.device),
                   return_complex=True).abs() ** 2
    freqs = torch.linspace(0, 16000, S.size(1), device=x.device)
    band = S[:, (freqs >= 2000) & (freqs <= 9000)].sum(dim=1)  # (frames,)
    return 10 * torch.log10(band + 1e-9)


def anti_collapse(pred, teacher):
    """Hinge guard against the S1W failure mode (wholesale suppression).

    dB-domain hinge on total energy plus a transient-band floor on the top-energy
    teacher frames. A GUARD only - it never rewards making output louder.
    """
    rms_p = torch.sqrt(torch.mean(pred ** 2) + 1e-12)
    rms_t = torch.sqrt(torch.mean(teacher ** 2) + 1e-12)
    ratio_db = 20 * torch.log10(rms_p / rms_t + 1e-9)
    deficit = torch.relu(COLLAPSE_HINGE_DB - ratio_db)
    l_energy = COLLAPSE_PENALTY_DB * deficit ** 2

    band_p = frame_band_energy(pred)
    band_t = frame_band_energy(teacher)
    k = max(1, int(TRANSIENT_FRAC * band_t.numel()))
    idx = torch.topk(band_t, k).indices
    band_deficit = torch.relu(TRANSIENT_GUARD_DB - (band_p[idx] - band_t[idx]))
    l_transient = (band_deficit ** 2).mean()
    return l_energy + l_transient, {
        "ratio_db": ratio_db.detach(),
        "l_energy": l_energy.detach(),
        "l_transient": l_transient.detach(),
    }


def weight_anchor(model, theta0):
    """L2-SP: ||theta - theta_zero_shot||^2 over trainable weights (section 26)."""
    total = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad:
            total = total + ((p - theta0[n]) ** 2).sum()
    return total


def rms(x):
    return torch.sqrt(torch.mean(x ** 2) + 1e-12)
