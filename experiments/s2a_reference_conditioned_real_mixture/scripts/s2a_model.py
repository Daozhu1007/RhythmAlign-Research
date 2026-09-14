# S2A model: reference-conditioned real-mixture separation (protocol sections 13-15).
#
# Base = FROZEN selected S1R checkpoint (never the failed S1W artifact). The only
# new component is a minimal aligned-reference adapter:
#   delta = CNN(standardized log-mag(mix), standardized log-mag(ref))
#   mask  = sigmoid(S1R mask logits + gate * delta),  gate init 0, delta head init 0
# so AT INITIALIZATION the model is EXACTLY S1R (near-no-op, section 14) - no
# second catastrophic-forgetting event. Late (mask-logit) gated residual fusion,
# section 15's preferred design; no new backbone.
#
# Reference-input ablations (section 16) are first-class:
#   CORRECT_REFERENCE  -> aligned pristine segment
#   ZERO_REFERENCE     -> all-zero waveform (adapter receives standardized zeros)
#   WRONG_REFERENCE    -> pristine segment of a DIFFERENT held-out song
import os
import sys
import numpy as np
import torch
import torch.nn as nn

S2A = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
S1R = os.path.abspath(os.path.join(S2A, "..", "s1r_real_mixture_adaptation"))
sys.path.insert(0, os.path.join(S1R, "scripts"))
import s1r_model as sm  # noqa: E402  (frozen base model wrapper, read-only)
import s1r_common as rc  # noqa: E402

TRAINABLE_PREFIXES = (
    "adapter.",
    "base_model.decoder_model.mask_net.",
)


class RefAdapter(nn.Module):
    """Small TF-domain CNN: (mix_logmag, ref_logmag) -> residual mask logits.

    ~12k parameters. The output layer is ZERO-initialized, which makes the
    adapter an exact no-op at initialization AND is the protocol-section-15
    "zero-initialized residual gate" (the head weights grow from zero, so all
    adapter layers receive gradient after the first update - no deadlock).
    Standardization happens outside (per-chunk, scale-free).
    """

    def __init__(self, ch=24, mid=8):
        super().__init__()
        self.conv1 = nn.Conv2d(2, ch, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=5, padding=2)
        self.up1 = nn.ConvTranspose2d(ch, mid, kernel_size=4, stride=2, padding=1)
        self.out = nn.ConvTranspose2d(mid, 1, kernel_size=5, padding=2)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, T, F):
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.conv2(h))
        h = torch.relu(self.up1(h))
        d = self.out(h)
        return d[:, :, :T, :F]


def _standardize(logmag, ref_is_zero=None):
    """Per-chunk whitening over (T, F); a all-zero ref standardizes to zeros."""
    if ref_is_zero:
        return torch.zeros_like(logmag)
    mu = logmag.mean()
    sd = logmag.std()
    return (logmag - mu) / (sd + 1e-6)


class S2AModel(nn.Module):
    """Frozen S1R base + trainable adapter + trainable final mask head.

    The base decoder forward is replicated (weights SHARED with the frozen base
    module - nothing duplicated) so the adapter can inject at mask logits.
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model      # full CLAPSep loaded from the S1R ckpt
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.adapter = RefAdapter()

    # ---- official feature capture (mirrors clapsep_train_lib.train_forward) ----
    def _features(self, mix32k):
        from torchlibrosa.stft import magphase

        model = self.base_model
        real, imag = model.stft(mix32k)
        mag, cos, sin = magphase(real, imag)
        feats = []

        def hook(_m, _i, out):
            feats.append(out)

        def hook_basic(_m, _i, out):
            feats.append(out[0])

        def spec_pad(_m, _i, out):
            return nn.functional.pad(out, (0, 0, 0, 1024 - out.size(2)))

        ha = model.audio_branch.patch_embed.register_forward_hook(hook)
        hl = [m.register_forward_hook(hook_basic) for m in model.audio_branch.layers]
        hs = model.audio_branch.spectrogram_extractor.register_forward_hook(spec_pad)
        try:
            with torch.no_grad():
                model.audio_branch({"waveform": model.resampler(mix32k)})
        finally:
            ha.remove()
            for m in hl:
                m.remove()
            hs.remove()
            del model.features[:]  # vendor persistent-hook leak guard (S1W note)
        feats = [f.detach() for f in feats]
        return mag, cos, sin, feats

    def _decoder_logits(self, mag, feats, embed_pos, embed_neg):
        """Replicates HTSAT_Decoder.forward UP TO mask_net (weights shared).

        Official ordering: features = [mag] + encoder feats; hidden_state =
        features[-1]; skip_features = features[:-1] reversed inside the decoder.
        """
        dec = self.base_model.decoder_model
        embed = torch.nn.functional.normalize(
            torch.cat([as_embed(embed_pos, mag.device),
                       as_embed(embed_neg, mag.device)], dim=-1), dim=-1)
        features = [mag] + feats
        skip_features = features[:-1][::-1]
        spec = skip_features[-1]
        h = dec.film(features[-1], embed)
        for layer, f, skip in zip(dec.layers, skip_features, dec.skip):
            h = layer(h)[0]
            h = skip(skip=f, embed=embed, x=h)
        h = dec.reshape_img2wav(dec.inverse_patch_embed(h)).squeeze(1)
        h = h[:, :spec.size(2), :]
        spec = spec.transpose(1, 3)
        spec = dec.spec_norm(spec).transpose(1, 3).squeeze(1)
        h = torch.concat([spec, h], dim=-1)
        return dec.mask_net(h), spec  # logits (B, T, 513); spec_normed mixture mag

    def forward(self, mix32k, ref32k, embed_pos, embed_neg):
        """Full forward. mix32k/ref32k: (B, T) @32 kHz; ref aligned (or zeros).

        Returns (mask, pred) exactly like the official path (pred input-domain).
        """
        mag, cos, sin, feats = self._features(mix32k)
        logits, _ = self._decoder_logits(mag, feats, embed_pos, embed_neg)

        _, _, T, Fdim = mag.shape
        with torch.no_grad():
            ref_real, ref_imag = self.base_model.stft(ref32k)
            ref_mag = torch.sqrt(ref_real ** 2 + ref_imag ** 2 + 1e-12)
        ref_is_zero = bool(float(ref32k.abs().sum()) == 0.0)
        mix_log = torch.log10(mag.squeeze(1) + 1e-5)
        ref_log = torch.log10(ref_mag.squeeze(1) + 1e-5)
        adapter_in = torch.stack([
            _standardize(mix_log),
            _standardize(ref_log, ref_is_zero),
        ], dim=1)
        delta = self.adapter(adapter_in, T, Fdim)
        mask = torch.sigmoid(logits.unsqueeze(1) + delta)
        pred = self.base_model.wav_reconstruct(mask, mag, cos, sin, length=mix32k.size(-1))
        return mask, pred


def embed_q1(model):
    return sm.embed_q1(model)


def build_model(device):
    """Load frozen S1R weights into the base and wrap with the adapter."""
    from s2a_common import S1R_CKPT

    model = sm.load_student_from_zero_shot(device, S1R_CKPT)
    sm.set_trainable_scope(model)
    s2a = S2AModel(model).to(device)
    set_trainable_scope(s2a)
    return s2a


def as_embed(e, device):
    if isinstance(e, torch.Tensor):
        return e.to(device=device, dtype=torch.float32)
    return torch.tensor(np.asarray(e), dtype=torch.float32, device=device)


def set_trainable_scope(model):
    """Adapter + final mask head train (section 15); everything else frozen."""
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if n.startswith(TRAINABLE_PREFIXES):
            p.requires_grad = True
    base = model.base_model
    base.clap_model.eval()
    base.audio_branch.eval()
    base.stft.eval()
    base.istft.eval()
    base.decoder_model.spec_norm.eval()
    return model


def set_training_mode(model, training=True):
    base = model.base_model
    base.decoder_model.train(training)
    base.decoder_model.spec_norm.eval()  # BN running state frozen (S1R convention)
    base.clap_model.eval()
    base.audio_branch.eval()
    return model


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


@torch.enable_grad()
def train_forward(model, mix32k, ref32k, embed_pos, embed_neg):
    """Official-protocol forward with grad (peak rescale handled by caller)."""
    _mask, pred = model(mix32k, ref32k, embed_pos, embed_neg)
    return _mask, pred


def snapshot_theta0(model):
    return {n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad}
