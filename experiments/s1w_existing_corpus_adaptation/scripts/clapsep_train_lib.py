# S1W shared CLAPSep wrapper.
#
# - Loads the official CLAPSep via R3's clapsep_lib (vendored model/*.py, read-only;
#   migrated research workspace copies hash-verified against S1E's RUN_MANIFEST).
# - Declares the S1W trainable scope: decoder_model ONLY (mask/output head + decoder).
#   Frozen: CLAP text/query encoder, audio_branch (incl. its LoRA weights),
#   spec_norm BatchNorm running state ("normalization state").
# - train_forward mirrors the official inference_from_data path with gradients,
#   audio_branch feature extraction under no_grad (fully frozen), detached features.
import os
import sys
import numpy as np
import torch

S1W = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R3_SCRIPTS = r"D:\Code\RhythmAlign-Research\experiments\r3_audio_query\scripts"
sys.path.insert(0, R3_SCRIPTS)
import clapsep_lib  # noqa: E402  (r3, read-only)

Q1_WAV = r"D:\Code\RhythmAlign-Research\experiments\r3_audio_query\queries\query_q1.wav"
CKPT_BEST = clapsep_lib.BEST_MODEL
CKPT_CLAP = clapsep_lib.CLAP_CKPT
CHUNK = 320000  # official 10 s @ 32 kHz


def load_model(device):
    return clapsep_lib.load_clapsep(device)


def embed_q1(model):
    return clapsep_lib.embed_audio_query(model, Q1_WAV)


def set_trainable_scope(model):
    """decoder_model trainable; everything else frozen; BN state frozen (eval)."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.decoder_model.parameters():
        p.requires_grad = True
    model.clap_model.eval()
    model.audio_branch.eval()
    model.stft.eval()
    model.istft.eval()
    return model


def set_training_mode(model, training=True):
    """decoder trains; spec_norm BN stays eval (running state frozen); frozen parts eval."""
    model.decoder_model.train(training)
    model.decoder_model.spec_norm.eval()
    model.clap_model.eval()
    model.audio_branch.eval()
    return model


def trainable_parameters(model):
    return [p for p in model.decoder_model.parameters() if p.requires_grad]


@torch.enable_grad()
def train_forward(model, mix32k, embed_pos, embed_neg):
    """Official inference path with grad; frozen feature extraction under no_grad.

    mix32k: (B, T<=320000) float32 tensor. embed_*: numpy (1,512) or tensors.
    Replicates CLAPSep.inference_from_data exactly: features list = [mag,
    patch_embed_out, layer1..4 outs]; hidden_state = last layer output;
    skip_features = [mag, patch, l1..l3]; decoder normalizes the concatenated
    (pos,neg) embed; wav_reconstruct applies relu(mag*mask) with mixture phase.
    Returns mask (B,1,T,F), pred (B,T) waveform estimate (input domain).
    """
    from torchlibrosa.stft import magphase

    real, imag = model.stft(mix32k)
    mag, cos, sin = magphase(real, imag)

    feats = []

    def hook(_m, _i, out):
        feats.append(out)

    def hook_basic(_m, _i, out):
        feats.append(out[0])

    def spec_pad(_m, _i, out):
        return torch.nn.functional.pad(out, (0, 0, 0, 1024 - out.size(2)))

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
        # the vendor __init__ installs PERSISTENT hooks appending to model.features
        # on every audio_branch call; inference_from_data clears them after each run
        # (del self.features[:]). We must do the same or features accumulate forever
        # (~32 MB/step leak). Our own local hooks already captured detached copies.
        del model.features[:]
    feats = [f.detach() for f in feats]

    features = [mag] + feats  # official ordering: mag first, then encoder features

    e_pos = as_embed(embed_pos, mix32k.device)
    e_neg = as_embed(embed_neg, mix32k.device)
    embed = torch.nn.functional.normalize(torch.cat([e_pos, e_neg], dim=-1), dim=-1)
    mask = model.decoder_model(hidden_state=features[-1],
                               skip_features=features[:-1], embed=embed)
    pred = model.wav_reconstruct(mask, mag, cos, sin, length=mix32k.size(-1))
    return mask, pred, None


def torchlibrosa_magphase(real, imag):
    """torchlibrosa.stft.magphase equivalent (mag clamp 1e-10)."""
    mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-12)
    mag = torch.clamp(mag, 1e-10, None)
    cos = real / mag
    sin = imag / mag
    return mag, cos, sin


def as_embed(e, device):
    if isinstance(e, torch.Tensor):
        return e.to(device=device, dtype=torch.float32)
    return torch.tensor(np.asarray(e), dtype=torch.float32, device=device)
