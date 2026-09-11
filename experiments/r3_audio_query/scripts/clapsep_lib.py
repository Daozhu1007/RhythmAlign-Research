# R3 shared CLAPSep loader - official Space app.py path, vendored model/*.py,
# runner-side compat only (no vendor file modified):
#   1. torch>=2.6 weights_only=False (official ckpts carry numpy scalars)
#   2. drop obsolete "position_ids" buffers for strict state_dict loads
#      (same known-old-CLAP-ckpt issue R2 hit with the vendored AudioSep encoder)
import os
import sys

R3 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
THIRD_PARTY = os.path.join(R3, "third_party", "CLAPSep")
CKPT_DIR = os.path.join(R3, "checkpoints")

BEST_MODEL = os.path.join(CKPT_DIR, "best_model.ckpt")
CLAP_CKPT = os.path.join(CKPT_DIR, "music_audioset_epoch_15_esc_90.14.pt")

MODEL_CONFIG = {"lan_embed_dim": 1024, "depths": [1, 1, 1, 1], "embed_dim": 128,
                "encoder_embed_dim": 128, "phase": False, "spec_factor": 8,
                "d_attn": 640, "n_masker_layer": 3, "conv": False}


def load_clapsep(device):
    sys.path.insert(0, THIRD_PARTY)
    import torch

    _orig_load = torch.load

    def _load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _load_compat

    _orig_lsd = torch.nn.Module.load_state_dict

    def _lsd_compat(self, state_dict, strict=True, **kwargs):
        if isinstance(state_dict, dict):
            state_dict = {k: v for k, v in state_dict.items() if "position_ids" not in k}
        return _orig_lsd(self, state_dict, strict=strict, **kwargs)

    torch.nn.Module.load_state_dict = _lsd_compat

    from model.CLAPSep import CLAPSep

    model = CLAPSep(MODEL_CONFIG, CLAP_CKPT).to(device)
    ckpt = torch.load(BEST_MODEL, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def embed_audio_query(model, wav_path):
    """Official audio-query path (app.py): get_audio_embedding_from_filelist."""
    import numpy as np

    emb = model.clap_model.get_audio_embedding_from_filelist([wav_path])
    return np.asarray(emb)


def separate(model, mixture32k, embed_pos, embed_neg, device):
    """Official chunked inference (app.py), scaled back to the input domain.

    app.py rescales the mixture to peak 0.9 when |max|>1; we run that protocol,
    then multiply the result by max/0.9 so residual = original_mixture - target
    stays in the original domain.
    """
    import torch

    x = torch.tensor(mixture32k, dtype=torch.float32)
    n_orig = len(x)
    max_value = torch.max(torch.abs(x))
    scale_back = 1.0
    if max_value > 1:
        x = x * (0.9 / max_value)
        scale_back = float(max_value / 0.9)

    pad = (320000 - (n_orig % 320000)) if n_orig % 320000 != 0 else 0
    x = torch.nn.functional.pad(x, (0, pad))

    import numpy as np

    e_pos = torch.tensor(np.asarray(embed_pos), dtype=torch.float32, device=device)
    e_neg = torch.tensor(np.asarray(embed_neg), dtype=torch.float32, device=device)

    chunks = torch.chunk(x, dim=0, chunks=len(x) // 320000)
    outs = []
    for chunk in chunks:
        with torch.no_grad():
            outs.append(model.inference_from_data(chunk.unsqueeze(0).to(device), e_pos, e_neg))
    sep = torch.concat(outs, dim=1).squeeze().cpu().numpy()[:n_orig]
    return (sep * scale_back).astype(np.float32)
