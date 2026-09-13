# S1R model scope + training forward (protocol sections 17-19).
#
# Student initializes from the EXACT frozen zero-shot CLAPSep weights (never from
# the S1W adapted checkpoint - that is a negative artifact). Minimal trainable
# surface, option B of protocol section 18: final separation/mask head
# (decoder_model.mask_net) + upper decoder layers (decoder_model.layers[3] finest
# stage, decoder_model.skip[3] final skip transform, decoder_model.inverse_patch_embed).
# Frozen: CLAP text/query encoder, audio_branch (+LoRA), film, layers[0..2],
# skip[0..2], spec_norm BatchNorm running state (eval mode).
import os
import sys
import numpy as np
import torch

S1R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
S1W = os.path.abspath(os.path.join(S1R, "..", "s1w_existing_corpus_adaptation"))
sys.path.insert(0, os.path.join(S1W, "scripts"))
import clapsep_train_lib as ctl  # noqa: E402  (S1W shared wrapper, read-only)

TRAINABLE_PREFIXES = (
    "decoder_model.mask_net.",
    "decoder_model.layers.3.",
    "decoder_model.skip.3.",
    "decoder_model.inverse_patch_embed.",
)

Q1_WAV = ctl.Q1_WAV


def load_model(device):
    return ctl.load_model(device)


def embed_q1(model):
    return ctl.embed_q1(model)


def trainable_param_names(model):
    return [n for n, p in model.named_parameters() if p.requires_grad]


def set_trainable_scope(model):
    """Zero-shot base; only the minimal S1R surface trains (protocol section 18)."""
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if n.startswith(TRAINABLE_PREFIXES):
            p.requires_grad = True
    model.clap_model.eval()
    model.audio_branch.eval()
    model.stft.eval()
    model.istft.eval()
    return model


def set_training_mode(model, training=True):
    """decoder subscope trains; spec_norm BN stays eval (running state frozen)."""
    model.decoder_model.train(training)
    model.decoder_model.spec_norm.eval()
    model.clap_model.eval()
    model.audio_branch.eval()
    return model


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


@torch.enable_grad()
def train_forward(model, mix32k, embed_pos, embed_neg):
    """Official inference path with grad; frozen feature extraction under no_grad.

    Delegates to the S1W wrapper (hash-verified against the official path,
    checkpoint-reload-verified in S1W). Returns (mask, pred) with pred in the
    input domain, shape (B, T).
    """
    _mask, pred, _ = ctl.train_forward(model, mix32k, embed_pos, embed_neg)
    return _mask, pred


def snapshot_zero_shot_state(model):
    """theta_0 copies of the trainable weights for the L2-SP weight anchor."""
    return {n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad}


def load_student_from_zero_shot(device, ckpt_path=None):
    """Student == frozen zero-shot weights; optionally load an S1R checkpoint."""
    model = load_model(device)
    set_trainable_scope(model)
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"], strict=True)
    return model
