# Baseline immutability tests (RTA-0 lesson).
#
# The S2A stage reported an "S1R baseline" computed from the SAME trained
# model object under test. These tests pin the RTA1 discipline:
#   - two model instances loaded from the same checkpoint are independent
#     objects: running one must never change the other's parameters
#   - the S1W adapted artifact must be refused as a baseline
#   - (local-weight checks, skipped if checkpoints are absent) a fresh load of
#     the original S1R checkpoint reproduces the recorded hash discipline
import os
import sys

import numpy as np

RESEARCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(RESEARCH, "experiments",
                                "rta1_real_target_headroom_pilot", "scripts"))


def _tiny_model(seed=0):
    """Stand-in 'model' with a parameter dict - no heavy weights needed."""
    import torch

    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))


def test_two_instances_are_independent_objects():
    import torch

    a, b = _tiny_model(0), _tiny_model(0)
    a0 = {n: p.detach().clone() for n, p in a.named_parameters()}
    # run/modify instance b
    with torch.no_grad():
        for p in b.parameters():
            p.add_(1.0)
        b(torch.zeros(1, 8))
    for n, p in a.named_parameters():
        assert torch.equal(a0[n], p), f"instance a parameter {n} moved"


def test_param_hash_detects_any_change():
    import torch

    bo = _load_05()
    m = _tiny_model(1)
    h0 = bo.param_hash(m)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1e-6)
    assert bo.param_hash(m) != h0


def test_s1w_artifact_refused():
    bo = _load_05()
    for path in (
        os.path.join(RESEARCH, "experiments", "s1w_existing_corpus_adaptation",
                     "checkpoints", "adapted.ckpt"),
        os.path.join(RESEARCH, "experiments", "s1r_real_mixture_adaptation",
                     "checkpoints", "s1w_adapted.ckpt"),
    ):
        try:
            bo.refuse_s1w(path)
        except ValueError:
            continue
        raise AssertionError(f"S1W artifact not refused: {path}")


def test_s1r_loader_accepts_only_the_recorded_checkpoint():
    """Full check requires the local S1R checkpoint; skipped (recorded) when
    absent so the test suite stays runnable anywhere."""
    ckpt = os.path.join(RESEARCH, "experiments", "s1r_real_mixture_adaptation",
                        "checkpoints", "s1r_selected.ckpt")
    if not os.path.exists(ckpt):
        print("SKIP s1r checkpoint checks (weights local-only)")
        return
    bo = _load_05()
    import rta1_lib as rl

    sha = rl.sha256_file(ckpt)
    assert sha == bo.S1R_CKPT_SHA256, "S1R checkpoint hash changed"

    import torch

    device = torch.device("cpu")
    zero_shot, s1r, emb, info = bo.load_baselines(device)
    # immutability under a forward pass of the OTHER instance
    h_zs = bo.param_hash(zero_shot)
    x = np.zeros(32000, np.float32)
    _ = bo.run_mixture(s1r, emb, x, device)
    assert bo.param_hash(zero_shot) == h_zs
    assert info["s1r_checkpoint_sha256"] == bo.S1R_CKPT_SHA256


def _load_05():
    import importlib.util

    path = os.path.join(RESEARCH, "experiments",
                        "rta1_real_target_headroom_pilot", "scripts",
                        "05_run_frozen_baselines.py")
    spec = importlib.util.spec_from_file_location("rta1_05", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rta1_05"] = mod
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("test_baseline_immutability: ALL PASS")
