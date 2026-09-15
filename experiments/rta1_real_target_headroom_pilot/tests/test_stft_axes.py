# STFT axis-correctness tests (RTA-0 audit follow-up).
#
# The historical S1R loss (s1r_losses.frame_band_energy) treated S.size(1) of
# a 1-D-input torch.stft as frequency; for such inputs it is TIME. These tests
# assert the RTA1 helpers cannot reproduce that failure mode.
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import rta1_lib as rl  # noqa: E402


def test_stft_shape_for_1d_input_is_freq_time():
    import torch

    x = np.random.default_rng(0).standard_normal(32000).astype(np.float32)
    S = rl.stft(x, n_fft=1024, hop=256)
    assert S.shape == (513, 126), tuple(S.shape)  # (freq, time) — NOT (time, freq)
    assert S.size(0) == 1024 // 2 + 1


def test_bin_frequencies_match_frequency_axis():
    import torch

    freqs = rl.bin_frequencies(1024, 32000)
    assert freqs.shape[0] == 513
    assert abs(float(freqs[0])) < 1e-6
    assert abs(float(freqs[-1]) - 16000.0) < 1e-6
    # bin k of the STFT (axis 0) corresponds to freqs[k]
    x = np.zeros(32000, np.float32)
    x[1000] = 1.0  # impulsive -> flat spectrum; check a pure tone instead:
    t = np.arange(32000) / 32000
    tone = (0.5 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    S = rl.stft(tone, 1024, 256)
    k = int(torch.abs(S).mean(dim=1).argmax())
    assert abs(float(freqs[k]) - 3000.0) < 200.0, (k, float(freqs[k]))


def test_band_profile_selects_frequency_not_time():
    """A 3 kHz tone must have nearly all its energy inside the 2-9 kHz band;
    a 300 Hz tone almost none. Under the historical axis bug (mask over time
    frames), both would show the same arbitrary middle-time 'band'."""
    sr = rl.SR
    t = np.arange(4 * sr) / sr
    tone_3k = (0.5 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    tone_300 = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    share_3k = rl.band_profile_db(tone_3k, sr=sr)[1]
    share_300 = rl.band_profile_db(tone_300, sr=sr)[1]
    assert share_3k > 0.95, share_3k
    assert share_300 < 0.01, share_300


def test_band_profile_rejects_impossible_band():
    try:
        rl.band_profile_db(np.zeros(32000, np.float32), lo_hz=20000, hi_hz=31000)
    except AssertionError:
        return
    raise AssertionError("empty band must raise, not return silence silently")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("test_stft_axes: ALL PASS")
