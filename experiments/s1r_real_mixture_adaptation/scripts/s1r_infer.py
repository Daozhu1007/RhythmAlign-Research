# S1R shared full-clip inference: EXACT 10-s chunk overlap-add (0.5 hann),
# identical geometry to S1W's 11_real_test protocol, applied IDENTICALLY to
# zero-shot and S1R student (S1W FAILURE_CASES section 2: exact chunks are the
# trained distribution; zero-shot is robust to both protocols).
import os
import sys
import numpy as np

S1R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(S1R, "scripts"))
import s1r_common as rc  # noqa: E402

CHUNK = 320000


def ola_separate(model, mix32, start, end, emb, zeros, device, student=False):
    """Overlap-add separation of mix32[start:end] with exact 10-s chunks.

    student=False -> frozen zero-shot via the official r3 path.
    student=True  -> S1R student via the grad-free official-protocol forward
    (same peak-0.9 rescale-and-undo inside each chunk).
    """
    hop = CHUNK // 2
    s = max(0, start)
    n = end - start
    total = ((n + hop - 1) // hop) * hop + hop
    e = min(len(mix32), s + total)
    while e - s < n:
        s = max(0, s - hop)
    region = mix32[s:e]
    window = np.hanning(CHUNK)
    acc = np.zeros(len(region), dtype=np.float64)
    wsum = np.zeros(len(region), dtype=np.float64)
    for i in range(0, len(region) - CHUNK + 1, hop):
        chunk = region[i:i + CHUNK]
        if student:
            import torch
            import s1r_model as sm

            arr = np.asarray(chunk, dtype=np.float32)
            max_val = float(np.max(np.abs(arr)))
            scale_back = 1.0
            x = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
            if max_val > 1:
                x = x * (0.9 / max_val)
                scale_back = max_val / 0.9
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    _mask, pred = sm.train_forward(model, x, emb, zeros)
            y = (pred.float().detach().squeeze(0).cpu().numpy() * scale_back).astype(np.float32)
        else:
            import clapsep_lib

            y = clapsep_lib.separate(model, chunk, emb, zeros, device)
        acc[i:i + CHUNK] += y * window
        wsum[i:i + CHUNK] += window
    wsum[wsum < 1e-6] = 1.0
    out = (acc / wsum).astype(np.float32)
    return out[(start - s):(start - s) + n]
