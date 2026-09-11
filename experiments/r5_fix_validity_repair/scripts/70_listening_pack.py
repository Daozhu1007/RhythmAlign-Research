# R5-FIX PART 7 - blind listening pack for the project owner.
# Six Holdout-D final mixes, each exactly 18 s, identical gain convention
# (0.5 * pristine aligned reference + 0.5 * stem; music-only at the same 0.5),
# NO loudness normalisation. Randomised order with a FIXED seed; the mapping is
# stored ONLY in listening_key.json (owner must not open it before rating).
import os
import random
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixenv as fx

SEED = 20260909


def main():
    raw = sf.read(os.path.join(fx.R56_OUT, "holdoutD_raw.wav"), dtype="float64", always_2d=True)[0]
    ref = sf.read(os.path.join(fx.R56_WORK, "holdoutD_ref_warp.wav"), dtype="float64", always_2d=True)[0][3 * fx.SR:21 * fx.SR]
    n = len(raw)

    def load(p):
        x = sf.read(p, dtype="float64", always_2d=True)[0]
        assert len(x) == n, p
        return x

    candidates = {
        # corrected E0 + C1 final mix (new)
        "E0_C1_fixed_final": sf.read(os.path.join(fx.FIX_OUT, "holdoutD_E0_C1_fixed_final_mix.wav"),
                                     dtype="float64", always_2d=True)[0],
        # OLD defective E0 + C1 final mix (historical, unchanged)
        "E0_C1_old_final": sf.read(os.path.join(fx.R56_OUT, "holdoutD_e0_C1_final_mix.wav"),
                                   dtype="float64", always_2d=True)[0],
        # corrected oracle + C1 final mix (new)
        "oracle_C1_fixed_final": sf.read(os.path.join(fx.FIX_OUT, "holdoutD_oracle_C1_fixed_final_mix.wav"),
                                         dtype="float64", always_2d=True)[0],
        # OLD defective oracle + C1 final mix (historical, unchanged)
        "oracle_C1_old_final": sf.read(os.path.join(fx.R56_OUT, "holdoutD_oracle_C1_final_mix.wav"),
                                       dtype="float64", always_2d=True)[0],
        # current/basic baseline: gating disabled (0.5 music + 0.5 raw)
        "baseline_raw_final": 0.5 * ref + 0.5 * raw,
        # music-only anchor (same final gain convention)
        "music_only": 0.5 * ref,
    }
    descs = {
        "E0_C1_fixed_final": "R5.5 E0 自动时序 + 修正版 C1，最终混音",
        "E0_C1_old_final": "R5.5 E0 自动时序 + 历史（缺陷）C1，最终混音",
        "oracle_C1_fixed_final": "Holdout-D oracle 时序 + 修正版 C1，最终混音",
        "oracle_C1_old_final": "Holdout-D oracle 时序 + 历史（缺陷）C1，最终混音",
        "baseline_raw_final": "当前基线（门控关闭）：0.5 对齐音乐 + 0.5 原始录音",
        "music_only": "纯对齐音乐锚点（0.5 增益）",
    }

    names = list(candidates)
    rng = random.Random(SEED)
    order = names[:]
    rng.shuffle(order)

    key = {"phase": "R5-FIX blind listening pack (Holdout-D, 18 s each)",
           "seed": SEED,
           "warning": "评分完成前不要打开本文件。",
           "files": {}}
    for i, name in enumerate(order, start=1):
        out = os.path.join(fx.FIX_LISTEN, f"L{i:02d}.wav")
        sf.write(out, candidates[name].astype(np.float32), fx.SR, subtype="FLOAT")
        key["files"][f"L{i:02d}.wav"] = {"identity": name, "description": descs[name]}

    fx.save_json(os.path.join(fx.FIX_LISTEN, "listening_key.json"), key)
    for f in sorted(os.listdir(fx.FIX_LISTEN)):
        print(" ", f)


if __name__ == "__main__":
    main()
