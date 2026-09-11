# R5 Phase 5 - dev audio reconstruction with the frozen C1 recipe.
# dev_auto_C1_*   : driven by automatic refined contacts
# dev_oracle_C1_* : driven by R4 oracle refined contacts (reproduction of R4 C1)
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import soundfile as sf

from r5_common import (R1_OUT, R1_WORK, R4_OUT, OUT, LOG, SR, load_json, save_json,
                       build_c1)

N = 15 * SR


def main():
    ref_full, _ = sf.read(os.path.join(R1_WORK, "ref_warp_fixed.wav"),
                          dtype="float64", always_2d=True)
    ref = ref_full[int(3.0 * SR):int(18.0 * SR)]
    assert len(ref) == N
    raw_path = os.path.join(R1_OUT, "golden_raw.wav")

    auto = load_json(os.path.join(OUT, "dev_contacts_auto_refined.json"))["events"]
    auto_t = [e["t_audio_refined"] for e in auto
              if e.get("refined_confidence") not in (None, "no_transient", "out_of_range")]
    d_auto = build_c1(raw_path, ref, auto_t,
                      os.path.join(OUT, "dev_auto_C1_interaction.wav"),
                      os.path.join(OUT, "dev_auto_C1_final_mix.wav"))
    print("auto C1:", len(auto_t), "events |", d_auto)

    orc = load_json(os.path.join(R4_OUT, "contacts_refined.json"))["events"]
    orc_t = [e["t_audio_refined"] for e in orc
             if e["refined_confidence"] != "no_transient" and e["type"] != "none"]
    d_orc = build_c1(raw_path, ref, orc_t,
                     os.path.join(OUT, "dev_oracle_C1_interaction.wav"),
                     os.path.join(OUT, "dev_oracle_C1_final_mix.wav"))
    print("oracle C1:", len(orc_t), "events |", d_orc)
    save_json({"auto": {"n_events": len(auto_t), **d_auto},
               "oracle": {"n_events": len(orc_t), **d_orc}},
              os.path.join(LOG, "dev_build_info.json"))


if __name__ == "__main__":
    main()
