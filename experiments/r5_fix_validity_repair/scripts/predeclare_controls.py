# R5-FIX PART 5 (predeclaration) - control definitions frozen BEFORE any
# control score is computed or inspected. This file is written once and must
# not be edited afterwards; the evaluation script asserts it exists.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixenv as fx

PRE = {
    "declared_before_scoring": True,
    "date": "2026-09-09",
    "control_A_music_only": {
        "definition": "interaction stem = digital zeros; final mix = 0.5 * pristine aligned reference "
                      "(same final gain convention as every other final mix; no loudness normalisation)",
        "purpose": "shows how many event/onset metrics can be satisfied without any interaction stem",
    },
    "control_B_shifted_gate": {
        "definition": "circular (np.roll) shift of the CORRECTED gate envelope by a fixed amount; "
                      "duty cycle and envelope shape preserved exactly; interaction = shifted_env * raw",
        "shift_golden_s": 7.5,
        "shift_holdoutD_s": 9.0,
        "rationale": ("both shifts are exactly half the window length, fixed a priori: maximal "
                      "temporal displacement, NOT aligned with window boundaries or event grid, "
                      "and chosen without inspecting any resulting score"),
        "shift_samples": {"golden": int(7.5 * fx.SR), "holdoutD": int(9.0 * fx.SR)},
        "purpose": "tests whether correct temporal placement matters beyond exposing ~the same "
                   "amount of raw audio",
        "interpretation_warning": "one shifted control per window; do not overinterpret statistically",
    },
    "allowed_contact_support": {
        "definition": "union over ok oracle contacts of [t - 40 ms, t + 175 ms] = frozen C1 "
                      "pre/post (15/150 ms) expanded by the +/-25 ms A/V mapping uncertainty "
                      "recorded in R4 metadata",
    },
}


def main():
    p = os.path.join(fx.FIX_LOG, "predeclared_controls.json")
    assert not os.path.exists(p), "predeclared_controls.json already exists - DO NOT EDIT"
    fx.save_json(p, PRE)
    print("predeclared ->", p)


if __name__ == "__main__":
    main()
