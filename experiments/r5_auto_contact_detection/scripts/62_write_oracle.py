# R5 Phase 8 - write the BLIND holdout oracle from the annotator's verdicts.
# Verdicts below were decided visually from pass-1 10fps sheets + pass-3 60fps
# zoom strips, BEFORE the frozen detector was run on this window.
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from r5_common import OUT, WORK, load_json, save_json

# candidate index -> (verdict, type, area, confidence, evidence note)
V = {
    0: ("contact", "slide", "N-NE", "med", "slide in progress + judgment text top"),
    1: ("contact", "slide", "NE-E", "med", "palm traverse on glass"),
    2: ("contact", "slide", "E-S", "med", "palm traverse on glass"),
    3: ("contact", "press", "SW", "med", "palm lands lower-left glass"),
    4: ("contact", "slide", "N-E", "med", "arm traverse + star effects + PERFECT text"),
    5: ("contact", "slide", "S", "med", "palm traverse + PERFECT text bottom"),
    6: ("contact", "press", "SW-S", "med", "palm press bottom-left + stars"),
    7: ("contact", "slide", "W-S", "med", "palm sweep bottom + PERFECT text"),
    8: ("contact", "press", "NW", "med", "palm press + star effect"),
    9: ("contact", "slide", "E", "med", "3-way fan slide, palm at E end"),
    10: ("contact", "press", "E-SE", "med", "palm pressing at fan end"),
    11: ("contact", "slide", "SE-S", "med", "palm sliding bottom-right"),
    12: ("contact", "press", "S", "med", "palm flat press bottom + text"),
    13: ("contact", "press", "S", "med", "palm press continues"),
    14: ("contact", "slide", "S-SE", "med", "sweep across bottom-right"),
    15: ("contact", "press", "E-SE", "med", "palm press right side"),
    16: ("contact", "touch", "N", "high", "gear break effect at N under hand"),
    17: ("contact", "press", "NW-N", "med", "press + orange element"),
    18: ("contact", "press", "N", "med", "press + gear/ring at N"),
    19: ("contact", "press", "NE", "med", "tap NE + pink ring"),
    20: ("contact", "press", "N", "med", "bezel tap N + pink ring on glass"),
    21: ("rest", "rest", "NE-E", "low", "hand on bezel between wedges, no new judgment"),
    22: ("contact", "press", "E", "med", "bezel tap E"),
    23: ("contact", "press", "E", "high", "fresh pink tap ring at E"),
    24: ("contact", "press", "E", "med", "press E continuing"),
    25: ("contact", "slide", "SE-S", "med", "palm sweep + slide text"),
    26: ("contact", "slide", "S", "med", "palm sweep bottom"),
    27: ("contact", "press", "S-SW", "med", "palm press + PERFECT text"),
    28: ("contact", "press", "W-SW", "med", "palm press + star effect"),
    29: ("contact", "slide", "W", "high", "palm sliding W along chevron trace"),
    30: ("contact", "press", "N-NW", "med", "press + pink ring"),
    31: ("contact", "press", "NE", "med", "press + rings near hand"),
    32: ("contact", "press", "N", "med", "orange tap ring at N"),
    33: ("contact", "press", "N", "high", "ring expanding at press point"),
    34: ("contact", "press", "NE", "med", "press at slide element"),
    35: ("contact", "slide", "NE", "high", "yellow flash arriving at NE slide end"),
    36: ("contact", "press", "NE-E", "med", "press glass at element"),
    37: ("contact", "press", "N-NE", "med", "press + chevrons"),
    38: ("contact", "press", "NE", "med", "press NE"),
    39: ("contact", "slide", "SE", "med", "arm sweep to SE + judgment text"),
    40: ("contact", "slide", "SW-S", "med", "palm slide + slide text"),
    41: ("contact", "press", "S-SW", "med", "palm press + pink ring"),
    42: ("contact", "press", "S", "med", "press + orange text"),
    43: ("contact", "slide", "W-S", "med", "sweep bottom blurred"),
    44: ("contact", "press", "S-SW", "med", "press + chevrons"),
    45: ("contact", "press", "SE", "med", "press + star sparks"),
    46: ("contact", "press", "SE", "high", "gear effect at press point"),
    47: ("contact", "press", "S", "high", "press + CRITICAL PERFECT text"),
    48: ("contact", "press", "S", "high", "press + text visible"),
    49: ("contact", "press", "S-SW", "med", "press continuing"),
    50: ("contact", "press", "S-SW", "med", "fingers pressing edge"),
    51: ("contact", "press", "S-SW", "med", "press + wedge LED lit"),
    52: ("contact", "press", "SE", "high", "gear effect + PERFECT at press"),
    53: ("contact", "press", "SE", "med", "press + gear effect"),
    54: ("contact", "slide", "S", "med", "sweep + yellow sparkles"),
    55: ("contact", "slide", "S", "med", "palm slide along chevron trace"),
    56: ("contact", "slide", "NE", "med", "fan slide in progress, hand at NE"),
    57: ("contact", "slide", "NE", "med", "fan slide continuing"),
    58: ("contact", "press", "S-SW", "high", "gear/star effects at press"),
    59: ("contact", "press", "S", "med", "press + stars + trace"),
    60: ("contact", "press", "S", "high", "gear effect at press point"),
    61: ("contact", "press", "S", "med", "press continuing, effect fading"),
    62: ("rest", "rest", "transition", "low", "hand moving between positions, no judgment evidence"),
    63: ("contact", "press", "S-SE", "med", "press + pink rings"),
    64: ("contact", "press", "SE", "med", "press + text"),
    65: ("contact", "slide", "S", "med", "sweep + ring appearing"),
    66: ("contact", "slide", "S-SE", "med", "sweep continuing"),
    67: ("contact", "press", "S", "med", "press + ring"),
    68: ("contact", "press", "SE", "med", "press + rings"),
    69: ("contact", "press", "SE", "high", "PERFECT text at press"),
    70: ("contact", "press", "SE-E", "med", "press + rings"),
    71: ("contact", "press", "SE", "high", "PERFECT text at press"),
    72: ("contact", "press", "NE-E", "med", "press + yellow element"),
    73: ("contact", "press", "N", "med", "press + orange rings"),
    74: ("contact", "press", "NE", "med", "press + yellow effects"),
    75: ("contact", "slide", "NE", "high", "flash at slide arrival"),
    76: ("contact", "slide", "E", "high", "flash at slide arrival"),
    77: ("contact", "slide", "S", "med", "sweep + PERFECT text"),
    78: ("contact", "press", "SE", "high", "press + PERFECT text"),
    79: ("contact", "press", "E-NE", "high", "rings + PERFECT text"),
    80: ("contact", "press", "NE", "med", "press + element"),
    81: ("contact", "press", "NW-N", "med", "L-hand tap at top / slide start"),
    82: ("contact", "press", "NW", "med", "fresh pink rings at NW"),
    83: ("contact", "press", "NW", "low", "possible tap, effects faint"),
    84: ("contact", "press", "NE", "low", "wedge LED lit, press likely"),
    85: ("contact", "press", "NE", "high", "CRITICAL PERFECT text fresh"),
    86: ("contact", "press", "E", "high", "yellow flash at press point"),
    87: ("contact", "slide", "N", "med", "chevron slide + judgment text"),
    88: ("contact", "slide", "N", "med", "L palm traverse across N"),
    89: ("contact", "slide", "N", "med", "L palm traverse continuing"),
    90: ("contact", "press", "N", "high", "gear effects at press"),
    91: ("contact", "press", "N-NE", "high", "rings + orange judgment text"),
    92: ("contact", "press", "NW-N", "high", "press + CRITICAL PERFECT text"),
    93: ("contact", "press", "NE", "med", "yellow glow at wedge, tap likely"),
    94: ("rest", "rest", "NE", "low", "no fresh effect; transient unexplained"),
}


def main():
    cands = load_json(os.path.join(WORK, "holdout_audio_candidates.json"))["candidates_s"]
    assert len(cands) == len(V)
    events = []
    for i, t in enumerate(cands):
        verdict, typ, area, conf, note = V[i]
        events.append({
            "id": f"h{i:02d}", "t_video": round(float(t), 3),
            "status": "ok" if verdict == "contact" else "rejected_noncontact",
            "type": typ if verdict == "contact" else "rest",
            "area": area, "confidence": conf, "evidence": note,
        })
    n_contact = sum(1 for e in events if e["status"] == "ok")
    doc = {
        "experiment": "R5 Phase 8 - BLIND holdout oracle",
        "source_video": "outputs/holdout_video.mp4 (handcam-native 84.0-102.0 s)",
        "video_timeline": {"fps": 60.04, "duration_s": 18.0,
                           "t_video": "seconds from holdout start (= handcam-native minus 84.0)"},
        "annotation_method": {
            "pass1": "10 fps contact sheets (7 sheets, whole 18 s seen)",
            "pass2": "30 fps contact sheets (19 sheets)",
            "pass3": "95 audio-onset candidates (R4-family comb envelope of music-removed "
                     "residual, med+4MAD, 55 ms merge), each verified on a 3-frame 60 fps "
                     "zoom strip; contact presence/type/area judged visually",
            "blind": "annotation completed BEFORE the frozen detector was run on this window; "
                     "no detector prediction was viewed prior to writing this file",
        },
        "annotation_completed_utc": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detector_frozen_at": "2026-09-09 02:35:00",
        "events": events,
        "regions": [
            {"id": "R1", "t0": 0.0, "t1": 3.9,
             "desc": "continuous: intro palm traverses + N/NE presses, dense", "confidence": "med"},
            {"id": "R2", "t0": 4.1, "t1": 10.7,
             "desc": "continuous: E/S/SE presses + slides, 3-way fan slide, near-continuous",
             "confidence": "med"},
            {"id": "G1", "t0": 10.75, "t1": 11.95,
             "desc": "sparse transition: hand repositions E->top, brief hover/rest", "confidence": "med"},
            {"id": "R3", "t0": 11.95, "t1": 18.0,
             "desc": "continuous: S/SE presses, N/NE slides + L-hand traverses, finale", "confidence": "med"},
        ],
        "notes": [
            "Vision determines identity; audio refines exact timing (Phase 9/10).",
            "status=rejected_noncontact marks audio transients judged NOT to be new player contacts.",
            "Judgment text appears 0-2 frames after physical contact; strip center frame = candidate time.",
        ],
    }
    save_json(os.path.join(OUT, "holdout_oracle_blind.json"), doc)
    print(f"holdout_oracle_blind.json written: {n_contact} contacts, "
          f"{len(events) - n_contact} rejected/rest, written at {doc['annotation_completed_utc']}")


if __name__ == "__main__":
    main()
