# R5.5 Phase 7 - write the BLIND Holdout-C oracle from the annotator's verdicts.
# Verdicts were decided visually from pass-1 10fps sheets + pass-2 30fps sheets
# + pass-3 60fps zoom strips, BEFORE the frozen R5.5 detector ran on this window.
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common55 import R55_OUT, R55_WORK, load_json, save_json

# candidate index -> (verdict, type, area, confidence, evidence note)
V = {
    0: ("contact", "press", "NW-N", "med", "L hand pressing top-left, chevron trace near"),
    1: ("contact", "press", "NW-N", "med", "L press top-left, fresh orange tap cubes at hand"),
    2: ("contact", "press", "N", "med", "both hands top, pink judgment text center"),
    3: ("contact", "slide", "W-S", "med", "L palm traversing bottom-left along chevrons"),
    4: ("contact", "slide", "SE-S", "med", "R palm on glass bottom-right, chevron fan NE"),
    5: ("contact", "press", "E", "med", "R hand at right bezel, pink ring fresh left-center"),
    6: ("contact", "press", "SE-S", "high", "yellow gear burst at press point"),
    7: ("contact", "press", "S-SE", "med", "press continues, yellow effect at press"),
    8: ("contact", "press", "S-SE", "high", "fresh warm PERFECT text at press"),
    9: ("contact", "press", "S-SE", "high", "orange hexagon gear + PERFECT text"),
    10: ("contact", "slide", "S-SW", "med", "R palm flat, slight traverse bottom"),
    11: ("contact", "press", "S", "med", "R press bottom, chevrons mid"),
    12: ("contact", "press", "S", "high", "fresh warm text at press point"),
    13: ("contact", "press", "N", "med", "both hands top, fresh star effect center"),
    14: ("contact", "slide", "E", "med", "R hand traverse at E, long chevron trail"),
    15: ("contact", "press", "S-SW", "high", "CRITICAL-style warm text fresh at S-SW"),
    16: ("contact", "press", "NE-E", "med", "R hand pressing NE, yellow loop element"),
    17: ("contact", "press", "E-NE", "med", "finger tap posture NE, pink ring center"),
    18: ("contact", "press", "E", "med", "re-press E, fresh small orange ring"),
    19: ("contact", "press", "E-SE", "high", "fresh pink tap ring at E under hand"),
    20: ("contact", "press", "SE", "med", "R press SE, pink ring nearby"),
    21: ("contact", "press", "S", "high", "fresh warm PERFECT text center at press"),
    22: ("contact", "press", "SW-W", "med", "L palm lands SW across body"),
    23: ("contact", "press", "SE-S", "high", "fresh orange gear at press point"),
    24: ("contact", "slide", "N", "med", "L hand on yellow slide band across top"),
    25: ("contact", "slide", "NW-N", "med", "L traverse top, pink text at W"),
    26: ("contact", "slide", "S-W", "med", "R arm sweep bottom, cyan ring center"),
    27: ("contact", "press", "SW", "high", "fresh CRITICAL yellow text at SW"),
    28: ("contact", "slide", "W", "med", "L palm sliding W along chevron trace"),
    29: ("contact", "press", "W", "med", "yellow slide loop element arrives at W hand"),
    30: ("contact", "slide", "W", "med", "arm sweep W with chevron trail"),
    31: ("contact", "press", "SE", "med", "R press SE, fresh star sparkles center-right"),
    32: ("contact", "press", "SE-E", "med", "press SE, pink star cluster fresh"),
    33: ("contact", "press", "W", "high", "warm PERFECT text fresh at W during hit"),
    34: ("contact", "press", "SW", "low", "fan-slide tick posture, chevron circle expanding"),
    35: ("contact", "press", "SW", "low", "fan-slide tick posture, circle continues"),
    36: ("contact", "press", "W", "high", "fresh pink star at center-right at press"),
    37: ("contact", "slide", "W", "med", "traverse W with pink starburst cluster"),
    38: ("contact", "press", "N", "med", "L press top-center, pink star center"),
    39: ("contact", "press", "N-NW", "high", "fresh orange hexagon gear + rings"),
    40: ("contact", "slide", "NW", "med", "fast L traverse to NW (motion blur)"),
    41: ("contact", "press", "N", "med", "R tap N, both hands top"),
    42: ("contact", "press", "E-NE", "high", "fresh pink star + orange sparkle at press"),
    43: ("contact", "press", "NE", "med", "press NE, pink star persists"),
    44: ("contact", "press", "NE", "med", "finger press NE, chevron trail mid"),
    45: ("contact", "press", "W", "med", "L press W, star sparkles center-left"),
    46: ("contact", "press", "W", "high", "fresh PERFECT text at W edge"),
    47: ("contact", "press", "W-SW", "med", "L press W-SW, fresh pink elements"),
    48: ("contact", "press", "N", "med", "L press N on big chevron fan circle"),
    49: ("contact", "slide", "S-SE", "med", "R arm sweep bottom, star sparkles SE"),
    50: ("contact", "press", "SW-W", "high", "fresh pink star at SW under hand"),
    51: ("contact", "press", "S+W", "med", "double flat-palm press bottom"),
    52: ("contact", "slide", "S-W", "med", "R arm traverse center-bottom (blur)"),
    53: ("contact", "slide", "SW", "med", "traverse continues, fresh pink star SW"),
    54: ("contact", "press", "S", "high", "double press, fresh pink ring at S center"),
    55: ("contact", "press", "S", "low", "press posture continues, ring persists"),
    56: ("contact", "slide", "S-SW", "high", "fresh warm text at SW during sweep"),
    57: ("contact", "slide", "S", "med", "traverse center, pink ring at S persists"),
    58: ("contact", "press", "N-NE", "med", "L finger press top, chevron trail"),
    59: ("contact", "slide", "S-W", "med", "R arm sweep bottom, yellow chevrons SW"),
    60: ("contact", "press", "N", "med", "L arm traverse top, pink star center-top"),
    61: ("contact", "press", "W-SW", "med", "L press SW, pink rings at S, warm text SW"),
    62: ("contact", "press", "SE", "high", "fresh pink ring + circle at SE press"),
    63: ("contact", "press", "NW-N", "med", "L press top-left, pink rings center"),
    64: ("contact", "press", "NE-E", "high", "double top posture, fresh PERFECT text W"),
    65: ("contact", "press", "E-NE", "high", "fresh pink ring at E under R hand"),
    66: ("contact", "slide", "SE-W", "med", "R arm sweep bottom (blur), yellow stars"),
    67: ("contact", "press", "SW", "high", "fresh pink ring at SW under L hand"),
}


def main():
    cands = load_json(os.path.join(R55_WORK, "holdoutC_audio_candidates.json"))["candidates_s"]
    assert len(cands) == len(V), (len(cands), len(V))
    events = []
    for i, t in enumerate(cands):
        verdict, typ, area, conf, note = V[i]
        events.append({
            "id": f"c{i:02d}", "t_video": round(float(t), 3),
            "status": "ok" if verdict == "contact" else "rejected_noncontact",
            "type": typ if verdict == "contact" else "rest",
            "area": area, "confidence": conf, "evidence": note,
        })
    n_contact = sum(1 for e in events if e["status"] == "ok")
    doc = {
        "experiment": "R5.5 Phase 7 - BLIND Holdout-C oracle",
        "source_video": "outputs/holdoutC_video.mp4 (handcam-native 40.0-58.0 s)",
        "video_timeline": {"fps": 60.04, "duration_s": 18.0,
                           "t_video": "seconds from holdout start (= handcam-native minus 40.0)"},
        "annotation_method": {
            "pass1": "10 fps contact sheets (7 sheets, whole 18 s seen)",
            "pass2": "30 fps contact sheets (19 sheets; spot re-check of marginal "
                     "candidates #34/#35 at 10.0-11.0 s and #55/#57 at 15-16 s)",
            "pass3": "68 audio-onset candidates (R4-family comb envelope of "
                     "music-removed residual, med+4MAD, 55 ms merge), each verified "
                     "on a 3-frame 60 fps zoom strip; contact presence/type/area "
                     "judged visually",
            "blind": "annotation completed BEFORE the frozen R5.5 detector was run "
                     "on this window; no detector prediction was viewed prior to "
                     "writing this file",
        },
        "annotation_completed_utc": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detector_frozen_at": "2026-09-09 03:51:51",
        "events": events,
        "regions": [
            {"id": "R1", "t0": 0.0, "t1": 18.0,
             "desc": "continuous two-hand play throughout: top/side/bottom presses, "
                     "yellow slide band N, cyan chevron fan circles, arm-sweep "
                     "slides, relative lulls ~7.6-8.1 and ~14.9-15.1 (hand never "
                     "leaves glass)", "confidence": "med"},
        ],
        "notes": [
            "Vision determines identity; audio refines exact timing (frozen pipeline).",
            "All 68 audio candidates judged as new player contacts (0 rejected): the "
            "window is dense-dominant by selection (mixed sparse/dense PROFILE was "
            "relative event spacing, not hand-off rests).",
            "Judgment text appears 0-2 frames after physical contact; strip center "
            "frame = candidate time.",
        ],
    }
    save_json(os.path.join(R55_OUT, "holdoutC_oracle_blind.json"), doc)
    print(f"holdoutC_oracle_blind.json written: {n_contact} contacts, "
          f"{len(events) - n_contact} rejected/rest, written at {doc['annotation_completed_utc']}")


if __name__ == "__main__":
    main()
