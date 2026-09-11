# R5.6 Holdout D - step 5: write the VIDEO-FIRST BLIND oracle.
# Event identity and times come from pass-1 (10 fps sweep) + pass-2 (30 fps
# sweep) + pass-3 (60 fps strips); audio candidates were consulted LAST and
# only for gap-checking (11 additions verified visually, 19 rejections).
# The oracle is NOT a superset of any detector output; the frozen R5.6
# detector has NOT run on this window as of this file's timestamp.
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common56 import R56_OUT, save_json

# (t_video, type, area, confidence, evidence)
E = [
    (0.00, "press", "NW-N", "med", "text fresh at window start, L hand pressing NW"),
    (0.50, "press", "N-NW", "high", "fresh yellow text burst at NW, sparkle"),
    (0.70, "press", "N", "med", "pink ring center, chevron trail N, R hand top"),
    (0.87, "press", "W", "med", "pink text at W, L hand W press posture"),
    (1.00, "press", "N", "med", "pink ring center-left, R hand top press"),
    (1.20, "press", "N", "med", "R hand press N, posture change"),
    (1.37, "press", "W", "med", "yellow text at W edge"),
    (1.54, "press", "N", "low", "press posture shift at N, no fresh text (audio gap-check add)"),
    (1.70, "press", "SW", "med", "pink ring SW, L press SW"),
    (1.90, "press", "S", "med", "orange sparkle bursts at S"),
    (2.20, "press", "N", "med", "chevrons brighten, both hands N"),
    (2.44, "press", "SE", "med", "R hand lands SE, fresh yellow star + ring (audio gap-check add)"),
    (2.53, "press", "SE", "med", "R hand SE press posture"),
    (2.70, "press", "W-SW", "med", "L press W, R press SE"),
    (2.87, "press", "W", "med", "yellow text at W"),
    (2.97, "press", "N", "high", "fresh yellow text at N, both hands top"),
    (3.20, "press", "NW-N", "med", "fresh starburst + R press top (strip: text fresh 3.20-3.23)"),
    (3.50, "press", "W", "med", "L press W on yellow slide bar, pink ring left"),
    (3.70, "slide", "SW", "med", "cyan chevron fan center, pink ring SW"),
    (3.88, "press", "NE", "med", "R press NE, text NE"),
    (3.97, "press", "W", "high", "L press W, fresh PERFECT text + star (audio gap-check add)"),
    (4.27, "press", "NW", "med", "occluded press NW: ring at 4.20 + fresh text 4.30 (merged)"),
    (4.53, "press", "N", "high", "fresh yellow text under hand at N, L press (strip)"),
    (4.70, "press", "N", "med", "R hand N press posture"),
    (4.90, "press", "N", "med", "yellow text at N, chevrons"),
    (5.01, "press", "N", "high", "double PERFECT text at N, R press (audio gap-check add)"),
    (5.20, "press", "N-NW", "med", "R hand down at N"),
    (5.53, "press", "N", "med", "R hand N press posture, chevron trail"),
    (5.73, "press", "SE", "med", "pink ring SE, press burst"),
    (5.87, "press", "N", "med", "orange sparkle burst center"),
    (8.03, "press", "E", "med", "fresh yellow ring at E under R hand (strip; rest ends)"),
    (8.21, "press", "SE", "high", "R press SE, fresh PERFECT text at SE (audio gap-check add)"),
    (8.37, "press", "W", "med", "L hand up W press"),
    (8.53, "press", "W", "med", "press on yellow slide bar W, R hand up N"),
    (8.70, "press", "W", "med", "yellow text at W"),
    (8.90, "press", "N", "med", "cyan chevron trail N appears, presses continue"),
    (9.00, "press", "N", "high", "fresh yellow text at N, both hands top"),
    (9.20, "press", "N", "med", "L arm crossing, press continues"),
    (9.37, "press", "N", "low", "pink ring under occluding arm (strip)"),
    (9.57, "press", "SE", "med", "R hand SE press, chevron fan trail"),
    (9.70, "press", "W", "high", "fresh yellow text at W, L press W"),
    (9.87, "press", "SW", "med", "yellow slide loop + gear SW"),
    (10.00, "press", "NW", "high", "fresh yellow text at NW, both hands top"),
    (10.20, "press", "E", "high", "fresh yellow text, R press E (strip)"),
    (10.33, "press", "NW", "high", "L press NW, fresh text (audio gap-check add)"),
    (10.50, "slide", "W", "med", "yellow chevron band W, L press-traverse"),
    (10.70, "press", "N", "med", "R press top, yellow band center"),
    (10.88, "press", "N", "med", "yellow text at N"),
    (11.00, "press", "N", "high", "fresh CRITICAL-style text at N, R press"),
    (11.20, "press", "NW", "med", "yellow bar slide left, L hand NW"),
    (11.40, "press", "N", "med", "pink text N"),
    (11.50, "press", "E", "med", "R press E, yellow ring + text NE (strip)"),
    (11.65, "press", "E", "med", "R press E, fresh text + gear (audio gap-check add)"),
    (11.87, "press", "SW", "med", "cyan chevron fan bottom, L press SW"),
    (12.03, "press", "E", "med", "R hand E press posture"),
    (12.20, "slide", "SW", "med", "yellow chevron band diagonal, L press-traverse SW"),
    (12.34, "press", "N", "high", "double top press, fresh text burst + stars (audio gap-check add)"),
    (12.53, "press", "W", "med", "yellow slide bar W + press"),
    (12.73, "press", "SE", "med", "R press SE posture, chevrons"),
    (12.90, "press", "SW", "med", "L press SW, yellow elements bottom"),
    (13.20, "slide", "W-SW", "med", "yellow chevron band lower-left, L traverse"),
    (13.37, "press", "S", "med", "both hands low pressing, yellow band center"),
    (13.53, "press", "W", "med", "pink ring W, L press"),
    (13.90, "press", "S", "med", "yellow slide loop bottom, R press S"),
    (14.20, "press", "NW", "high", "fresh yellow text NW, R press N-NW"),
    (14.30, "press", "W", "high", "L press W, fresh PERFECT text at W (audio gap-check add)"),
    (14.40, "press", "W", "med", "L press W, yellow cross effect"),
    (14.57, "press", "NE", "med", "R hand NE press, yellow loop right"),
    (14.87, "press", "SE", "med", "yellow loop bottom-right, R press SE, gear effects"),
    (15.00, "press", "E", "high", "fresh orange gear at E, R press"),
    (15.20, "slide", "NW", "med", "yellow arc trace NW + orange gear"),
    (15.30, "press", "NE", "high", "R press NE, fresh PERFECT text + gear (audio gap-check add)"),
    (15.53, "press", "NE", "med", "cyan chevron trail center, R press NE"),
    (15.63, "press", "SE", "high", "R press SE, fresh yellow text at E-SE (audio gap-check add)"),
    (15.87, "press", "SW", "med", "cyan chevron fan bottom-left, L press SW"),
    (16.00, "press", "NW", "high", "fresh yellow text at NW, L press"),
    (16.20, "press", "N", "med", "both hands top, pink star center"),
    (16.40, "press", "N", "med", "pink ring + text, R press N"),
    (16.57, "press", "W", "med", "cyan chevron fan bottom, L press W, pink ring SE"),
    (16.87, "slide", "SW", "med", "cyan chevron fan expands, L press-traverse SW"),
    (17.03, "press", "N", "med", "cyan fan trail, R hand top press"),
    (17.23, "press", "N", "med", "R press N posture"),
    (17.37, "press", "W", "med", "L press W, effects fading"),
    (17.53, "press", "E", "high", "fresh yellow text at E, press E"),
    (17.70, "press", "E", "med", "yellow bar + text E, gear effects"),
    (17.90, "press", "NW", "high", "fresh yellow text at NW, L press (window ends mid-play)"),
]


def main():
    assert len(E) == len({round(t, 3) for t, *_ in E})
    ts = [t for t, *_ in E]
    assert all(b > a for a, b in zip(ts, ts[1:])), "times must be sorted/unique"
    events = [{"id": f"d{i:02d}", "t_video": round(t, 3), "status": "ok",
               "type": typ, "area": area, "confidence": conf, "evidence": ev}
              for i, (t, typ, area, conf, ev) in enumerate(E)]
    doc = {
        "experiment": "R5.6 Holdout-D oracle (VIDEO-FIRST blind annotation)",
        "source_video": "outputs/holdoutD_video.mp4 (handcam-native 60.0-78.0 s)",
        "video_timeline": {"fps": 60.04, "duration_s": 18.0,
                           "t_video": "seconds from holdout start (= handcam-native minus 60.0)"},
        "annotation_method": {
            "pass1": "10 fps contact sheets, 8 sheets x 25 tiles, whole 18 s seen",
            "pass2": "30 fps contact sheets, 18 sheets x 30 tiles (frame-level "
                     "judgment-text / hit-effect timing)",
            "pass3": "60 fps 5-frame zoom strips at 8 ambiguous/occluded moments",
            "pass4_last": "audio-comb candidates (R4 family, med+4MAD, 55 ms merge) used "
                          "ONLY as a gap check: 107 audio candidates; 18 dense-segment gaps "
                          "each verified on a 60 fps strip -> 11 confirmed additions, 6 "
                          "rejections, 1 merge; 13 rest-segment onsets REJECTED because "
                          "video shows hands off the glass (music leakage through residual)",
            "video_first": "the candidate universe comes from vision, not from any audio "
                           "detector; audio could only ADD after visual verification",
            "blind": "no detector prediction was viewed before this file was written; the "
                     "frozen R5.6 detector has not been run on holdout-D",
        },
        "annotation_completed_utc": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frozen_detector_at": "2026-09-09 13:13:37",
        "events": events,
        "regions": [
            {"id": "R1", "t0": 0.0, "t1": 6.0,
             "desc": "dense continuous two-hand play: alternating top/side presses, "
                     "chevron slides, heavy arm-crossing occlusion ~2.0-2.2 and 4.1-4.5",
             "confidence": "high"},
            {"id": "G1", "t0": 6.0, "t1": 8.0,
             "desc": "REST/HOVER: glass nearly empty, both hands parked at the bottom "
                     "bezel; a yellow slide-ribbon note display advances WITHOUT a "
                     "tracing hand (strip-verified) - the 13 audio onsets in this span "
                     "are music leakage, NOT contacts",
             "confidence": "high"},
            {"id": "R2", "t0": 8.0, "t1": 18.0,
             "desc": "dense continuous play: bottom/side presses, yellow slide bands, "
                     "cyan chevron fan slides (~16.5-17.4), double presses, occlusion "
                     "~9.3, 11.5, 13.1; ends mid-play at window end",
             "confidence": "high"},
        ],
        "notes": [
            "86 contacts, 0 rest events inside R regions; G1 declared from video, "
            "not from detector state.",
            "Judgment text appears 0-2 frames after physical contact; strip center "
            "frame = candidate time.",
            "Known limitation: visually-silent sub-events inside 90-120 ms bursts "
            "may still be missing (video-first cannot see them either); audio "
            "gap-check only added strip-verified events.",
        ],
    }
    save_json(os.path.join(R56_OUT, "holdoutD_oracle_blind.json"), doc)
    print(f"holdoutD_oracle_blind.json: {len(events)} contacts, "
          f"written {doc['annotation_completed_utc']} (frozen detector not yet run)")


if __name__ == "__main__":
    main()
