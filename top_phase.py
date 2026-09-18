"""
LoL VOD - Top Notification Event Counter
-----------------------------------------
Reads a LoL VOD, splits it into early/mid/late game phases, and OCRs only
the "Top Notification" ROI. Because a notification banner (e.g. "First
Blood", "Double Kill", "Baron Nashor") stays on screen for several frames,
raw per-frame OCR would count the SAME event many times. This script:

  1. OCRs every Nth frame (FRAME_SKIP) instead of every single frame, for
     speed.
  2. Uses fuzzy text similarity (difflib) to decide whether a new OCR
     reading is a CONTINUATION of the currently-visible notification, or a
     genuinely NEW event -> only new/changed events increment a count.
  3. Uses fuzzy similarity again to merge near-duplicate OCR readings of
     the same real-world text (OCR noise) into one distinct/canonical
     event label, so "Double Kill" and "Doub1e Ki11" aren't counted as two
     different kinds of events.
  4. Locks each active event to the phase (early/mid/late) it STARTED in,
     and keeps it there even if the game clock crosses a phase boundary
     while that same notification is still on screen. This avoids an
     event getting split or double-counted across two phases at the edge.
     New-event phase assignment itself is smoothed with a short
     majority-vote window so single-sample flicker right at the boundary
     doesn't cause an abrupt phase switch either.

No per-frame logging - only the final summary + JSON are printed/written.
"""

import cv2
import easyocr
import json
import os
from collections import deque, Counter
from difflib import SequenceMatcher

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
#VIDEO_PATH = r'D:\League of Legend\LOL_2min.mp4'
#OUTPUT_JSON = r'D:\League of Legend\LOL_2min_top_notification_ocr.json'
VIDEO_PATH = r"C:\Users\gdeep\Downloads\LOL_36.mp4"
OUTPUT_JSON = r"C:\Users\gdeep\Downloads\LOL_36_top_notification_ocr.json"

# ROI format: name -> (x, y, width, height)
ROIS = {
    "Top Notification": (765, 95, 350, 100),
}

# Standard LoL phase marks, in seconds (in-game clock; assumes video starts
# at game time 0:00, i.e. champ select / loading screen already trimmed off)
EARLY_MID_MARK_SEC = 14 * 60   # 14:00
MID_LATE_MARK_SEC = 25 * 60    # 25:00

# If the video is shorter than the marks above, fall back to proportional
# thirds of the actual video length instead. Set True to always force this.
FORCE_PROPORTIONAL = False

# Only run OCR every Nth raw video frame (perf).
FRAME_SKIP = 5

# How many sampled OCR reads to look back/forward across when deciding
# whether a detected notification is a continuation of the active one
# (rather than a new event), and how many consecutive "misses" (empty /
# different) are needed before an active event is considered truly over.
WINDOW_SIZE = 5

# Similarity ratio (0-1). Two OCR strings scoring >= this are "the same".
CONTINUATION_SIMILARITY = 0.70   # same on-screen event vs a new one
CANONICAL_MERGE_SIMILARITY = 0.75  # merge OCR-noise variants into 1 label

MIN_TEXT_LEN = 2  # ignore OCR noise shorter than this many characters

SHOW_PREVIEW = False  # live preview window (slows processing; keep off for full VODs)

# ----------------------------------------------------------------------


def text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def get_phase_boundaries(duration_sec: float):
    if not FORCE_PROPORTIONAL and duration_sec >= MID_LATE_MARK_SEC:
        return EARLY_MID_MARK_SEC, MID_LATE_MARK_SEC
    third = duration_sec / 3.0
    return third, 2 * third


def get_phase_for_time(t_sec: float, early_end: float, mid_end: float) -> str:
    if t_sec < early_end:
        return "early_game"
    elif t_sec < mid_end:
        return "mid_game"
    return "late_game"


def format_timestamp(t_sec: float) -> str:
    m = int(t_sec // 60)
    s = int(t_sec % 60)
    return f"{m:02d}:{s:02d}"


class EventTracker:
    """
    Turns a stream of per-sample OCR readings for one ROI into a count of
    DISTINCT events per game phase, handling repeat-frame dedup, OCR-noise
    fuzzy merging, and smooth phase-boundary attribution.
    """

    def __init__(self, window_size, continuation_sim, canonical_sim):
        self.window_size = window_size
        self.continuation_sim = continuation_sim
        self.canonical_sim = canonical_sim

        self.active_text = None        # last raw OCR text of the active event
        self.active_canonical = None   # its canonical/distinct label
        self.misses_since_active = 0   # consecutive non-matching samples

        self.recent_phases = deque(maxlen=window_size)  # majority-vote smoothing
        self.canonical_texts = []      # all distinct labels seen so far
        self.counts = {
            "early_game": Counter(),
            "mid_game": Counter(),
            "late_game": Counter(),
        }

    def _smoothed_phase(self, current_phase):
        # Majority vote over the last `window_size` sampled phases smooths
        # out single-sample flicker right at a boundary crossing, instead
        # of switching phases the instant the clock ticks over.
        self.recent_phases.append(current_phase)
        return Counter(self.recent_phases).most_common(1)[0][0]

    def _match_canonical(self, text):
        best_match, best_score = None, 0.0
        for c in self.canonical_texts:
            score = text_similarity(text, c)
            if score > best_score:
                best_match, best_score = c, score
        if best_match is not None and best_score >= self.canonical_sim:
            return best_match
        self.canonical_texts.append(text)
        return text

    def observe(self, text: str, current_phase: str):
        text = text.strip()
        smoothed_phase = self._smoothed_phase(current_phase)

        if len(text) < MIN_TEXT_LEN:
            # Nothing on screen right now -> a "miss" for whatever was active
            if self.active_text is not None:
                self.misses_since_active += 1
                if self.misses_since_active >= self.window_size:
                    # Truly gone now; clear so the same text reappearing
                    # later counts as a brand new event.
                    self.active_text = None
                    self.active_canonical = None
                    self.misses_since_active = 0
            return

        is_continuation = self.active_text is not None and max(
            text_similarity(text, self.active_text),
            text_similarity(text, self.active_canonical or ""),
        ) >= self.continuation_sim

        if is_continuation:
            self.active_text = text
            self.misses_since_active = 0
            return

        # New / changed notification -> count it once, lock in its phase.
        canonical = self._match_canonical(text)
        self.counts[smoothed_phase][canonical] += 1

        self.active_text = text
        self.active_canonical = canonical
        self.misses_since_active = 0


def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Video not found: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Could not open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    early_end, mid_end = get_phase_boundaries(duration_sec)
    used_proportional = FORCE_PROPORTIONAL or duration_sec < MID_LATE_MARK_SEC

    reader = easyocr.Reader(["en"])

    trackers = {
        name: EventTracker(WINDOW_SIZE, CONTINUATION_SIMILARITY, CANONICAL_MERGE_SIMILARITY)
        for name in ROIS
    }

    if SHOW_PREVIEW:
        window = "Top Notification OCR"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_number = 0
    last_texts = {name: "" for name in ROIS}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % FRAME_SKIP == 0:
            t_sec = frame_number / fps
            current_phase = get_phase_for_time(t_sec, early_end, mid_end)

            for name, (x, y, w, h) in ROIS.items():
                roi_frame = frame[y:y + h, x:x + w]
                gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                ocr_results = reader.readtext(gray)
                text = " ".join(t for _, t, _ in ocr_results).strip()
                last_texts[name] = text
                trackers[name].observe(text, current_phase)

        if SHOW_PREVIEW:
            display = frame.copy()
            for name, (x, y, w, h) in ROIS.items():
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display, f"{name}: {last_texts[name]}", (x, max(25, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(window, display)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

        frame_number += 1

    cap.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    results = {
        "video_info": {
            "path": VIDEO_PATH,
            "duration_seconds": round(duration_sec, 2),
            "duration_timestamp": format_timestamp(duration_sec),
            "fps": fps,
            "total_frames": total_frames,
            "frame_skip": FRAME_SKIP,
            "phase_mode": "proportional" if used_proportional else "standard",
            "phase_boundaries": {
                "early_to_mid_sec": round(early_end, 2),
                "mid_to_late_sec": round(mid_end, 2),
            },
        },
        "early_game": {},
        "mid_game": {},
        "late_game": {},
        "distinct_events_found": {},
    }

    for name, tracker in trackers.items():
        for phase in ("early_game", "mid_game", "late_game"):
            results[phase][name] = dict(tracker.counts[phase])
        results["distinct_events_found"][name] = tracker.canonical_texts

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Duration: {format_timestamp(duration_sec)} | Phase mode: {results['video_info']['phase_mode']}")
    print(f"Early->Mid at {format_timestamp(early_end)} | Mid->Late at {format_timestamp(mid_end)}")
    print(f"Results written to: {OUTPUT_JSON}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()