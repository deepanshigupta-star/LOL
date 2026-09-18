"""
LoL VOD Phase-Based OCR Reader
------------------------------
Reads a League of Legends VOD, determines its duration, splits the video into
three game phases (early / mid / late game) using LoL's standard time marks,
runs OCR on the configured ROIs at a sampling interval, and writes the results
into a JSON file with one dictionary per phase.

Standard phase marks used (in-game clock, not video time offset):
    Early game : 0:00  - 14:00
    Mid game   : 14:00 - 25:00
    Late game  : 25:00 - end

If the video is SHORTER than 25 minutes (e.g. a trimmed clip rather than a
full VOD), those fixed marks would collapse the phases, so the script
automatically falls back to splitting the video into proportional thirds
(33% / 33% / 33%). You can force proportional mode with FORCE_PROPORTIONAL.
"""

import cv2
import easyocr
import json
import os

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

VIDEO_PATH = r"C:\Users\gdeep\Downloads\LOL_36.mp4"
OUTPUT_JSON = r"C:\Users\gdeep\Downloads\LOL_36_phase_ocr.json"

# ROI format: name -> (x, y, width, height)
ROIS = {
    "Players Ability Count": (455, 955, 180, 125),
    "PlayerUI Skillup":      (720, 915, 290, 80),
}

# Standard LoL phase marks, in seconds (in-game clock == video timeline here,
# assuming the video starts at 0:00 of the game / champ select is trimmed off)
EARLY_MID_MARK_SEC = 14 * 60   # 14:00
MID_LATE_MARK_SEC = 25 * 60    # 25:00

# Force proportional thirds instead of the fixed marks above, regardless of
# video length. Leave False to let the script auto-decide.
FORCE_PROPORTIONAL = False

# How often to run OCR, in seconds of video time. 1.0 = once per second.
# Set this lower (e.g. 0.5) for finer resolution, higher (e.g. 2.0) for speed.
SAMPLE_INTERVAL_SEC = 1.0

# Show a live preview window while processing (slows things down; set False
# for a faster headless run, which is recommended for full VODs).
SHOW_PREVIEW = False

# ----------------------------------------------------------------------


def get_phase_boundaries(duration_sec: float):
    """
    Returns (early_end, mid_end) in seconds, i.e. the boundaries between
    early/mid and mid/late game, based on either the standard LoL marks or
    proportional thirds if the video is too short for the standard marks.
    """
    if not FORCE_PROPORTIONAL and duration_sec >= MID_LATE_MARK_SEC:
        return EARLY_MID_MARK_SEC, MID_LATE_MARK_SEC
    else:
        # proportional thirds fallback
        third = duration_sec / 3.0
        return third, 2 * third


def get_phase_for_time(t_sec: float, early_end: float, mid_end: float) -> str:
    if t_sec < early_end:
        return "early_game"
    elif t_sec < mid_end:
        return "mid_game"
    else:
        return "late_game"


def format_timestamp(t_sec: float) -> str:
    m = int(t_sec // 60)
    s = int(t_sec % 60)
    return f"{m:02d}:{s:02d}"


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

    print(f"Video duration : {format_timestamp(duration_sec)} ({duration_sec:.1f}s)")
    print(f"FPS            : {fps:.2f}")
    print(f"Total frames   : {total_frames}")
    print(f"Phase mode     : {'proportional thirds' if used_proportional else 'standard LoL marks'}")
    print(f"Early -> Mid at: {format_timestamp(early_end)}")
    print(f"Mid -> Late at : {format_timestamp(mid_end)}")
    print("-" * 50)

    reader = easyocr.Reader(["en"])

    # Result structure: one dict per phase, each holding a list of readings
    # per ROI.
    results = {
        "video_info": {
            "path": VIDEO_PATH,
            "duration_seconds": round(duration_sec, 2),
            "duration_timestamp": format_timestamp(duration_sec),
            "fps": fps,
            "total_frames": total_frames,
            "phase_mode": "proportional" if used_proportional else "standard",
            "phase_boundaries": {
                "early_to_mid_sec": round(early_end, 2),
                "mid_to_late_sec": round(mid_end, 2),
            },
        },
        "early_game": {name: [] for name in ROIS},
        "mid_game": {name: [] for name in ROIS},
        "late_game": {name: [] for name in ROIS},
    }

    frame_interval = max(1, int(round(fps * SAMPLE_INTERVAL_SEC)))

    if SHOW_PREVIEW:
        window = "Player Ability OCR"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_number = 0
    last_texts = {name: "" for name in ROIS}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only run OCR every `frame_interval` frames (sampling), not every frame
        if frame_number % frame_interval == 0:
            t_sec = frame_number / fps
            phase = get_phase_for_time(t_sec, early_end, mid_end)

            for name, (x, y, w, h) in ROIS.items():
                roi_frame = frame[y:y + h, x:x + w]
                gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                ocr_results = reader.readtext(gray)
                text = " ".join(t for _, t, _ in ocr_results).strip()

                last_texts[name] = text

                results[phase][name].append({
                    "frame": frame_number,
                    "time_sec": round(t_sec, 2),
                    "timestamp": format_timestamp(t_sec),
                    "text": text,
                })

                print(f"[{phase}] Frame {frame_number} ({format_timestamp(t_sec)}) | {name}: {text!r}")

        if SHOW_PREVIEW:
            display = frame.copy()
            for name, (x, y, w, h) in ROIS.items():
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    display,
                    f"{name}: {last_texts[name]}",
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        frame_number += 1

    cap.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("-" * 50)
    print(f"Done. Results written to: {OUTPUT_JSON}")
    for phase in ("early_game", "mid_game", "late_game"):
        counts = {name: len(v) for name, v in results[phase].items()}
        print(f"  {phase}: {counts}")


if __name__ == "__main__":
    main()