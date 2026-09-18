import difflib

import cv2
import easyocr

video_path = r"D:\League of Legend\LOL_2min.mp4"

# ROI format: (x, y, width, height)
ROI = (765, 95, 350, 100)

# Fuzzy matching / cooldown tuning
FUZZY_THRESHOLD = 0.75   # similarity ratio (0-1) to treat two OCR reads as the same event
COOLDOWN_SECONDS = 2.0   # gap of silence after which a still-active event is closed

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 30
cooldown_frames = int(COOLDOWN_SECONDS * fps)

reader = easyocr.Reader(["en"])

x, y, w, h = ROI

window = "Top Notification OCR"

cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    window,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


def frame_to_timestamp(frame_number):
    seconds = frame_number / fps
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def similarity(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


# --------------------------------------------------
# Detection state
# --------------------------------------------------

events = []   # finalized: {text, start_frame, start_time, end_frame, end_time, hits}
active = None  # in-progress event, same shape as above

frame_number = 0
ocr_interval = 15  # run OCR every N frames
last_text = ""


def close_active():
    global active
    if active is not None:
        events.append(active)
        print(
            f"[{active['start_time']} - {active['end_time']}] "
            f"{active['text']!r} (hits={active['hits']})"
        )
        active = None


while True:

    ret, frame = cap.read()

    if not ret:
        break

    roi_frame = frame[y:y + h, x:x + w]

    if frame_number % ocr_interval == 0:
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray)
        text = " ".join(t for _, t, _ in results).strip()

        if text:
            last_text = text

            if active is not None and similarity(text, active["text"]) >= FUZZY_THRESHOLD:
                # same ongoing notification (OCR noise aside) -> extend it
                active["end_frame"] = frame_number
                active["end_time"] = frame_to_timestamp(frame_number)
                active["hits"] += 1

                if len(text) > len(active["text"]):
                    active["text"] = text

            else:
                # a genuinely new notification -> close the old one, start a new one
                close_active()

                active = {
                    "text": text,
                    "start_frame": frame_number,
                    "start_time": frame_to_timestamp(frame_number),
                    "end_frame": frame_number,
                    "end_time": frame_to_timestamp(frame_number),
                    "hits": 1,
                }

        elif active is not None and (frame_number - active["end_frame"]) > cooldown_frames:
            # notification has been gone long enough -> close it out
            close_active()

    display = frame.copy()

    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        display,
        last_text,
        (x, max(25, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(window, display)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        break

    frame_number += 1

close_active()

cap.release()
cv2.destroyAllWindows()

print("\n========== Detected notifications ==========")
for e in events:
    print(f"[{e['start_time']} - {e['end_time']}] {e['text']!r} (hits={e['hits']})")
