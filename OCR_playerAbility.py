import cv2
import easyocr

video_path = r"C:\Users\gdeep\Downloads\LOL_36.mp4"

# ROI format: name -> (x, y, width, height)
ROIS = {
    "Players Ability Count": (455, 955, 180, 125),
    "PlayerUI Skillup":      (720, 915, 290, 80),
}

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video")
    exit()

reader = easyocr.Reader(["en"])

window = "Player Ability OCR"

cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    window,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

frame_number = 0
last_texts = {name: "" for name in ROIS}

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run OCR on every frame (no skipping) for each ROI
    for name, (x, y, w, h) in ROIS.items():
        roi_frame = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray)
        text = " ".join(t for _, t, _ in results).strip()

        last_texts[name] = text

        print(f"Frame {frame_number} | {name}: {text!r}")

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
            2
        )

    cv2.imshow(window, display)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        break

    frame_number += 1

cap.release()
cv2.destroyAllWindows()
