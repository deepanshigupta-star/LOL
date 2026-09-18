#scoreboard , mapp 
import cv2

video_path = r"D:\League of Legend\LOL_2min.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video")
    exit()

# ROI format:
# name: (x, y, width, height)

ROIS = {
    "Top Notification":     (765, 95, 350, 100),
    "Players Ability Count": (455, 955, 180, 125),
    "PlayerUI Bar":          (695, 955, 400, 125),
    "PlayerUI Ability":      (705, 965, 385, 65),
    "PlayerUI Skillup":      (720, 915, 290, 80),
    "teammates":             (1585, 505, 330, 255),
    "topScore":              (1545, 0, 370, 70),
    "Minmap":                (1570, 740, 345, 335),
    "Scoreboard":            (390, 240, 1150, 450),
}

window = "ROI Regions"

cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    window,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    display = frame.copy()

    # Draw all ROIs
    for name, (x, y, w, h) in ROIS.items():

        # ROI rectangle
        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # ROI name above rectangle
        cv2.putText(
            display,
            name,
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

cap.release()
cv2.destroyAllWindows()