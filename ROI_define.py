import cv2

"""
Controls
A → move left
D → move right
W → move up
S → move down

J → decrease width
L → increase width
I → decrease height
K → increase height

. → next frame
, → previous frame

P → print coordinates
Q → quit
"""

video_path = r"D:\League of Legend\LOL_2min.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video")
    exit()

# Video information
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, frame = cap.read()

if not ret:
    print("Could not read video")
    exit()

h, w = frame.shape[:2]

# --------------------------------------------------
# Center ROI
# --------------------------------------------------

roi_w, roi_h = 400, 200

x = (w - roi_w) // 2
y = (h - roi_h) // 2

step = 5

# Current frame
frame_number = 0

# --------------------------------------------------
# Fullscreen window
# --------------------------------------------------

window = "ROI Selector"

cv2.namedWindow(window, cv2.WINDOW_NORMAL)

cv2.setWindowProperty(
    window,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# --------------------------------------------------
# Main loop
# --------------------------------------------------

while True:

    display = frame.copy()

    # Draw ROI
    cv2.rectangle(
        display,
        (x, y),
        (x + roi_w, y + roi_h),
        (0, 255, 0),
        2
    )

    # Display frame number
    cv2.putText(
        display,
        f"Frame: {frame_number}/{total_frames - 1}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # Show fullscreen
    cv2.imshow(window, display)

    key = cv2.waitKey(30) & 0xFF

    # --------------------------------------------------
    # Move ROI
    # --------------------------------------------------

    if key == ord('a'):
        x -= step

    elif key == ord('d'):
        x += step

    elif key == ord('w'):
        y -= step

    elif key == ord('s'):
        y += step

    # --------------------------------------------------
    # Resize ROI
    # --------------------------------------------------

    elif key == ord('j'):
        roi_w = max(1, roi_w - step)

    elif key == ord('l'):
        roi_w += step

    elif key == ord('i'):
        roi_h = max(1, roi_h - step)

    elif key == ord('k'):
        roi_h += step

    # --------------------------------------------------
    # NEXT FRAME  .
    # --------------------------------------------------

    elif key == ord('.'):

        if frame_number < total_frames - 1:

            frame_number += 1

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number
            )

            ret, frame = cap.read()

            if not ret:
                break

    # --------------------------------------------------
    # PREVIOUS FRAME  ,
    # --------------------------------------------------

    elif key == ord(','):

        if frame_number > 0:

            frame_number -= 1

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number
            )

            ret, frame = cap.read()

            if not ret:
                break

    # --------------------------------------------------
    # PRINT ROI
    # --------------------------------------------------

    elif key == ord('p'):

        print("\n========== ROI ==========")

        print(f"Frame  = {frame_number}")

        print(f"x      = {x}")
        print(f"y      = {y}")
        print(f"width  = {roi_w}")
        print(f"height = {roi_h}")

        print(
            f"(x1,y1,x2,y2) = "
            f"({x}, {y}, {x + roi_w}, {y + roi_h})"
        )

        print("=========================")

    # --------------------------------------------------
    # QUIT
    # --------------------------------------------------

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



