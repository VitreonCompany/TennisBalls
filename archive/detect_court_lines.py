import cv2
import numpy as np

VIDEO_PATH = "videos/test_match.mp4"
OUTPUT_PATH = "output/automatic_lines.jpg"

# Load frame
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, 60 * 1000)

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not load video frame")


# --------------------------------------------------
# 1. Convert to HSV
# --------------------------------------------------

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Court lines are relatively bright and low saturation
# compared with the orange court.
lower = np.array([0, 0, 120])
upper = np.array([180, 120, 255])

mask = cv2.inRange(hsv, lower, upper)


# --------------------------------------------------
# 2. Only examine the region where our court exists
# --------------------------------------------------

roi_mask = np.zeros(mask.shape, dtype=np.uint8)

court_polygon = np.array([
    [40, 420],
    [520, 420],
    [1700, 590],
    [900, 770]
], dtype=np.int32)

cv2.fillPoly(
    roi_mask,
    [court_polygon],
    255
)

mask = cv2.bitwise_and(mask, roi_mask)


# --------------------------------------------------
# 3. Clean the mask
# --------------------------------------------------

kernel = np.ones((3, 3), np.uint8)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)


# --------------------------------------------------
# 4. Edge detection
# --------------------------------------------------

edges = cv2.Canny(
    mask,
    50,
    150
)


# --------------------------------------------------
# 5. Hough line detection
# --------------------------------------------------

lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=40,
    minLineLength=40,
    maxLineGap=30
)


# --------------------------------------------------
# Draw detected lines
# --------------------------------------------------

output = frame.copy()

line_count = 0

if lines is not None:
    
    for line in lines.reshape(-1, 4):

        x1, y1, x2, y2 = line

        length = np.hypot(
            x2 - x1,
            y2 - y1
        )

        # Ignore tiny segments
        if length < 50:
            continue

        cv2.line(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        line_count += 1


cv2.imwrite(
    OUTPUT_PATH,
    output
)

cv2.imwrite(
    "output/court_line_mask.jpg",
    mask
)

cv2.imwrite(
    "output/court_edges.jpg",
    edges
)


print()
print(f"Detected {line_count} candidate lines")
print()
print("Saved:")
print("output/automatic_lines.jpg")
print("output/court_line_mask.jpg")
print("output/court_edges.jpg")