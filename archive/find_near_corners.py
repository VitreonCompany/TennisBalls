import cv2
import numpy as np
import math


VIDEO_PATH = "videos/test_match.mp4"
OUTPUT_PATH = "output/near_corners.jpg"
MANUAL_PATH = "data/court_points_8.npy"


# ============================================================
# LOAD FRAME
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, 60 * 1000)

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read video")

height, width = frame.shape[:2]

print(f"Frame: {width} x {height}")


# ============================================================
# 1. FIND ORANGE PLAYING SURFACE
# ============================================================

hsv = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2HSV
)

orange = cv2.inRange(
    hsv,
    np.array([5, 70, 70]),
    np.array([35, 255, 255])
)

kernel = np.ones(
    (7, 7),
    np.uint8
)

orange = cv2.morphologyEx(
    orange,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

orange = cv2.morphologyEx(
    orange,
    cv2.MORPH_OPEN,
    np.ones((3, 3), np.uint8)
)


# ============================================================
# KEEP MAIN LOWER ORANGE COMPONENT
# ============================================================

num_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        orange,
        connectivity=8
    )
)

court_mask = np.zeros_like(orange)

best_label = None
best_score = -1

for i in range(1, num_labels):

    y = stats[i, cv2.CC_STAT_TOP]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]

    bottom = y + h

    if area < 10000:
        continue

    if bottom < height * 0.55:
        continue

    score = area + bottom * 100

    if score > best_score:

        best_score = score
        best_label = i


if best_label is None:
    raise RuntimeError(
        "Could not isolate orange court"
    )


court_mask[
    labels == best_label
] = 255


# ============================================================
# FILL MAIN CONTOUR
# ============================================================

contours, _ = cv2.findContours(
    court_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

largest = max(
    contours,
    key=cv2.contourArea
)

court_mask[:] = 0

cv2.drawContours(
    court_mask,
    [largest],
    -1,
    255,
    thickness=cv2.FILLED
)


# ============================================================
# 2. FIND BRIGHT COURT LINES
# ============================================================

expanded = cv2.dilate(
    court_mask,
    np.ones((15, 15), np.uint8),
    iterations=1
)

white = cv2.inRange(
    hsv,
    np.array([0, 0, 120]),
    np.array([180, 120, 255])
)

white = cv2.bitwise_and(
    white,
    expanded
)

white = cv2.morphologyEx(
    white,
    cv2.MORPH_CLOSE,
    np.ones((5, 5), np.uint8)
)


edges = cv2.Canny(
    white,
    50,
    150
)


# ============================================================
# 3. HOUGH LINES
# ============================================================

lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi / 360,
    threshold=25,
    minLineLength=60,
    maxLineGap=40
)

if lines is None:
    raise RuntimeError(
        "No lines detected"
    )


# ============================================================
# HELPER
# ============================================================

def line_info(line):

    x1, y1, x2, y2 = line

    dx = x2 - x1
    dy = y2 - y1

    length = math.hypot(
        dx,
        dy
    )

    angle = math.degrees(
        math.atan2(
            dy,
            dx
        )
    )

    midpoint_y = (
        y1 + y2
    ) / 2

    return (
        length,
        angle,
        midpoint_y
    )


# ============================================================
# 4. FIND NEAR BASELINE
#
# Near baseline should:
#
# - be long
# - be in lower half
# - slope upward toward the right
#
# In image coordinates that means negative angle.
# ============================================================

candidates = []

for line in lines.reshape(-1, 4):

    length, angle, mid_y = (
        line_info(line)
    )

    # Lower portion of image
    if mid_y < height * 0.55:
        continue

    # Expected broad near-baseline direction
    if not (-25 <= angle <= -3):
        continue

    if length < 150:
        continue

    # Prefer long and lower lines
    score = (
        length
        +
        mid_y * 0.5
    )

    candidates.append(
        (
            score,
            line.copy(),
            length,
            angle
        )
    )


candidates.sort(
    key=lambda x: x[0],
    reverse=True
)


print()
print(
    f"Near-baseline candidates: "
    f"{len(candidates)}"
)


if not candidates:

    raise RuntimeError(
        "Could not find near baseline"
    )


# ============================================================
# SHOW TOP CANDIDATES
# ============================================================

output = frame.copy()

top = candidates[:10]

for rank, item in enumerate(
    top,
    start=1
):

    score, line, length, angle = item

    x1, y1, x2, y2 = line

    cv2.line(
        output,
        (x1, y1),
        (x2, y2),
        (255, 0, 255),
        3
    )

    cv2.putText(
        output,
        f"C{rank}",
        (x1, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    print(
        f"{rank}: "
        f"({x1},{y1}) -> "
        f"({x2},{y2}) | "
        f"length={length:.1f} | "
        f"angle={angle:.1f}"
    )


# ============================================================
# BEST CURRENT BASELINE
# ============================================================

best_line = candidates[0][1]

x1, y1, x2, y2 = best_line

cv2.line(
    output,
    (x1, y1),
    (x2, y2),
    (0, 255, 255),
    5
)


# ============================================================
# SAVE
# ============================================================

cv2.imwrite(
    OUTPUT_PATH,
    output
)


print()
print(
    f"Best near baseline:"
    f" ({x1},{y1}) -> ({x2},{y2})"
)

print()
print(
    f"Saved: {OUTPUT_PATH}"
)


# ============================================================
# MANUAL POINTS FOR BENCHMARK DISPLAY ONLY
#
# NOT USED IN DETECTION.
# ============================================================

try:

    manual = np.load(
        MANUAL_PATH
    )

    print()
    print("Hidden manual reference:")
    print(
        f"P7 = {manual[6]}"
    )
    print(
        f"P8 = {manual[7]}"
    )

except FileNotFoundError:

    pass