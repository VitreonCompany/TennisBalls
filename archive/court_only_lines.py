import cv2
import numpy as np
import math

VIDEO_PATH = "videos/test_match.mp4"

OUTPUT_IMAGE = "output/court_only_lines.jpg"

# --------------------------------------------------
# LOAD FRAME
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, 60 * 1000)

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read frame")

height, width = frame.shape[:2]

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


# --------------------------------------------------
# ORANGE COURT MASK
# --------------------------------------------------

orange = cv2.inRange(
    hsv,
    np.array([5, 70, 70]),
    np.array([35, 255, 255])
)

orange = cv2.morphologyEx(
    orange,
    cv2.MORPH_CLOSE,
    np.ones((7, 7), np.uint8),
    iterations=2
)

orange = cv2.morphologyEx(
    orange,
    cv2.MORPH_OPEN,
    np.ones((3, 3), np.uint8)
)


# --------------------------------------------------
# SELECT FOREGROUND ORANGE REGION
# --------------------------------------------------

num_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        orange,
        connectivity=8
    )
)

best_label = None
best_score = -1

for i in range(1, num_labels):

    area = stats[i, cv2.CC_STAT_AREA]

    y = stats[i, cv2.CC_STAT_TOP]
    h = stats[i, cv2.CC_STAT_HEIGHT]

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
    raise RuntimeError("Could not find foreground court")


court_mask = np.zeros_like(orange)

court_mask[labels == best_label] = 255


# --------------------------------------------------
# FILL COURT REGION
# --------------------------------------------------

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
    cv2.FILLED
)


# --------------------------------------------------
# EXPAND MASK SLIGHTLY
#
# Important because the white paint itself may sit
# just outside the detected orange pixels.
# --------------------------------------------------

court_mask = cv2.dilate(
    court_mask,
    np.ones((15, 15), np.uint8),
    iterations=1
)


# --------------------------------------------------
# WHITE / LIGHT LINE EVIDENCE
# --------------------------------------------------

white = cv2.inRange(
    hsv,
    np.array([0, 0, 120]),
    np.array([180, 120, 255])
)


# --------------------------------------------------
# ONLY KEEP WHITE PIXELS NEAR OUR COURT
# --------------------------------------------------

court_white = cv2.bitwise_and(
    white,
    court_mask
)


court_white = cv2.morphologyEx(
    court_white,
    cv2.MORPH_CLOSE,
    np.ones((3, 3), np.uint8)
)


# --------------------------------------------------
# EDGES
# --------------------------------------------------

edges = cv2.Canny(
    court_white,
    50,
    150
)


# --------------------------------------------------
# HOUGH LINES
# --------------------------------------------------

lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 360,
    threshold=25,
    minLineLength=40,
    maxLineGap=35
)


candidates = []

if lines is not None:

    for x1, y1, x2, y2 in lines.reshape(-1, 4):

        dx = x2 - x1
        dy = y2 - y1

        length = math.hypot(dx, dy)

        if length < 60:
            continue

        angle = math.degrees(
            math.atan2(dy, dx)
        )

        if angle > 90:
            angle -= 180

        if angle < -90:
            angle += 180

        candidates.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "length": length,
            "angle": angle
        })


# --------------------------------------------------
# SORT BY LENGTH
# --------------------------------------------------

candidates.sort(
    key=lambda x: x["length"],
    reverse=True
)

strongest = candidates[:40]


# --------------------------------------------------
# DRAW
# --------------------------------------------------

output = frame.copy()

for i, line in enumerate(strongest, start=1):

    x1 = line["x1"]
    y1 = line["y1"]
    x2 = line["x2"]
    y2 = line["y2"]

    cv2.line(
        output,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    mx = int((x1 + x2) / 2)
    my = int((y1 + y2) / 2)

    cv2.putText(
        output,
        str(i),
        (mx, my),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2
    )


# --------------------------------------------------
# SAVE
# --------------------------------------------------

cv2.imwrite(
    "output/court_only_white.jpg",
    court_white
)

cv2.imwrite(
    "output/court_only_edges.jpg",
    edges
)

cv2.imwrite(
    OUTPUT_IMAGE,
    output
)


print()
print(f"Court-only line candidates: {len(candidates)}")
print()
print("Strongest lines:")

for i, line in enumerate(strongest, start=1):

    print(
        f"{i:2d}: "
        f"length={line['length']:.1f}px | "
        f"angle={line['angle']:.1f}°"
    )

print()
print(f"Saved: {OUTPUT_IMAGE}")