import cv2
import numpy as np
import math

VIDEO_PATH = "videos/test_match.mp4"
OUTPUT_PATH = "output/strong_court_lines.jpg"

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, 60 * 1000)

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read video frame")

height, width = frame.shape[:2]

# --------------------------------------------------
# 1. Find light/white markings
# --------------------------------------------------

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(
    hsv,
    np.array([0, 0, 115]),
    np.array([180, 145, 255])
)

# Ignore very top/bottom of image
roi = np.zeros_like(mask)

roi[
    int(height * 0.27):int(height * 0.90),
    :
] = 255

mask = cv2.bitwise_and(mask, roi)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    np.ones((3, 3), np.uint8)
)

edges = cv2.Canny(mask, 50, 150)

# --------------------------------------------------
# 2. Detect line segments
# --------------------------------------------------

lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 360,   # half-degree resolution
    threshold=35,
    minLineLength=50,
    maxLineGap=35
)

candidates = []

if lines is not None:

    for x1, y1, x2, y2 in lines.reshape(-1, 4):

        dx = x2 - x1
        dy = y2 - y1

        length = math.hypot(dx, dy)

        if length < 70:
            continue

        angle = math.degrees(
            math.atan2(dy, dx)
        )

        # Normalize to -90 ... +90
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
# 3. Keep strongest lines
# --------------------------------------------------

candidates.sort(
    key=lambda line: line["length"],
    reverse=True
)

strongest = candidates[:30]

print()
print(f"Total candidates: {len(candidates)}")
print()
print("30 strongest lines:")
print("-" * 70)

for i, line in enumerate(strongest, start=1):

    print(
        f"{i:2d}: "
        f"length={line['length']:7.1f}px | "
        f"angle={line['angle']:6.1f}° | "
        f"({line['x1']},{line['y1']}) -> "
        f"({line['x2']},{line['y2']})"
    )


# --------------------------------------------------
# 4. Draw them with numbers
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

    mid_x = int((x1 + x2) / 2)
    mid_y = int((y1 + y2) / 2)

    cv2.putText(
        output,
        str(i),
        (mid_x, mid_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2
    )


cv2.imwrite(
    OUTPUT_PATH,
    output
)

print()
print(f"Saved: {OUTPUT_PATH}")