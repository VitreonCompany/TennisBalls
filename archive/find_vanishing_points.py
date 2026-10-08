import cv2
import numpy as np
import math
import random

VIDEO_PATH = "videos/test_match.mp4"
OUTPUT_PATH = "output/vanishing_points.jpg"

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
# REBUILD FOREGROUND COURT MASK
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
    raise RuntimeError("Could not identify court")

court_mask = np.zeros_like(orange)
court_mask[labels == best_label] = 255

contours, _ = cv2.findContours(
    court_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

largest = max(contours, key=cv2.contourArea)

court_mask[:] = 0

cv2.drawContours(
    court_mask,
    [largest],
    -1,
    255,
    cv2.FILLED
)

court_mask = cv2.dilate(
    court_mask,
    np.ones((15, 15), np.uint8),
    iterations=1
)


# --------------------------------------------------
# WHITE LINE EVIDENCE
# --------------------------------------------------

white = cv2.inRange(
    hsv,
    np.array([0, 0, 120]),
    np.array([180, 120, 255])
)

court_white = cv2.bitwise_and(
    white,
    court_mask
)

court_white = cv2.morphologyEx(
    court_white,
    cv2.MORPH_CLOSE,
    np.ones((3, 3), np.uint8)
)

edges = cv2.Canny(
    court_white,
    50,
    150
)


# --------------------------------------------------
# HOUGH SEGMENTS
# --------------------------------------------------

lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 360,
    threshold=25,
    minLineLength=40,
    maxLineGap=35
)

segments = []

if lines is not None:

    for x1, y1, x2, y2 in lines.reshape(-1, 4):

        length = math.hypot(
            x2 - x1,
            y2 - y1
        )

        if length < 60:
            continue

        angle = math.degrees(
            math.atan2(
                y2 - y1,
                x2 - x1
            )
        )

        if angle > 90:
            angle -= 180

        if angle < -90:
            angle += 180

        segments.append({
            "p1": np.array([x1, y1], dtype=float),
            "p2": np.array([x2, y2], dtype=float),
            "length": length,
            "angle": angle
        })


print(f"Usable segments: {len(segments)}")


# --------------------------------------------------
# LINE MATH
# --------------------------------------------------

def infinite_line(segment):

    x1, y1 = segment["p1"]
    x2, y2 = segment["p2"]

    p1 = np.array([x1, y1, 1.0])
    p2 = np.array([x2, y2, 1.0])

    return np.cross(p1, p2)


def intersection(seg1, seg2):

    l1 = infinite_line(seg1)
    l2 = infinite_line(seg2)

    point = np.cross(l1, l2)

    if abs(point[2]) < 1e-8:
        return None

    return point[:2] / point[2]


def distance_point_to_line(point, segment):

    line = infinite_line(segment)

    a, b, c = line

    x, y = point

    return abs(
        a * x + b * y + c
    ) / math.sqrt(a * a + b * b)


# --------------------------------------------------
# FIND VANISHING POINT USING RANSAC
# --------------------------------------------------

def find_vanishing_point(candidate_segments):

    if len(candidate_segments) < 2:
        return None, []

    best_point = None
    best_inliers = []
    best_score = -1

    random.seed(42)

    for _ in range(5000):

        seg1, seg2 = random.sample(
            candidate_segments,
            2
        )

        # Nearly identical angles give unstable
        # intersections.
        if abs(
            seg1["angle"] - seg2["angle"]
        ) < 1.0:
            continue

        point = intersection(
            seg1,
            seg2
        )

        if point is None:
            continue

        # Reject absurd numerical solutions
        if abs(point[0]) > width * 20:
            continue

        if abs(point[1]) > height * 20:
            continue

        inliers = []
        score = 0

        for seg in candidate_segments:

            error = distance_point_to_line(
                point,
                seg
            )

            # Longer segments count more
            if error < 12:
                inliers.append(seg)

                score += seg["length"]

        if score > best_score:

            best_score = score
            best_point = point
            best_inliers = inliers

    return best_point, best_inliers


# --------------------------------------------------
# TWO BROAD FAMILIES
#
# Family A:
# lines roughly horizontal in this image.
#
# Family B:
# stronger diagonals.
#
# These ranges are deliberately broad.
# --------------------------------------------------

family_a = [
    s for s in segments
    if -15 <= s["angle"] <= 12
]

family_b = [
    s for s in segments
    if 12 < s["angle"] <= 35
]


print(f"Family A segments: {len(family_a)}")
print(f"Family B segments: {len(family_b)}")


vp_a, inliers_a = find_vanishing_point(
    family_a
)

vp_b, inliers_b = find_vanishing_point(
    family_b
)


print()
print("Vanishing point A:")
print(vp_a)
print(f"Inliers: {len(inliers_a)}")

print()
print("Vanishing point B:")
print(vp_b)
print(f"Inliers: {len(inliers_b)}")


# --------------------------------------------------
# DRAW RESULTS
# --------------------------------------------------

output = frame.copy()


# Family A = green
for seg in inliers_a:

    p1 = tuple(
        np.round(seg["p1"]).astype(int)
    )

    p2 = tuple(
        np.round(seg["p2"]).astype(int)
    )

    cv2.line(
        output,
        p1,
        p2,
        (0, 255, 0),
        3
    )


# Family B = cyan
for seg in inliers_b:

    p1 = tuple(
        np.round(seg["p1"]).astype(int)
    )

    p2 = tuple(
        np.round(seg["p2"]).astype(int)
    )

    cv2.line(
        output,
        p1,
        p2,
        (255, 255, 0),
        3
    )


# --------------------------------------------------
# DRAW VANISHING DIRECTIONS
#
# Vanishing points may be far outside the image,
# so instead of trying to display the actual point,
# draw rays toward it.
# --------------------------------------------------

def draw_vp_rays(image, vp, inliers, color):

    if vp is None:
        return

    for seg in inliers[:8]:

        midpoint = (
            seg["p1"] + seg["p2"]
        ) / 2

        direction = vp - midpoint

        norm = np.linalg.norm(direction)

        if norm == 0:
            continue

        direction /= norm

        end = midpoint + direction * 300

        cv2.line(
            image,
            tuple(np.round(midpoint).astype(int)),
            tuple(np.round(end).astype(int)),
            color,
            1
        )


draw_vp_rays(
    output,
    vp_a,
    inliers_a,
    (0, 255, 0)
)

draw_vp_rays(
    output,
    vp_b,
    inliers_b,
    (255, 255, 0)
)


cv2.imwrite(
    OUTPUT_PATH,
    output
)

print()
print(f"Saved: {OUTPUT_PATH}")