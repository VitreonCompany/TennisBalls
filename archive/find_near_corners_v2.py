import cv2
import numpy as np
import math


VIDEO_PATH = "videos/test_match.mp4"
OUTPUT_PATH = "output/near_corners_v2.jpg"
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

print()
print(f"Frame: {width} x {height}")


# ============================================================
# 1. ORANGE COURT MASK
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
# 2. WHITE COURT-LINE EVIDENCE
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
# 3. HOUGH SEGMENTS
# ============================================================

lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi / 360,
    threshold=25,
    minLineLength=50,
    maxLineGap=40
)

if lines is None:
    raise RuntimeError(
        "No court lines detected"
    )

segments = lines.reshape(-1, 4)


# ============================================================
# HELPERS
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
        math.atan2(dy, dx)
    )

    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    return (
        length,
        angle,
        mid_x,
        mid_y
    )


def infinite_line(line):

    x1, y1, x2, y2 = line

    # ax + by + c = 0

    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1

    norm = math.hypot(a, b)

    if norm == 0:
        return None

    return np.array(
        [a / norm, b / norm, c / norm],
        dtype=np.float64
    )


def intersection(line_a, line_b):

    a1, b1, c1 = line_a
    a2, b2, c2 = line_b

    det = (
        a1 * b2
        -
        a2 * b1
    )

    if abs(det) < 1e-8:
        return None

    x = (
        b1 * c2
        -
        b2 * c1
    ) / det

    y = (
        c1 * a2
        -
        c2 * a1
    ) / det

    return np.array(
        [x, y],
        dtype=np.float64
    )


# ============================================================
# 4. FIND NEAR BASELINE
#
# Same idea as our successful V1 test.
# ============================================================

baseline_candidates = []

for segment in segments:

    length, angle, mid_x, mid_y = (
        line_info(segment)
    )

    if mid_y < height * 0.55:
        continue

    if not (-25 <= angle <= -3):
        continue

    if length < 150:
        continue

    score = (
        length
        +
        mid_y * 0.5
    )

    baseline_candidates.append(
        (
            score,
            segment.copy(),
            length,
            angle
        )
    )


baseline_candidates.sort(
    key=lambda x: x[0],
    reverse=True
)


if not baseline_candidates:

    raise RuntimeError(
        "Could not find near baseline"
    )


baseline_segment = (
    baseline_candidates[0][1]
)

baseline_line = infinite_line(
    baseline_segment
)


print()
print("NEAR BASELINE")
print("=" * 65)

print(
    f"Segment: "
    f"{baseline_segment}"
)

print(
    f"Angle: "
    f"{baseline_candidates[0][3]:.2f} degrees"
)


# ============================================================
# 5. FIND SIDELINE CANDIDATES
#
# We want lines that:
#
# - run lengthwise down the court
# - intersect our near baseline
# - intersect it inside/near the visible orange court
#
# We intentionally keep this fairly broad for this first test.
# ============================================================

sideline_candidates = []


for segment in segments:

    length, angle, mid_x, mid_y = (
        line_info(segment)
    )

    if length < 80:
        continue

    # Ignore lines with approximately the same direction
    # as the near baseline.

    baseline_angle = (
        baseline_candidates[0][3]
    )

    angle_difference = abs(
        angle - baseline_angle
    )

    angle_difference = min(
        angle_difference,
        180 - angle_difference
    )

    if angle_difference < 8:
        continue


    candidate_line = infinite_line(
        segment
    )

    if candidate_line is None:
        continue


    p = intersection(
        baseline_line,
        candidate_line
    )

    if p is None:
        continue

    x, y = p


    # Intersection must be around the visible image.

    if x < -100:
        continue

    if x > width + 100:
        continue

    if y < height * 0.45:
        continue

    if y > height * 0.95:
        continue


    # We prefer:
    #
    # - long line evidence
    # - intersection reasonably low in frame
    # - clear directional difference from baseline

    score = (
        length
        +
        angle_difference * 5
    )


    sideline_candidates.append(
        {
            "score": score,
            "segment": segment.copy(),
            "line": candidate_line,
            "intersection": p,
            "length": length,
            "angle": angle
        }
    )


print()
print(
    f"Possible sideline segments: "
    f"{len(sideline_candidates)}"
)


# ============================================================
# 6. GROUP INTERSECTIONS
#
# Multiple Hough segments often represent the same painted line.
#
# Rather than treating every segment as a separate sideline,
# cluster intersections that land near one another on the
# baseline.
# ============================================================

sideline_candidates.sort(
    key=lambda item:
        item["intersection"][0]
)


clusters = []

CLUSTER_DISTANCE = 12


for candidate in sideline_candidates:

    x = candidate[
        "intersection"
    ][0]

    placed = False

    for cluster in clusters:

        cluster_x = np.average(
            [
                item["intersection"][0]
                for item in cluster
            ],
            weights=[
                item["length"]
                for item in cluster
            ]
        )

        if abs(
            x - cluster_x
        ) < CLUSTER_DISTANCE:

            cluster.append(
                candidate
            )

            placed = True
            break


    if not placed:

        clusters.append(
            [candidate]
        )


# ============================================================
# 7. SCORE CLUSTERS
# ============================================================

cluster_results = []


for cluster in clusters:

    weights = np.array(
        [
            item["length"]
            for item in cluster
        ]
    )

    points = np.array(
        [
            item["intersection"]
            for item in cluster
        ]
    )

    weighted_point = np.average(
        points,
        axis=0,
        weights=weights
    )


    total_line_length = sum(
        item["length"]
        for item in cluster
    )


    cluster_results.append(
        {
            "point": weighted_point,
            "strength": total_line_length,
            "members": cluster
        }
    )


# Only keep plausible intersections.

cluster_results = [
    c
    for c in cluster_results

    if (
        0 <= c["point"][0] <= width
        and
        height * 0.50
        <= c["point"][1]
        <= height * 0.90
    )
]


cluster_results.sort(
    key=lambda c:
        c["strength"],
    reverse=True
)


print()
print("INTERSECTION CLUSTERS")
print("=" * 65)


for i, cluster in enumerate(
    cluster_results[:10],
    start=1
):

    p = cluster["point"]

    print(
        f"C{i}: "
        f"({p[0]:.1f}, {p[1]:.1f}) "
        f"strength="
        f"{cluster['strength']:.1f}"
    )


# ============================================================
# 8. SELECT LEFT AND RIGHT SINGLES INTERSECTIONS
#
# For this first version:
#
# take strong separated intersections.
#
# We expect the singles court to span a large portion of the
# near baseline.
# ============================================================

if len(cluster_results) < 2:

    raise RuntimeError(
        "Not enough sideline intersections"
    )


best_pair = None
best_pair_score = -1


for i in range(
    len(cluster_results)
):

    for j in range(
        i + 1,
        len(cluster_results)
    ):

        p_a = cluster_results[i][
            "point"
        ]

        p_b = cluster_results[j][
            "point"
        ]

        left = (
            p_a
            if p_a[0] < p_b[0]
            else p_b
        )

        right = (
            p_b
            if p_a[0] < p_b[0]
            else p_a
        )


        separation = (
            right[0] - left[0]
        )


        # Singles baseline should be a substantial visible span.

        if separation < width * 0.25:
            continue

        if separation > width * 0.75:
            continue


        strength = (
            cluster_results[i][
                "strength"
            ]
            +
            cluster_results[j][
                "strength"
            ]
        )


        # Reward both strong evidence and a useful court width.

        score = (
            strength
            +
            separation * 0.5
        )


        if score > best_pair_score:

            best_pair_score = score

            best_pair = (
                left,
                right
            )


if best_pair is None:

    raise RuntimeError(
        "Could not select near singles corners"
    )


p7, p8 = best_pair


# ============================================================
# RESULTS
# ============================================================

print()
print("AUTOMATIC NEAR CORNERS")
print("=" * 65)

print(
    f"P7 = "
    f"({p7[0]:.1f}, {p7[1]:.1f})"
)

print(
    f"P8 = "
    f"({p8[0]:.1f}, {p8[1]:.1f})"
)


# ============================================================
# HIDDEN MANUAL BENCHMARK
#
# NOT USED ABOVE.
# ============================================================

manual = np.load(
    MANUAL_PATH
).astype(np.float64)

manual_p7 = manual[6]
manual_p8 = manual[7]


error7 = np.linalg.norm(
    p7 - manual_p7
)

error8 = np.linalg.norm(
    p8 - manual_p8
)


print()
print("HIDDEN MANUAL BENCHMARK")
print("=" * 65)

print(
    f"P7 AUTO="
    f"({p7[0]:.1f}, {p7[1]:.1f}) | "
    f"MANUAL="
    f"({manual_p7[0]:.1f}, "
    f"{manual_p7[1]:.1f}) | "
    f"ERROR={error7:.1f}px"
)

print(
    f"P8 AUTO="
    f"({p8[0]:.1f}, {p8[1]:.1f}) | "
    f"MANUAL="
    f"({manual_p8[0]:.1f}, "
    f"{manual_p8[1]:.1f}) | "
    f"ERROR={error8:.1f}px"
)

print()

print(
    f"Mean error: "
    f"{(error7 + error8) / 2:.1f}px"
)


# ============================================================
# DRAW DEBUG IMAGE
# ============================================================

output = frame.copy()


# Draw detected near baseline

x1, y1, x2, y2 = (
    baseline_segment
)

cv2.line(
    output,
    (x1, y1),
    (x2, y2),
    (0, 255, 255),
    5
)


# Draw top intersection clusters

for i, cluster in enumerate(
    cluster_results[:10],
    start=1
):

    p = cluster["point"]

    x = int(round(p[0]))
    y = int(round(p[1]))

    cv2.circle(
        output,
        (x, y),
        5,
        (255, 0, 0),
        -1
    )

    cv2.putText(
        output,
        f"C{i}",
        (x + 5, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2
    )


# Draw selected P7/P8

for number, point in [
    (7, p7),
    (8, p8)
]:

    x = int(round(point[0]))
    y = int(round(point[1]))

    cv2.circle(
        output,
        (x, y),
        10,
        (0, 255, 255),
        -1
    )

    cv2.putText(
        output,
        f"P{number}",
        (x + 12, y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


# Draw selected baseline between P7/P8

cv2.line(
    output,
    tuple(
        np.round(p7).astype(int)
    ),
    tuple(
        np.round(p8).astype(int)
    ),
    (255, 0, 255),
    4
)


cv2.imwrite(
    OUTPUT_PATH,
    output
)


print()
print(
    f"Saved: {OUTPUT_PATH}"
)