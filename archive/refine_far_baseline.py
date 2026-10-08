import cv2
import numpy as np
import math

VIDEO_PATH = "videos/test_match.mp4"
AI_POINTS_PATH = "data/ai_court_points_v2.npy"

OUTPUT_PATH = "output/refined_far_baseline.jpg"

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
# LOAD AI-GENERATED COURT
# --------------------------------------------------

points = np.load(
    AI_POINTS_PATH
).astype(np.float32)

p1 = points[0].copy()
p2 = points[1].copy()

p3 = points[2].copy()
p4 = points[3].copy()

p5 = points[4].copy()
p6 = points[5].copy()

p7 = points[6].copy()
p8 = points[7].copy()


print()
print("Original AI far baseline:")
print("P1:", p1)
print("P2:", p2)


# --------------------------------------------------
# WHITE-LINE EVIDENCE
#
# We deliberately use a reasonably broad definition.
# The far baseline is faint.
# --------------------------------------------------

white = cv2.inRange(
    hsv,
    np.array([0, 0, 105]),
    np.array([180, 145, 255])
)


# --------------------------------------------------
# SEARCH REGION
#
# The far baseline must be close to the AI estimate.
#
# We only examine the upper/far part of the court.
# --------------------------------------------------

search_mask = np.zeros(
    (height, width),
    dtype=np.uint8
)

polygon = np.array([
    [0, 400],
    [750, 400],
    [750, 485],
    [0, 485]
], dtype=np.int32)

cv2.fillPoly(
    search_mask,
    [polygon],
    255
)

search_white = cv2.bitwise_and(
    white,
    search_mask
)


# --------------------------------------------------
# CLEAN MASK
# --------------------------------------------------

search_white = cv2.morphologyEx(
    search_white,
    cv2.MORPH_CLOSE,
    np.ones((3, 3), np.uint8)
)


# --------------------------------------------------
# EDGE DETECTION
# --------------------------------------------------

edges = cv2.Canny(
    search_white,
    40,
    120
)


# --------------------------------------------------
# FIND LINE SEGMENTS
# --------------------------------------------------

lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 720,
    threshold=20,
    minLineLength=70,
    maxLineGap=50
)


# --------------------------------------------------
# EXPECTED FAR BASELINE DIRECTION
#
# P1 -> P2 gives us the AI's approximate baseline
# orientation.
# --------------------------------------------------

expected_angle = math.degrees(
    math.atan2(
        p2[1] - p1[1],
        p2[0] - p1[0]
    )
)

print()
print(
    f"Expected baseline angle: "
    f"{expected_angle:.2f} degrees"
)


# --------------------------------------------------
# HELPER
# --------------------------------------------------

def angle_difference(a, b):

    diff = abs(a - b)

    while diff > 180:
        diff -= 180

    if diff > 90:
        diff = 180 - diff

    return abs(diff)


# --------------------------------------------------
# SCORE CANDIDATE BASELINES
# --------------------------------------------------

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

        angle_error = angle_difference(
            angle,
            expected_angle
        )

        # Far baseline should be close in direction
        # to the AI estimate.
        if angle_error > 8:
            continue

        midpoint_y = (
            y1 + y2
        ) / 2

        midpoint_x = (
            x1 + x2
        ) / 2

        # Expected baseline is around y=457.
        expected_y = (
            p1[1] + p2[1]
        ) / 2

        vertical_error = abs(
            midpoint_y - expected_y
        )

        # Prefer:
        # - long segments
        # - correct orientation
        # - sensible vertical location
        score = (
            length
            - angle_error * 20
            - vertical_error * 2
        )

        candidates.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "length": length,
            "angle": angle,
            "angle_error": angle_error,
            "vertical_error": vertical_error,
            "score": score
        })


candidates.sort(
    key=lambda x: x["score"],
    reverse=True
)


# --------------------------------------------------
# PRINT CANDIDATES
# --------------------------------------------------

print()
print(
    f"Far-baseline candidates: "
    f"{len(candidates)}"
)

print()

for i, c in enumerate(
    candidates[:15],
    start=1
):

    print(
        f"{i:2d}: "
        f"({c['x1']},{c['y1']}) -> "
        f"({c['x2']},{c['y2']}) | "
        f"len={c['length']:.1f} | "
        f"angle={c['angle']:.2f} | "
        f"score={c['score']:.1f}"
    )


# --------------------------------------------------
# DRAW DEBUG IMAGE
# --------------------------------------------------

output = frame.copy()


# Original AI baseline = RED
cv2.line(
    output,
    tuple(np.round(p1).astype(int)),
    tuple(np.round(p2).astype(int)),
    (0, 0, 255),
    3
)


# AI points = RED
cv2.circle(
    output,
    tuple(np.round(p1).astype(int)),
    8,
    (0, 0, 255),
    -1
)

cv2.circle(
    output,
    tuple(np.round(p2).astype(int)),
    8,
    (0, 0, 255),
    -1
)


# --------------------------------------------------
# DRAW TOP CANDIDATES
# --------------------------------------------------

colors = [
    (0, 255, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 165, 255),
    (255, 0, 0)
]

for i, candidate in enumerate(
    candidates[:5]
):

    color = colors[i]

    cv2.line(
        output,
        (
            candidate["x1"],
            candidate["y1"]
        ),
        (
            candidate["x2"],
            candidate["y2"]
        ),
        color,
        3
    )

    mx = int(
        (
            candidate["x1"]
            + candidate["x2"]
        ) / 2
    )

    my = int(
        (
            candidate["y1"]
            + candidate["y2"]
        ) / 2
    )

    cv2.putText(
        output,
        f"C{i + 1}",
        (mx, my - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )


# --------------------------------------------------
# ALSO DRAW GOOD POINTS 3-8
# --------------------------------------------------

for i in range(2, 8):

    x, y = points[i]

    cv2.circle(
        output,
        (int(round(x)), int(round(y))),
        5,
        (0, 255, 255),
        -1
    )

    cv2.putText(
        output,
        str(i + 1),
        (
            int(round(x)) + 7,
            int(round(y)) - 7
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1
    )


# --------------------------------------------------
# SAVE
# --------------------------------------------------

cv2.imwrite(
    OUTPUT_PATH,
    output
)

cv2.imwrite(
    "output/far_baseline_search_mask.jpg",
    search_white
)

print()
print(f"Saved: {OUTPUT_PATH}")