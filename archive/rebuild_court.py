import cv2
import numpy as np

VIDEO_PATH = "videos/test_match.mp4"

AI_POINTS_PATH = "data/ai_court_points_v2.npy"

OUTPUT_PATH = "output/rebuilt_court.jpg"
OUTPUT_POINTS = "data/rebuilt_court_points.npy"


# --------------------------------------------------
# LOAD FRAME
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, 60 * 1000)

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read frame")


# --------------------------------------------------
# LOAD EXISTING AI POINTS
# --------------------------------------------------

old_points = np.load(
    AI_POINTS_PATH
).astype(np.float32)

# Good points from our previous result
p5 = old_points[4]
p6 = old_points[5]
p7 = old_points[6]
p8 = old_points[7]


# --------------------------------------------------
# DETECTED FAR BASELINE
#
# This is the strongest candidate from our previous
# automatic search:
#
# (0,447) -> (564,447)
#
# IMPORTANT:
# We use it as an INFINITE LINE.
# Its endpoints are NOT court corners.
# --------------------------------------------------

baseline_a = np.array(
    [0.0, 447.0],
    dtype=np.float32
)

baseline_b = np.array(
    [564.0, 447.0],
    dtype=np.float32
)


# --------------------------------------------------
# LINE INTERSECTION
# --------------------------------------------------

def line_intersection(a1, a2, b1, b2):

    x1, y1 = a1
    x2, y2 = a2

    x3, y3 = b1
    x4, y4 = b2

    denominator = (
        (x1 - x2) * (y3 - y4)
        -
        (y1 - y2) * (x3 - x4)
    )

    if abs(denominator) < 1e-8:
        raise RuntimeError(
            "Lines are parallel"
        )

    px = (
        (x1 * y2 - y1 * x2)
        * (x3 - x4)
        -
        (x1 - x2)
        * (x3 * y4 - y3 * x4)
    ) / denominator

    py = (
        (x1 * y2 - y1 * x2)
        * (y3 - y4)
        -
        (y1 - y2)
        * (x3 * y4 - y3 * x4)
    ) / denominator

    return np.array(
        [px, py],
        dtype=np.float32
    )


# --------------------------------------------------
# EXTEND THE TWO GOOD SIDELINES BACKWARDS
#
# Left singles sideline:
# P5 -> P7
#
# Right singles sideline:
# P6 -> P8
#
# Their intersections with the far baseline
# give us P1 and P2.
# --------------------------------------------------

p1 = line_intersection(
    p5,
    p7,
    baseline_a,
    baseline_b
)

p2 = line_intersection(
    p6,
    p8,
    baseline_a,
    baseline_b
)


print()
print("Calculated far corners:")
print("P1:", p1)
print("P2:", p2)


# --------------------------------------------------
# BUILD HOMOGRAPHY FROM FOUR OUTER CORNERS
# --------------------------------------------------

COURT_WIDTH = 8.23
COURT_LENGTH = 23.77
SERVICE = 5.485


world_corners = np.array([

    [0, 0],

    [COURT_WIDTH, 0],

    [COURT_WIDTH, COURT_LENGTH],

    [0, COURT_LENGTH]

], dtype=np.float32)


image_corners = np.array([

    p1,
    p2,
    p8,
    p7

], dtype=np.float32)


H = cv2.getPerspectiveTransform(
    world_corners,
    image_corners
)


# --------------------------------------------------
# PROJECT WORLD -> IMAGE
# --------------------------------------------------

def project(points):

    points = np.array(
        [points],
        dtype=np.float32
    )

    projected = cv2.perspectiveTransform(
        points,
        H
    )

    return projected[0]


# --------------------------------------------------
# REBUILD ALL 8 LANDMARKS
# --------------------------------------------------

world_landmarks = np.array([

    # Far baseline
    [0, 0],
    [COURT_WIDTH, 0],

    # Far service line
    [0, SERVICE],
    [COURT_WIDTH, SERVICE],

    # Near service line
    [0, COURT_LENGTH - SERVICE],
    [COURT_WIDTH, COURT_LENGTH - SERVICE],

    # Near baseline
    [0, COURT_LENGTH],
    [COURT_WIDTH, COURT_LENGTH]

], dtype=np.float32)


new_points = project(
    world_landmarks
)


# --------------------------------------------------
# SAVE POINTS
# --------------------------------------------------

np.save(
    OUTPUT_POINTS,
    new_points
)


# --------------------------------------------------
# DRAW
# --------------------------------------------------

output = frame.copy()


# Outer singles court
outer_indices = [
    (0, 1),
    (1, 7),
    (7, 6),
    (6, 0)
]

for a, b in outer_indices:

    pa = tuple(
        np.round(
            new_points[a]
        ).astype(int)
    )

    pb = tuple(
        np.round(
            new_points[b]
        ).astype(int)
    )

    cv2.line(
        output,
        pa,
        pb,
        (255, 0, 255),
        3
    )


# Far service line
cv2.line(
    output,
    tuple(np.round(new_points[2]).astype(int)),
    tuple(np.round(new_points[3]).astype(int)),
    (255, 0, 255),
    2
)


# Near service line
cv2.line(
    output,
    tuple(np.round(new_points[4]).astype(int)),
    tuple(np.round(new_points[5]).astype(int)),
    (255, 0, 255),
    2
)


# Centre service line
centre = project(
    np.array([

        [
            COURT_WIDTH / 2,
            SERVICE
        ],

        [
            COURT_WIDTH / 2,
            COURT_LENGTH - SERVICE
        ]

    ], dtype=np.float32)
)

cv2.line(
    output,
    tuple(np.round(centre[0]).astype(int)),
    tuple(np.round(centre[1]).astype(int)),
    (255, 0, 255),
    2
)


# --------------------------------------------------
# DRAW POINTS
# --------------------------------------------------

for i, point in enumerate(
    new_points,
    start=1
):

    x = int(round(point[0]))
    y = int(round(point[1]))

    cv2.circle(
        output,
        (x, y),
        7,
        (0, 255, 255),
        -1
    )

    cv2.putText(
        output,
        str(i),
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


# --------------------------------------------------
# BENCHMARK ONLY
#
# Manual points are NOT involved in calculation.
# --------------------------------------------------

manual = np.load(
    "data/court_points_8.npy"
).astype(np.float32)

errors = np.linalg.norm(
    new_points - manual,
    axis=1
)


print()
print("RESULTS")
print("-" * 60)

for i in range(8):

    print(
        f"Point {i + 1}: "
        f"AUTO=({new_points[i][0]:.1f}, "
        f"{new_points[i][1]:.1f}) | "
        f"MANUAL=({manual[i][0]:.1f}, "
        f"{manual[i][1]:.1f}) | "
        f"error={errors[i]:.1f}px"
    )


print()
print(
    f"Mean error: "
    f"{errors.mean():.1f}px"
)

print(
    f"Max error: "
    f"{errors.max():.1f}px"
)


# --------------------------------------------------
# SAVE IMAGE
# --------------------------------------------------

cv2.imwrite(
    OUTPUT_PATH,
    output
)

print()
print(f"Saved: {OUTPUT_PATH}")
print(f"Saved: {OUTPUT_POINTS}")