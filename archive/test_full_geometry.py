import cv2
import numpy as np


# ============================================================
# FILES
# ============================================================

VIDEO_PATH = "videos/test_match.mp4"

AI_POINTS_PATH = "data/ai_court_points_v2.npy"
MANUAL_POINTS_PATH = "data/court_points_8.npy"

OUTPUT_PATH = "output/test_full_geometry.jpg"


# ============================================================
# RESULTS FROM OUR AUTOMATIC CV TESTS
# ============================================================

# Automatically detected sideline vanishing point
VP = np.array(
    [-287.17292619, 337.27677988],
    dtype=np.float64
)

# Automatically detected far baseline
FAR_BASELINE_Y = 447.0


# ============================================================
# REAL TENNIS COURT DIMENSIONS
# ============================================================

COURT_WIDTH = 8.23
COURT_LENGTH = 23.77

BASELINE_TO_SERVICE = 5.485


# ============================================================
# LOAD VIDEO FRAME
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

cap.set(
    cv2.CAP_PROP_POS_MSEC,
    60 * 1000
)

success, frame = cap.read()

cap.release()

if not success:
    raise RuntimeError(
        "Could not read video frame"
    )


# ============================================================
# LOAD GOOD AI NEAR CORNERS
#
# We only use points 7 and 8.
# ============================================================

ai_points = np.load(
    AI_POINTS_PATH
).astype(np.float64)

p7 = ai_points[6]
p8 = ai_points[7]


# ============================================================
# CALCULATE FAR CORNERS 1 AND 2
# ============================================================

def intersect_with_horizontal(
    vp,
    point,
    target_y
):

    vx, vy = vp
    px, py = point

    t = (
        target_y - vy
    ) / (
        py - vy
    )

    x = vx + t * (px - vx)

    return np.array(
        [x, target_y],
        dtype=np.float64
    )


p1 = intersect_with_horizontal(
    VP,
    p7,
    FAR_BASELINE_Y
)

p2 = intersect_with_horizontal(
    VP,
    p8,
    FAR_BASELINE_Y
)


# ============================================================
# DEFINE REAL-WORLD OUTER COURT CORNERS
#
# Coordinate system:
#
# far baseline = y 0
# near baseline = y 23.77
# ============================================================

world_outer = np.array(
    [
        [0.0, 0.0],                 # point 1
        [COURT_WIDTH, 0.0],         # point 2
        [0.0, COURT_LENGTH],        # point 7
        [COURT_WIDTH, COURT_LENGTH] # point 8
    ],
    dtype=np.float32
)


image_outer = np.array(
    [
        p1,
        p2,
        p7,
        p8
    ],
    dtype=np.float32
)


# ============================================================
# CREATE HOMOGRAPHY
#
# REAL COURT -> IMAGE
# ============================================================

H = cv2.getPerspectiveTransform(
    world_outer,
    image_outer
)


# ============================================================
# DEFINE ALL 8 REAL COURT POINTS
# ============================================================

world_points = np.array(
    [
        [0.0, 0.0],                         # 1
        [COURT_WIDTH, 0.0],                 # 2

        [0.0, BASELINE_TO_SERVICE],          # 3
        [COURT_WIDTH, BASELINE_TO_SERVICE],  # 4

        [
            0.0,
            COURT_LENGTH - BASELINE_TO_SERVICE
        ],                                   # 5

        [
            COURT_WIDTH,
            COURT_LENGTH - BASELINE_TO_SERVICE
        ],                                   # 6

        [0.0, COURT_LENGTH],                 # 7
        [COURT_WIDTH, COURT_LENGTH]          # 8
    ],
    dtype=np.float32
)


# OpenCV expects shape:
# (N, 1, 2)

world_points_cv = world_points.reshape(
    -1,
    1,
    2
)


# ============================================================
# PROJECT COURT INTO IMAGE
# ============================================================

projected = cv2.perspectiveTransform(
    world_points_cv,
    H
).reshape(
    -1,
    2
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("FULL GEOMETRY RESULT")
print("=" * 65)

for i, point in enumerate(
    projected,
    start=1
):

    print(
        f"Point {i}: "
        f"({point[0]:.1f}, {point[1]:.1f})"
    )


# ============================================================
# HIDDEN MANUAL BENCHMARK
#
# These manual coordinates are NOT used in the calculation.
# ============================================================

manual = np.load(
    MANUAL_POINTS_PATH
).astype(np.float32)


errors = np.linalg.norm(
    projected - manual,
    axis=1
)


print()
print("HIDDEN MANUAL BENCHMARK")
print("=" * 65)


for i in range(8):

    print(
        f"Point {i + 1}: "
        f"AUTO=({projected[i][0]:.1f}, "
        f"{projected[i][1]:.1f}) | "
        f"MANUAL=({manual[i][0]:.1f}, "
        f"{manual[i][1]:.1f}) | "
        f"ERROR={errors[i]:.1f}px"
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


# ============================================================
# DRAW RESULT
# ============================================================

output = frame.copy()


# Court lines

court_lines = [

    # Left singles sideline
    (0, 2),
    (2, 4),
    (4, 6),

    # Right singles sideline
    (1, 3),
    (3, 5),
    (5, 7),

    # Far baseline
    (0, 1),

    # Far service line
    (2, 3),

    # Near service line
    (4, 5),

    # Near baseline
    (6, 7)
]


for a, b in court_lines:

    pa = tuple(
        np.round(
            projected[a]
        ).astype(int)
    )

    pb = tuple(
        np.round(
            projected[b]
        ).astype(int)
    )

    cv2.line(
        output,
        pa,
        pb,
        (255, 0, 255),
        3
    )


# ============================================================
# DRAW POINTS
# ============================================================

for i, point in enumerate(
    projected,
    start=1
):

    x = int(
        round(point[0])
    )

    y = int(
        round(point[1])
    )

    cv2.circle(
        output,
        (x, y),
        8,
        (0, 255, 255),
        -1
    )

    cv2.putText(
        output,
        str(i),
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
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
    f"Saved: {OUTPUT_PATH}"
)