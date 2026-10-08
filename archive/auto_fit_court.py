import cv2
import numpy as np

VIDEO_PATH = "videos/test_match.mp4"
MANUAL_POINTS_PATH = "data/court_points_8.npy"

OUTPUT_IMAGE = "output/auto_court_fit.jpg"
OUTPUT_POINTS = "data/auto_court_points.npy"

# Regulation singles court
COURT_WIDTH = 8.23
COURT_LENGTH = 23.77
SERVICE = 5.485

# --------------------------------------------------
# Load our test frame
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, 60 * 1000)

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read video frame")

height, width = frame.shape[:2]


# --------------------------------------------------
# Build image evidence
#
# We want bright/light court markings, but we'll use
# distance-to-evidence rather than requiring an exact
# pixel match.
# --------------------------------------------------

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

lower = np.array([0, 0, 115])
upper = np.array([180, 145, 255])

mask = cv2.inRange(hsv, lower, upper)

# Restrict ourselves roughly to the lower part of image.
# This is NOT using your clicked court coordinates.
roi = np.zeros_like(mask)

roi[int(height * 0.25):int(height * 0.90), :] = 255

mask = cv2.bitwise_and(mask, roi)

kernel = np.ones((3, 3), np.uint8)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)

edges = cv2.Canny(mask, 50, 150)


# --------------------------------------------------
# Distance transform
#
# Every pixel now tells us how far it is from visual
# line evidence.
# --------------------------------------------------

inverse_edges = cv2.bitwise_not(edges)

distance = cv2.distanceTransform(
    inverse_edges,
    cv2.DIST_L2,
    3
)


# --------------------------------------------------
# Tennis court geometry
# --------------------------------------------------

# Four outer singles corners:
#
# 0 = far-left
# 1 = far-right
# 2 = near-right
# 3 = near-left

court_corners = np.array([
    [0.0, 0.0],
    [COURT_WIDTH, 0.0],
    [COURT_WIDTH, COURT_LENGTH],
    [0.0, COURT_LENGTH]
], dtype=np.float32)


def project_points(H, points):

    pts = np.array(
        [points],
        dtype=np.float32
    )

    return cv2.perspectiveTransform(
        pts,
        H
    )[0]


def sample_line(p1, p2, samples=100):

    xs = np.linspace(p1[0], p2[0], samples)
    ys = np.linspace(p1[1], p2[1], samples)

    return np.column_stack((xs, ys))


def score_projected_line(p1, p2):

    samples = sample_line(p1, p2)

    xs = np.round(samples[:, 0]).astype(int)
    ys = np.round(samples[:, 1]).astype(int)

    valid = (
        (xs >= 0) &
        (xs < width) &
        (ys >= 0) &
        (ys < height)
    )

    if valid.sum() < 20:
        return 50.0

    xs = xs[valid]
    ys = ys[valid]

    # Cap huge distances so a few bad pixels
    # don't completely dominate the score.
    d = np.minimum(distance[ys, xs], 25)

    return float(np.mean(d))


def score_court(H):

    # Full court lines expressed in real court coords
    lines_world = [

        # singles sidelines
        ([0, 0], [0, COURT_LENGTH]),
        ([COURT_WIDTH, 0], [COURT_WIDTH, COURT_LENGTH]),

        # baselines
        ([0, 0], [COURT_WIDTH, 0]),
        ([0, COURT_LENGTH], [COURT_WIDTH, COURT_LENGTH]),

        # service lines
        ([0, SERVICE], [COURT_WIDTH, SERVICE]),
        (
            [0, COURT_LENGTH - SERVICE],
            [COURT_WIDTH, COURT_LENGTH - SERVICE]
        ),

        # centre service line
        (
            [COURT_WIDTH / 2, SERVICE],
            [COURT_WIDTH / 2, COURT_LENGTH - SERVICE]
        )
    ]

    scores = []

    for a, b in lines_world:

        projected = project_points(
            H,
            np.array([a, b], dtype=np.float32)
        )

        scores.append(
            score_projected_line(
                projected[0],
                projected[1]
            )
        )

    return np.mean(scores)


# --------------------------------------------------
# Initial rough search
#
# These are broad IMAGE regions, not exact court
# coordinates.
# --------------------------------------------------

far_left_region = (
    (int(width * 0.00), int(width * 0.15)),
    (int(height * 0.30), int(height * 0.55))
)

far_right_region = (
    (int(width * 0.25), int(width * 0.60)),
    (int(height * 0.30), int(height * 0.55))
)

near_left_region = (
    (int(width * 0.35), int(width * 0.65)),
    (int(height * 0.55), int(height * 0.90))
)

near_right_region = (
    (int(width * 0.65), int(width * 0.98)),
    (int(height * 0.45), int(height * 0.80))
)


def random_point(region):

    (xmin, xmax), (ymin, ymax) = region

    return np.array([
        np.random.uniform(xmin, xmax),
        np.random.uniform(ymin, ymax)
    ], dtype=np.float32)


# Reproducible experiment
np.random.seed(42)

best_score = float("inf")
best_corners = None
best_H = None

ITERATIONS = 15000

print("Searching for court...")

for i in range(ITERATIONS):

    candidate = np.array([
        random_point(far_left_region),
        random_point(far_right_region),
        random_point(near_right_region),
        random_point(near_left_region)
    ], dtype=np.float32)

    # Basic sanity checks

    # Far baseline should be above near baseline
    if candidate[0][1] >= candidate[3][1]:
        continue

    if candidate[1][1] >= candidate[2][1]:
        continue

    # left/right ordering
    if candidate[0][0] >= candidate[1][0]:
        continue

    if candidate[3][0] >= candidate[2][0]:
        continue

    H = cv2.getPerspectiveTransform(
        court_corners,
        candidate
    )

    score = score_court(H)

    if score < best_score:

        best_score = score
        best_corners = candidate.copy()
        best_H = H.copy()

    if i % 1000 == 0:
        print(
            f"{i}/{ITERATIONS} "
            f"best score = {best_score:.3f}"
        )


if best_H is None:
    raise RuntimeError("Could not find a candidate court")


# --------------------------------------------------
# Local refinement
# --------------------------------------------------

print()
print("Refining...")

step_sizes = [30, 15, 7, 3, 1]

for step in step_sizes:

    improved = True

    while improved:

        improved = False

        for corner_index in range(4):

            for axis in range(2):

                for direction in [-1, 1]:

                    candidate = best_corners.copy()

                    candidate[
                        corner_index,
                        axis
                    ] += direction * step

                    H = cv2.getPerspectiveTransform(
                        court_corners,
                        candidate
                    )

                    score = score_court(H)

                    if score < best_score:

                        best_score = score
                        best_corners = candidate
                        best_H = H
                        improved = True


# --------------------------------------------------
# Generate our 8 landmarks AUTOMATICALLY
# --------------------------------------------------

court_landmarks = np.array([

    [0, 0],
    [COURT_WIDTH, 0],

    [0, SERVICE],
    [COURT_WIDTH, SERVICE],

    [0, COURT_LENGTH - SERVICE],
    [COURT_WIDTH, COURT_LENGTH - SERVICE],

    [0, COURT_LENGTH],
    [COURT_WIDTH, COURT_LENGTH]

], dtype=np.float32)

auto_points = project_points(
    best_H,
    court_landmarks
)

np.save(
    OUTPUT_POINTS,
    auto_points.astype(np.float32)
)


# --------------------------------------------------
# Draw complete predicted court
# --------------------------------------------------

output = frame.copy()

projected_corners = project_points(
    best_H,
    court_corners
)

for i in range(4):

    a = projected_corners[i]
    b = projected_corners[(i + 1) % 4]

    cv2.line(
        output,
        tuple(np.round(a).astype(int)),
        tuple(np.round(b).astype(int)),
        (255, 0, 255),
        3
    )


# Draw service lines + centre line

extra_lines = [
    ([0, SERVICE], [COURT_WIDTH, SERVICE]),
    (
        [0, COURT_LENGTH - SERVICE],
        [COURT_WIDTH, COURT_LENGTH - SERVICE]
    ),
    (
        [COURT_WIDTH / 2, SERVICE],
        [COURT_WIDTH / 2, COURT_LENGTH - SERVICE]
    )
]

for a, b in extra_lines:

    projected = project_points(
        best_H,
        np.array([a, b], dtype=np.float32)
    )

    cv2.line(
        output,
        tuple(np.round(projected[0]).astype(int)),
        tuple(np.round(projected[1]).astype(int)),
        (255, 0, 255),
        2
    )


# Draw 8 automatic landmarks

for i, point in enumerate(auto_points):

    x, y = np.round(point).astype(int)

    cv2.circle(
        output,
        (x, y),
        7,
        (0, 255, 255),
        -1
    )

    cv2.putText(
        output,
        str(i + 1),
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


cv2.imwrite(
    OUTPUT_IMAGE,
    output
)


# --------------------------------------------------
# BENCHMARK ONLY
#
# These points were NOT used during fitting.
# --------------------------------------------------

manual = np.load(
    MANUAL_POINTS_PATH
).astype(np.float32)

errors = np.linalg.norm(
    auto_points - manual,
    axis=1
)

print()
print("AUTOMATIC RESULTS")
print("-" * 65)

for i in range(8):

    print(
        f"Point {i + 1}: "
        f"AUTO=({auto_points[i][0]:.0f}, "
        f"{auto_points[i][1]:.0f}) | "
        f"MANUAL=({manual[i][0]:.0f}, "
        f"{manual[i][1]:.0f}) | "
        f"error={errors[i]:.1f}px"
    )

print()
print(f"Mean pixel error: {errors.mean():.1f}px")
print(f"Max pixel error:  {errors.max():.1f}px")
print(f"Geometry score:  {best_score:.3f}")

print()
print(f"Saved: {OUTPUT_POINTS}")
print(f"Saved: {OUTPUT_IMAGE}")