import cv2
import numpy as np


# ============================================================
# FILES
# ============================================================

COURT_POINTS_PATH = "data/court_points_8.npy"
OUTPUT_PATH = "data/homography.npy"


# ============================================================
# REAL TENNIS COURT DIMENSIONS
# ============================================================

COURT_WIDTH = 8.23
COURT_LENGTH = 23.77

SERVICE_DISTANCE = 6.40


# ============================================================
# LOAD MANUALLY VERIFIED IMAGE POINTS
# ============================================================

image_points = np.load(
    COURT_POINTS_PATH
).astype(np.float32)

print()
print("Loaded image court points:")
print("=" * 60)

for i, point in enumerate(image_points, start=1):
    print(
        f"P{i}: "
        f"({point[0]:.1f}, {point[1]:.1f})"
    )


# ============================================================
# DEFINE MATCHING REAL-WORLD COURT POINTS
#
# Coordinate system:
#
# X:
#   left singles sideline  = -4.115 m
#   centre                 =  0.000 m
#   right singles sideline = +4.115 m
#
# Y:
#   net           =  0.000 m
#   far baseline  = -11.885 m
#   near baseline = +11.885 m
#
# Point order matches court_detector.py:
#
# P1 P2 = far baseline
# P3 P4 = far service line
# P5 P6 = near service line
# P7 P8 = near baseline
# ============================================================

HALF_WIDTH = COURT_WIDTH / 2.0
HALF_LENGTH = COURT_LENGTH / 2.0

world_points = np.array(
    [
        [-HALF_WIDTH, -HALF_LENGTH],   # P1
        [ HALF_WIDTH, -HALF_LENGTH],   # P2

        [-HALF_WIDTH, -SERVICE_DISTANCE],  # P3
        [ HALF_WIDTH, -SERVICE_DISTANCE],  # P4

        [-HALF_WIDTH,  SERVICE_DISTANCE],  # P5
        [ HALF_WIDTH,  SERVICE_DISTANCE],  # P6

        [-HALF_WIDTH,  HALF_LENGTH],   # P7
        [ HALF_WIDTH,  HALF_LENGTH],   # P8
    ],
    dtype=np.float32
)


# ============================================================
# CALCULATE IMAGE -> COURT HOMOGRAPHY
# ============================================================

H, mask = cv2.findHomography(
    image_points,
    world_points,
    method=0
)

if H is None:
    raise RuntimeError(
        "Could not calculate homography."
    )


# ============================================================
# SAVE
# ============================================================

np.save(
    OUTPUT_PATH,
    H
)

print()
print("Homography:")
print("=" * 60)
print(H)

print()
print(
    f"Saved homography to: {OUTPUT_PATH}"
)


# ============================================================
# VALIDATION
#
# Transform the original image landmarks back into court
# coordinates and compare against their known true locations.
# ============================================================

transformed = cv2.perspectiveTransform(
    image_points.reshape(-1, 1, 2),
    H
).reshape(-1, 2)

errors = np.linalg.norm(
    transformed - world_points,
    axis=1
)

print()
print("CALIBRATION CHECK")
print("=" * 60)

for i in range(len(image_points)):

    expected_x = world_points[i][0]
    expected_y = world_points[i][1]

    calculated_x = transformed[i][0]
    calculated_y = transformed[i][1]

    print(
        f"P{i + 1}: "
        f"expected=({expected_x:.3f}, {expected_y:.3f})  "
        f"calculated=({calculated_x:.3f}, {calculated_y:.3f})  "
        f"error={errors[i]:.3f}m"
    )

print()
print(
    f"Mean calibration error: "
    f"{errors.mean():.3f}m"
)

print(
    f"Maximum calibration error: "
    f"{errors.max():.3f}m"
)

print()
print("Finished.")