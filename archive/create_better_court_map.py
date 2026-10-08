import cv2
import numpy as np

# Load your 8 manually clicked image points
image_points = np.load("data/court_points_8.npy").astype(np.float32)

# Regulation singles court dimensions
COURT_WIDTH = 8.23
COURT_LENGTH = 23.77

# Distance from baseline to service line
SERVICE_LINE_Y = 5.485

# Matching real-world court coordinates.
#
# Our coordinate system:
#
# (0,0) -------- (8.23,0)
#       FAR BASELINE
#
#       FAR SERVICE LINE
#
#            NET
#
#       NEAR SERVICE LINE
#
# (0,23.77) ---- (8.23,23.77)

court_points = np.array([

    # 1, 2 - far baseline
    [0.0, 0.0],
    [COURT_WIDTH, 0.0],

    # 3, 4 - far service line
    [0.0, SERVICE_LINE_Y],
    [COURT_WIDTH, SERVICE_LINE_Y],

    # 5, 6 - near service line
    [0.0, COURT_LENGTH - SERVICE_LINE_Y],
    [COURT_WIDTH, COURT_LENGTH - SERVICE_LINE_Y],

    # 7, 8 - near baseline
    [0.0, COURT_LENGTH],
    [COURT_WIDTH, COURT_LENGTH],

], dtype=np.float32)


# Find best-fit homography using all 8 points
H, mask = cv2.findHomography(
    image_points,
    court_points,
    method=0
)

np.save("data/homography_8.npy", H)

print("New homography:")
print(H)

print()
print("Landmark accuracy:")
print("-" * 65)

errors = []

for i, (pixel, expected) in enumerate(
    zip(image_points, court_points),
    start=1
):

    point = np.array(
        [[[pixel[0], pixel[1]]]],
        dtype=np.float32
    )

    predicted = cv2.perspectiveTransform(
        point,
        H
    )[0][0]

    error = np.linalg.norm(predicted - expected)

    errors.append(error)

    print(
        f"Point {i}: "
        f"expected ({expected[0]:.2f}, {expected[1]:.2f}) | "
        f"predicted ({predicted[0]:.2f}, {predicted[1]:.2f}) | "
        f"error = {error:.3f} m"
    )


print()
print("-" * 65)

print(f"Mean error: {np.mean(errors):.3f} m")
print(f"Max error:  {np.max(errors):.3f} m")

print()
print("Saved:")
print("data/homography_8.npy")