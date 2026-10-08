import cv2
import numpy as np

# Load the 4 points you clicked
image_points = np.load("data/court_points.npy").astype(np.float32)

# Your clicked order:
# 1 = far-left
# 2 = far-right
# 3 = near-right
# 4 = near-left

# Regulation singles court dimensions in metres
COURT_WIDTH = 8.23
COURT_LENGTH = 23.77

# Corresponding real-world court coordinates
court_points = np.array([
    [0.0, 0.0],                    # far-left
    [COURT_WIDTH, 0.0],            # far-right
    [COURT_WIDTH, COURT_LENGTH],   # near-right
    [0.0, COURT_LENGTH],           # near-left
], dtype=np.float32)

# Calculate transformation
H = cv2.getPerspectiveTransform(image_points, court_points)

# Save it - we'll use this constantly later
np.save("data/homography.npy", H)

print("Homography created!")
print()
print(H)


def pixel_to_court(x, y):
    pixel = np.array([[[x, y]]], dtype=np.float32)

    transformed = cv2.perspectiveTransform(pixel, H)

    court_x = transformed[0][0][0]
    court_y = transformed[0][0][1]

    return court_x, court_y


# Test our four corners
print("\nTesting clicked points:")

for i, (x, y) in enumerate(image_points, start=1):
    cx, cy = pixel_to_court(x, y)

    print(
        f"Point {i}: pixel ({x:.0f}, {y:.0f}) "
        f"-> court ({cx:.2f}m, {cy:.2f}m)"
    )