import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "videos/test_match.mp4"
OUTPUT_PATH = "output/player_court_positions.jpg"

# Load our court transformation
H = np.load("data/homography_8.npy")

# Load YOLO
model = YOLO("yolo11n.pt")

# Get the same frame at 60 seconds
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, 60 * 1000)

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read video.")


def pixel_to_court(x, y):
    """Convert image pixel coordinates into court coordinates (metres)."""

    point = np.array([[[x, y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, H)[0][0]

    return float(transformed[0]), float(transformed[1])


# Run YOLO
results = model(
    frame,
    conf=0.10,
    imgsz=1280,
    verbose=False
)[0]

player_number = 1

for box in results.boxes:

    class_id = int(box.cls[0])

    # COCO class 0 = person
    if class_id != 0:
        continue

    confidence = float(box.conf[0])

    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

    # Bottom-centre of bounding box ≈ player's position on ground
    foot_x = (x1 + x2) / 2
    foot_y = y2

    court_x, court_y = pixel_to_court(foot_x, foot_y)

    # Only accept people reasonably close to the tennis court
    # Small margin included because players can stand behind baseline.
    if not (-3 <= court_x <= 11.23 and -5 <= court_y <= 28.77):
        continue

    print(
        f"Player {player_number}: "
        f"confidence={confidence:.2f} | "
        f"pixel=({foot_x:.0f}, {foot_y:.0f}) | "
        f"court=({court_x:.2f}m, {court_y:.2f}m)"
    )

    # Draw bounding box
    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 255, 0),
        3
    )

    # Draw feet position
    cv2.circle(
        frame,
        (int(foot_x), int(foot_y)),
        8,
        (0, 0, 255),
        -1
    )

    label = f"P{player_number}: ({court_x:.2f}m, {court_y:.2f}m)"

    cv2.putText(
        frame,
        label,
        (int(x1), int(y1) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    player_number += 1


cv2.imwrite(OUTPUT_PATH, frame)

print()
print(f"Saved result to: {OUTPUT_PATH}")