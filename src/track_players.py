import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

VIDEO_PATH = "videos/test_match.mp4"
OUTPUT_CSV = "data/player_tracking.csv"
OUTPUT_VIDEO = "output/tracking_test.mp4"

START_TIME = 60       # seconds
DURATION = 10         # only test 10 seconds

H = np.load("data/homography.npy")
model = YOLO("yolo11n.pt")


def pixel_to_court(x, y):
    point = np.array([[[x, y]]], dtype=np.float32)
    result = cv2.perspectiveTransform(point, H)[0][0]
    return float(result[0]), float(result[1])


cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(cv2.CAP_PROP_POS_MSEC, START_TIME * 1000)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

data = []

total_frames = int(DURATION * fps)

for frame_number in range(total_frames):

    success, frame = cap.read()

    if not success:
        break

    # Track people between frames
    results = model.track(
        frame,
        persist=True,
        classes=[0],       # people only
        conf=0.10,
        imgsz=1280,
        verbose=False
    )[0]

    if results.boxes is not None:

        for box in results.boxes:

            if box.id is None:
                continue

            track_id = int(box.id[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Approximate ground position
            foot_x = (x1 + x2) / 2
            foot_y = y2

            court_x, court_y = pixel_to_court(foot_x, foot_y)

            # Ignore detections far away from our court
            if not (-3 <= court_x <= 11.23):
                continue

            if not (-5 <= court_y <= 28.77):
                continue

            time_seconds = START_TIME + (frame_number / fps)

            data.append({
                "frame": frame_number,
                "time": time_seconds,
                "track_id": track_id,
                "confidence": confidence,
                "pixel_x": foot_x,
                "pixel_y": foot_y,
                "court_x": court_x,
                "court_y": court_y
            })

            # Draw bounding box
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            # Ground point
            cv2.circle(
                frame,
                (int(foot_x), int(foot_y)),
                6,
                (0, 0, 255),
                -1
            )

            label = (
                f"ID {track_id} "
                f"({court_x:.2f}, {court_y:.2f})"
            )

            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    writer.write(frame)

    if frame_number % 50 == 0:
        print(f"Processed {frame_number}/{total_frames} frames")


cap.release()
writer.release()

df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)

print()
print("Finished!")
print(f"Tracking data: {OUTPUT_CSV}")
print(f"Tracking video: {OUTPUT_VIDEO}")
print()
print(df.head(20))