import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

VIDEO_PATH = "videos/test_match.mp4"
OUTPUT_CSV = "data/players_clean.csv"

START_TIME = 60
DURATION = 10

H = np.load("data/homography.npy")
model = YOLO("yolo11n.pt")


def pixel_to_court(x, y):
    point = np.array([[[x, y]]], dtype=np.float32)
    result = cv2.perspectiveTransform(point, H)[0][0]
    return float(result[0]), float(result[1])


cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_MSEC, START_TIME * 1000)

total_frames = int(DURATION * fps)

rows = []

for frame_num in range(total_frames):

    success, frame = cap.read()

    if not success:
        break

    time = START_TIME + frame_num / fps

    # -----------------------------
    # NEAR PLAYER
    # -----------------------------

    results = model(
        frame,
        classes=[0],
        conf=0.10,
        imgsz=1280,
        verbose=False
    )[0]

    near_candidates = []

    for box in results.boxes:

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        foot_x = (x1 + x2) / 2
        foot_y = y2

        cx, cy = pixel_to_court(foot_x, foot_y)

        # Near half / behind near baseline
        if 11.885 < cy < 30:

            near_candidates.append((
                float(box.conf[0]),
                foot_x,
                foot_y,
                cx,
                cy
            ))

    if near_candidates:

        # Highest confidence candidate
        near = max(near_candidates, key=lambda x: x[0])

        rows.append({
            "frame": frame_num,
            "time": time,
            "player": "P1",
            "pixel_x": near[1],
            "pixel_y": near[2],
            "court_x": near[3],
            "court_y": near[4],
            "confidence": near[0],
            "detected": True
        })


    # -----------------------------
    # FAR PLAYER - ZOOMED REGION
    # -----------------------------

    crop_x1 = 0
    crop_y1 = 120
    crop_x2 = 1000
    crop_y2 = 520

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    results_far = model(
        crop,
        classes=[0],
        conf=0.10,
        imgsz=1280,
        verbose=False
    )[0]

    far_candidates = []

    for box in results_far.boxes:

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Convert crop coordinates back to full frame
        x1 += crop_x1
        x2 += crop_x1
        y1 += crop_y1
        y2 += crop_y1

        foot_x = (x1 + x2) / 2
        foot_y = y2

        cx, cy = pixel_to_court(foot_x, foot_y)

        # Far half + some space behind baseline
        if -15 < cy < 11.885:

            far_candidates.append((
                float(box.conf[0]),
                foot_x,
                foot_y,
                cx,
                cy
            ))

    if far_candidates:

        far = max(far_candidates, key=lambda x: x[0])

        rows.append({
            "frame": frame_num,
            "time": time,
            "player": "P2",
            "pixel_x": far[1],
            "pixel_y": far[2],
            "court_x": far[3],
            "court_y": far[4],
            "confidence": far[0],
            "detected": True
        })

    if frame_num % 50 == 0:
        print(f"Processed {frame_num}/{total_frames}")


cap.release()

df = pd.DataFrame(rows)

# ------------------------------------------------
# Create one row for EVERY frame for BOTH players
# ------------------------------------------------

all_frames = pd.MultiIndex.from_product(
    [
        range(total_frames),
        ["P1", "P2"]
    ],
    names=["frame", "player"]
)

df = (
    df.set_index(["frame", "player"])
      .reindex(all_frames)
      .reset_index()
)

df["time"] = START_TIME + df["frame"] / fps

# Mark which positions were genuinely detected
df["detected"] = df["detected"].fillna(False)

# ------------------------------------------------
# Fill short gaps
# ------------------------------------------------

for player in ["P1", "P2"]:

    mask = df["player"] == player

    for column in [
        "pixel_x",
        "pixel_y",
        "court_x",
        "court_y"
    ]:

        df.loc[mask, column] = (
            df.loc[mask, column]
            .interpolate(
                method="linear",
                limit=10,
                limit_direction="both"
            )
        )

df.to_csv(OUTPUT_CSV, index=False)

print()
print("Finished!")
print(f"Saved: {OUTPUT_CSV}")

print()
print("Detection rates:")

for player in ["P1", "P2"]:

    player_data = df[df["player"] == player]

    rate = player_data["detected"].mean() * 100

    print(f"{player}: {rate:.1f}%")