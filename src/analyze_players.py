import cv2
import numpy as np
from ultralytics import YOLO

from court_detector import detect_court


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "videos/test_match.mp4"
OUTPUT_PATH = "output/analyze_players.jpg"

TIME_SECONDS = 60

COURT_WIDTH = 8.23
COURT_LENGTH = 23.77


# ============================================================
# LOAD FRAME
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

cap.set(
    cv2.CAP_PROP_POS_MSEC,
    TIME_SECONDS * 1000
)

success, frame = cap.read()

cap.release()

if not success:
    raise RuntimeError(
        "Could not read video frame"
    )


print()
print("PLAYER + AUTOMATIC COURT TEST")
print("=" * 70)


# ============================================================
# AUTOMATIC COURT DETECTION
# ============================================================

print()
print("Detecting court...")

court_result = detect_court(frame)

court_points = court_result["points"]

# court_result["homography"] maps:
#
# COURT -> IMAGE
#
# For players we need:
#
# IMAGE -> COURT

H_court_to_image = court_result[
    "homography"
]

H_image_to_court = np.linalg.inv(
    H_court_to_image
)


print("Court detected.")


# ============================================================
# LOAD YOLO
# ============================================================

print()
print("Loading YOLO...")

model = YOLO(
    "yolo11n.pt"
)


# ============================================================
# FULL-FRAME PERSON DETECTION
#
# This should reliably find the near player.
# ============================================================

results = model.predict(
    frame,
    classes=[0],
    conf=0.10,
    imgsz=1280,
    verbose=False
)

full_detections = []


for box in results[0].boxes:

    xyxy = (
        box.xyxy[0]
        .cpu()
        .numpy()
    )

    confidence = float(
        box.conf[0]
    )

    x1, y1, x2, y2 = xyxy

    full_detections.append(
        {
            "box": np.array(
                [x1, y1, x2, y2]
            ),
            "confidence": confidence
        }
    )


# ============================================================
# FAR-PLAYER CROP
#
# We already proved this crop dramatically improves detection
# of the tiny far player.
#
# Later we'll derive this region automatically from the court.
# ============================================================

crop_x1 = 0
crop_y1 = 120
crop_x2 = 1000
crop_y2 = 520

far_crop = frame[
    crop_y1:crop_y2,
    crop_x1:crop_x2
]


far_results = model.predict(
    far_crop,
    classes=[0],
    conf=0.10,
    imgsz=1280,
    verbose=False
)


far_detections = []


for box in far_results[0].boxes:

    xyxy = (
        box.xyxy[0]
        .cpu()
        .numpy()
    )

    confidence = float(
        box.conf[0]
    )

    x1, y1, x2, y2 = xyxy

    # Convert crop coordinates back to
    # full-image coordinates.

    x1 += crop_x1
    x2 += crop_x1

    y1 += crop_y1
    y2 += crop_y1

    far_detections.append(
        {
            "box": np.array(
                [x1, y1, x2, y2]
            ),
            "confidence": confidence
        }
    )


# ============================================================
# IMAGE POINT -> COURT METRES
# ============================================================

def image_to_court(point):

    point = np.array(
        [[point]],
        dtype=np.float32
    )

    mapped = cv2.perspectiveTransform(
        point,
        H_image_to_court.astype(
            np.float64
        )
    )

    return mapped[
        0,
        0
    ]


# ============================================================
# CONVERT DETECTION TO GROUND POSITION
#
# Bottom-centre of person bounding box approximates feet.
# ============================================================

def detection_ground_position(
    detection
):

    x1, y1, x2, y2 = (
        detection["box"]
    )

    foot_x = (
        x1 + x2
    ) / 2

    foot_y = y2

    image_point = np.array(
        [
            foot_x,
            foot_y
        ],
        dtype=np.float32
    )

    court_point = image_to_court(
        image_point
    )

    return (
        image_point,
        court_point
    )


# ============================================================
# CHOOSE NEAR PLAYER
#
# Near player should map beyond / around the near baseline.
# We select the strongest plausible full-frame detection.
# ============================================================

near_candidates = []


for detection in full_detections:

    image_point, court_point = (
        detection_ground_position(
            detection
        )
    )

    court_x, court_y = court_point

    # Broad bounds intentionally.
    #
    # Players can stand outside the court.

    if not (
        -5 <= court_x <= COURT_WIDTH + 5
    ):
        continue

    if not (
        COURT_LENGTH * 0.45
        <= court_y
        <= COURT_LENGTH + 15
    ):
        continue

    near_candidates.append(
        (
            detection["confidence"],
            detection,
            image_point,
            court_point
        )
    )


near_candidates.sort(
    key=lambda item:
        item[0],
    reverse=True
)


near_player = (
    near_candidates[0]
    if near_candidates
    else None
)


# ============================================================
# CHOOSE FAR PLAYER
#
# Use zoomed crop detections.
# ============================================================

far_candidates = []


for detection in far_detections:

    image_point, court_point = (
        detection_ground_position(
            detection
        )
    )

    court_x, court_y = court_point

    if not (
        -5 <= court_x <= COURT_WIDTH + 5
    ):
        continue

    # Far player may stand behind far baseline,
    # therefore allow negative y.

    if not (
        -15
        <= court_y
        <= COURT_LENGTH * 0.55
    ):
        continue

    far_candidates.append(
        (
            detection["confidence"],
            detection,
            image_point,
            court_point
        )
    )


far_candidates.sort(
    key=lambda item:
        item[0],
    reverse=True
)


far_player = (
    far_candidates[0]
    if far_candidates
    else None
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("PLAYER RESULTS")
print("=" * 70)


if near_player is not None:

    conf, detection, image_p, court_p = (
        near_player
    )

    print(
        f"NEAR PLAYER | "
        f"confidence={conf:.2f} | "
        f"pixel=({image_p[0]:.1f}, "
        f"{image_p[1]:.1f}) | "
        f"court=({court_p[0]:.2f}m, "
        f"{court_p[1]:.2f}m)"
    )

else:

    print(
        "NEAR PLAYER: not detected"
    )


if far_player is not None:

    conf, detection, image_p, court_p = (
        far_player
    )

    print(
        f"FAR PLAYER  | "
        f"confidence={conf:.2f} | "
        f"pixel=({image_p[0]:.1f}, "
        f"{image_p[1]:.1f}) | "
        f"court=({court_p[0]:.2f}m, "
        f"{court_p[1]:.2f}m)"
    )

else:

    print(
        "FAR PLAYER: not detected"
    )


# ============================================================
# DRAW RESULT
# ============================================================

output = frame.copy()


# ------------------------------------------------------------
# Draw automatically detected singles court
# ------------------------------------------------------------

court_lines = [
    (0, 2),
    (2, 4),
    (4, 6),

    (1, 3),
    (3, 5),
    (5, 7),

    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7)
]


for a, b in court_lines:

    pa = tuple(
        np.round(
            court_points[a]
        ).astype(int)
    )

    pb = tuple(
        np.round(
            court_points[b]
        ).astype(int)
    )

    cv2.line(
        output,
        pa,
        pb,
        (255, 0, 255),
        2
    )


# ============================================================
# DRAW PLAYER
# ============================================================

def draw_player(
    item,
    label
):

    if item is None:
        return

    conf, detection, image_p, court_p = (
        item
    )

    x1, y1, x2, y2 = (
        detection["box"]
        .astype(int)
    )

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        3
    )

    foot = (
        int(round(image_p[0])),
        int(round(image_p[1]))
    )

    cv2.circle(
        output,
        foot,
        8,
        (0, 255, 255),
        -1
    )

    text = (
        f"{label} "
        f"({court_p[0]:.2f}m, "
        f"{court_p[1]:.2f}m)"
    )

    cv2.putText(
        output,
        text,
        (
            x1,
            max(
                30,
                y1 - 10
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


draw_player(
    near_player,
    "NEAR"
)

draw_player(
    far_player,
    "FAR"
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