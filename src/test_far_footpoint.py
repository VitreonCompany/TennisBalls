import cv2
import numpy as np
from ultralytics import YOLO

from court_detector import detect_court


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "videos/test_match.mp4"

OUTPUT_PATH = "output/far_footpoint_test.jpg"
CROP_OUTPUT_PATH = "output/far_footpoint_crop.jpg"

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
    raise RuntimeError("Could not read video frame")


height, width = frame.shape[:2]


print()
print("FAR PLAYER FOOTPOINT TEST")
print("=" * 70)


# ============================================================
# AUTOMATIC COURT
# ============================================================

print()
print("Detecting court...")

court_result = detect_court(frame)

H_court_to_image = court_result[
    "homography"
].astype(np.float64)

H_image_to_court = np.linalg.inv(
    H_court_to_image
)

print("Court detected.")


# ============================================================
# IMAGE -> COURT
# ============================================================

def image_to_court(x, y):

    point = np.array(
        [[[x, y]]],
        dtype=np.float32
    )

    mapped = cv2.perspectiveTransform(
        point,
        H_image_to_court
    )

    return mapped[0, 0]


# ============================================================
# FAR PLAYER CROP
#
# Same crop that previously gave us ~95% detection.
# ============================================================

crop_x1 = 0
crop_y1 = 120
crop_x2 = 1000
crop_y2 = 520

crop = frame[
    crop_y1:crop_y2,
    crop_x1:crop_x2
].copy()


# ============================================================
# YOLO
# ============================================================

print()
print("Running far-player detector...")

model = YOLO("yolo11n.pt")

results = model.predict(
    crop,
    classes=[0],
    conf=0.10,
    imgsz=1280,
    verbose=False
)


detections = []


for box in results[0].boxes:

    x1, y1, x2, y2 = (
        box.xyxy[0]
        .cpu()
        .numpy()
    )

    confidence = float(
        box.conf[0]
    )

    # Back to full-frame coordinates

    full_x1 = x1 + crop_x1
    full_y1 = y1 + crop_y1
    full_x2 = x2 + crop_x1
    full_y2 = y2 + crop_y1

    centre_x = (
        full_x1 + full_x2
    ) / 2

    detections.append(
        {
            "box": np.array(
                [
                    full_x1,
                    full_y1,
                    full_x2,
                    full_y2
                ],
                dtype=np.float64
            ),
            "confidence": confidence,
            "centre_x": centre_x
        }
    )


if not detections:
    raise RuntimeError(
        "Far player was not detected"
    )


# ============================================================
# CHOOSE FAR PLAYER
#
# For this test the far player is expected in the left half.
# ============================================================

far_candidates = [
    d for d in detections
    if d["centre_x"] < width * 0.55
]


if not far_candidates:
    raise RuntimeError(
        "No plausible far-player detection"
    )


far_detection = max(
    far_candidates,
    key=lambda d:
        d["confidence"]
)


x1, y1, x2, y2 = (
    far_detection["box"]
)

confidence = (
    far_detection["confidence"]
)

box_width = x2 - x1
box_height = y2 - y1

centre_x = (
    x1 + x2
) / 2


print()
print("YOLO DETECTION")
print("=" * 70)

print(
    f"Confidence: {confidence:.3f}"
)

print(
    f"Box: "
    f"({x1:.1f}, {y1:.1f}) -> "
    f"({x2:.1f}, {y2:.1f})"
)

print(
    f"Box size: "
    f"{box_width:.1f} x {box_height:.1f}px"
)

print(
    f"YOLO bottom-centre: "
    f"({centre_x:.1f}, {y2:.1f})"
)


# ============================================================
# TEST A RANGE OF POSSIBLE GROUND POINTS
#
# Instead of blindly trusting y2, inspect positions below it.
#
# This is a diagnostic test only.
# ============================================================

offsets = [
    0,
    3,
    6,
    9,
    12,
    15,
    18,
    21,
    24
]


print()
print("GROUND-POINT TEST")
print("=" * 70)


ground_tests = []


for offset in offsets:

    test_y = y2 + offset

    court_point = image_to_court(
        centre_x,
        test_y
    )

    ground_tests.append(
        {
            "offset": offset,
            "pixel": np.array(
                [
                    centre_x,
                    test_y
                ]
            ),
            "court": court_point
        }
    )

    print(
        f"+{offset:2d}px | "
        f"pixel=({centre_x:.1f}, {test_y:.1f}) | "
        f"court=("
        f"{court_point[0]:.2f}m, "
        f"{court_point[1]:.2f}m)"
    )


# ============================================================
# ALSO TEST THE FAR BASELINE AT PLAYER X
#
# This gives us a useful geometric reference:
#
# Where does the automatically detected far baseline cross
# the player's horizontal position?
# ============================================================

far_segment = court_result[
    "far_baseline"
].astype(np.float64)

bx1, by1, bx2, by2 = far_segment


if abs(bx2 - bx1) > 1e-6:

    baseline_y_at_player = (
        by1
        +
        (
            centre_x - bx1
        )
        *
        (
            by2 - by1
        )
        /
        (
            bx2 - bx1
        )
    )

else:

    baseline_y_at_player = (
        by1 + by2
    ) / 2


baseline_court = image_to_court(
    centre_x,
    baseline_y_at_player
)


print()
print("FAR BASELINE REFERENCE")
print("=" * 70)

print(
    f"At player x={centre_x:.1f}:"
)

print(
    f"far baseline pixel y="
    f"{baseline_y_at_player:.1f}"
)

print(
    f"mapped court position="
    f"({baseline_court[0]:.2f}m, "
    f"{baseline_court[1]:.2f}m)"
)

print(
    f"YOLO bottom is "
    f"{baseline_y_at_player - y2:.1f}px "
    f"above the detected far baseline"
)


# ============================================================
# DRAW FULL FRAME
# ============================================================

output = frame.copy()


# YOLO box

cv2.rectangle(
    output,
    (
        int(round(x1)),
        int(round(y1))
    ),
    (
        int(round(x2)),
        int(round(y2))
    ),
    (0, 255, 255),
    2
)


# Original YOLO bottom

cv2.circle(
    output,
    (
        int(round(centre_x)),
        int(round(y2))
    ),
    6,
    (0, 0, 255),
    -1
)


# Offset test points

for test in ground_tests:

    px = int(
        round(
            test["pixel"][0]
        )
    )

    py = int(
        round(
            test["pixel"][1]
        )
    )

    offset = test[
        "offset"
    ]

    cv2.circle(
        output,
        (px, py),
        4,
        (255, 0, 255),
        -1
    )

    cv2.putText(
        output,
        f"+{offset}",
        (
            px + 8,
            py + 3
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 0, 255),
        1
    )


# Far baseline reference point

baseline_pixel = (
    int(round(centre_x)),
    int(round(baseline_y_at_player))
)

cv2.circle(
    output,
    baseline_pixel,
    7,
    (0, 255, 0),
    -1
)

cv2.putText(
    output,
    "FAR BASELINE",
    (
        baseline_pixel[0] + 10,
        baseline_pixel[1]
    ),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0, 255, 0),
    2
)


cv2.imwrite(
    OUTPUT_PATH,
    output
)


# ============================================================
# MAKE A LARGE DEBUG CROP
# ============================================================

padding_x = 80
padding_top = 60
padding_bottom = 80


debug_x1 = max(
    0,
    int(x1 - padding_x)
)

debug_y1 = max(
    0,
    int(y1 - padding_top)
)

debug_x2 = min(
    width,
    int(x2 + padding_x)
)

debug_y2 = min(
    height,
    int(
        max(
            y2,
            baseline_y_at_player
        )
        + padding_bottom
    )
)


debug_crop = output[
    debug_y1:debug_y2,
    debug_x1:debug_x2
]


# Enlarge it so we can actually inspect
# the tiny far player's feet.

debug_crop = cv2.resize(
    debug_crop,
    None,
    fx=4,
    fy=4,
    interpolation=cv2.INTER_NEAREST
)


cv2.imwrite(
    CROP_OUTPUT_PATH,
    debug_crop
)


print()
print(
    f"Saved: {OUTPUT_PATH}"
)

print(
    f"Saved enlarged crop: "
    f"{CROP_OUTPUT_PATH}"
)