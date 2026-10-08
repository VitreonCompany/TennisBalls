import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from court_detector import detect_court


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "videos/test_match.mp4"

CSV_OUTPUT = "output/far_tracking_test.csv"
VIDEO_OUTPUT = "output/far_tracking_test.mp4"

START_SECONDS = 60
DURATION_SECONDS = 10

# TEMPORARY experimental correction.
# We are testing this, NOT adopting it permanently.
GROUND_OFFSET_RATIO = 0.27


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    raise RuntimeError("Could not determine video FPS")

start_frame = int(
    START_SECONDS * fps
)

frames_to_process = int(
    DURATION_SECONDS * fps
)

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    start_frame
)

success, first_frame = cap.read()

if not success:
    raise RuntimeError(
        "Could not read starting frame"
    )


height, width = first_frame.shape[:2]


print()
print("FAR PLAYER TRACKING TEST")
print("=" * 70)

print(f"FPS: {fps:.2f}")
print(f"Frames: {frames_to_process}")
print(
    f"Ground offset ratio: "
    f"{GROUND_OFFSET_RATIO:.2f}"
)


# ============================================================
# DETECT COURT ONCE
#
# Fixed camera, so we don't need to redetect it every frame.
# ============================================================

print()
print("Detecting court...")

court_result = detect_court(
    first_frame
)

H_image_to_court = np.linalg.inv(
    court_result["homography"]
).astype(np.float64)

far_baseline = court_result[
    "far_baseline"
].astype(np.float64)

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
# FAR BASELINE Y AT GIVEN X
# ============================================================

def baseline_y_at_x(x):

    x1, y1, x2, y2 = far_baseline

    if abs(x2 - x1) < 1e-6:
        return (y1 + y2) / 2

    return (
        y1
        +
        (x - x1)
        *
        (y2 - y1)
        /
        (x2 - x1)
    )


# ============================================================
# YOLO
# ============================================================

print()
print("Loading YOLO...")

model = YOLO(
    "yolo11n.pt"
)


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    VIDEO_OUTPUT,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError(
        "Could not create output video"
    )


# ============================================================
# RESET TO START
# ============================================================

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    start_frame
)


rows = []

previous_x = None


# ============================================================
# PROCESS FRAMES
# ============================================================

print()
print("Tracking...")
print()


for local_frame in range(
    frames_to_process
):

    success, frame = cap.read()

    if not success:
        break


    # --------------------------------------------------------
    # FAR COURT CROP
    # --------------------------------------------------------

    crop_x1 = 0
    crop_y1 = 120
    crop_x2 = 1000
    crop_y2 = 520

    crop = frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]


    # --------------------------------------------------------
    # PERSON DETECTION
    # --------------------------------------------------------

    results = model.predict(
        crop,
        classes=[0],
        conf=0.10,
        imgsz=1280,
        verbose=False
    )


    candidates = []


    for box in results[0].boxes:

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        confidence = float(
            box.conf[0]
        )

        x1 += crop_x1
        x2 += crop_x1

        y1 += crop_y1
        y2 += crop_y1


        box_width = x2 - x1
        box_height = y2 - y1

        centre_x = (
            x1 + x2
        ) / 2


        # --------------------------------------------
        # Broad far-player filtering
        # --------------------------------------------

        if centre_x > width * 0.55:
            continue

        if box_height < 20:
            continue

        if box_height > 180:
            continue

        if y2 < 300:
            continue

        if y2 > 520:
            continue


        # --------------------------------------------
        # Candidate score
        #
        # Confidence matters most.
        #
        # Once tracking has begun, also prefer a
        # detection near the previous X position.
        # --------------------------------------------

        score = confidence

        if previous_x is not None:

            x_distance = abs(
                centre_x - previous_x
            )

            score -= (
                x_distance / 1000
            )


        candidates.append(
            {
                "score": score,
                "confidence": confidence,

                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,

                "centre_x": centre_x,

                "width": box_width,
                "height": box_height
            }
        )


    # ========================================================
    # NO DETECTION
    # ========================================================

    if not candidates:

        rows.append(
            {
                "frame": (
                    start_frame
                    + local_frame
                ),

                "time_seconds": (
                    START_SECONDS
                    +
                    local_frame / fps
                ),

                "detected": False,

                "confidence": np.nan,

                "box_height": np.nan,

                "box_bottom_y": np.nan,

                "corrected_ground_y": np.nan,

                "court_x": np.nan,

                "court_y": np.nan,

                "baseline_y": np.nan
            }
        )

        writer.write(frame)

        continue


    # ========================================================
    # BEST FAR PLAYER
    # ========================================================

    best = max(
        candidates,
        key=lambda item:
            item["score"]
    )


    previous_x = best[
        "centre_x"
    ]


    x1 = best["x1"]
    y1 = best["y1"]

    x2 = best["x2"]
    y2 = best["y2"]

    box_height = best[
        "height"
    ]

    centre_x = best[
        "centre_x"
    ]


    # ========================================================
    # TEMPORARY GROUND CORRECTION
    # ========================================================

    corrected_ground_y = (
        y2
        +
        GROUND_OFFSET_RATIO
        * box_height
    )


    # ========================================================
    # COURT POSITION
    # ========================================================

    court_point = image_to_court(
        centre_x,
        corrected_ground_y
    )


    court_x = float(
        court_point[0]
    )

    court_y = float(
        court_point[1]
    )


    baseline_y = baseline_y_at_x(
        centre_x
    )


    # ========================================================
    # SAVE DATA
    # ========================================================

    rows.append(
        {
            "frame": (
                start_frame
                + local_frame
            ),

            "time_seconds": (
                START_SECONDS
                +
                local_frame / fps
            ),

            "detected": True,

            "confidence": best[
                "confidence"
            ],

            "box_height": box_height,

            "box_bottom_y": y2,

            "corrected_ground_y":
                corrected_ground_y,

            "court_x": court_x,

            "court_y": court_y,

            "baseline_y": baseline_y
        }
    )


    # ========================================================
    # DRAW
    # ========================================================

    cv2.rectangle(
        frame,
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


    # YOLO bottom = red

    box_point = (
        int(round(centre_x)),
        int(round(y2))
    )

    cv2.circle(
        frame,
        box_point,
        5,
        (0, 0, 255),
        -1
    )


    # Corrected ground = green

    corrected_point = (
        int(round(centre_x)),
        int(
            round(
                corrected_ground_y
            )
        )
    )

    cv2.circle(
        frame,
        corrected_point,
        6,
        (0, 255, 0),
        -1
    )


    # Baseline at player's X = blue

    baseline_point = (
        int(round(centre_x)),
        int(round(baseline_y))
    )

    cv2.circle(
        frame,
        baseline_point,
        5,
        (255, 0, 0),
        -1
    )


    text = (
        f"court "
        f"({court_x:.2f}, "
        f"{court_y:.2f})m"
    )


    cv2.putText(
        frame,
        text,
        (
            int(round(x1)),
            max(
                25,
                int(round(y1)) - 8
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )


    writer.write(
        frame
    )


    # Progress

    if (
        local_frame % 50
        == 0
    ):

        print(
            f"Processed "
            f"{local_frame}/"
            f"{frames_to_process}"
        )


# ============================================================
# FINISH
# ============================================================

cap.release()
writer.release()


df = pd.DataFrame(
    rows
)

df.to_csv(
    CSV_OUTPUT,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

detected = df[
    df["detected"] == True
].copy()


print()
print("RESULTS")
print("=" * 70)

print(
    f"Frames processed: "
    f"{len(df)}"
)

print(
    f"Frames detected: "
    f"{len(detected)}"
)

if len(df) > 0:

    print(
        f"Detection rate: "
        f"{100 * len(detected) / len(df):.1f}%"
    )


if len(detected) > 0:

    print()
    print("COURT Y")

    print(
        f"Mean: "
        f"{detected['court_y'].mean():.2f}m"
    )

    print(
        f"Median: "
        f"{detected['court_y'].median():.2f}m"
    )

    print(
        f"Min: "
        f"{detected['court_y'].min():.2f}m"
    )

    print(
        f"Max: "
        f"{detected['court_y'].max():.2f}m"
    )

    print()

    print("BOX HEIGHT")

    print(
        f"Mean: "
        f"{detected['box_height'].mean():.1f}px"
    )

    print(
        f"Min: "
        f"{detected['box_height'].min():.1f}px"
    )

    print(
        f"Max: "
        f"{detected['box_height'].max():.1f}px"
    )


print()
print(
    f"Saved CSV: {CSV_OUTPUT}"
)

print(
    f"Saved video: {VIDEO_OUTPUT}"
)