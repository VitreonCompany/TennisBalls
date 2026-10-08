import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from court_detector import detect_court


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "videos/test_match.mp4"

OUTPUT_VIDEO = "output/both_player_detection.mp4"
OUTPUT_CSV = "output/both_player_detection.csv"

START_SECONDS = 60
DURATION_SECONDS = 10


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    raise RuntimeError("Could not determine FPS")

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

start_frame = int(
    START_SECONDS * fps
)

frames_to_process = int(
    DURATION_SECONDS * fps
)


print()
print("BOTH PLAYER DETECTION TEST")
print("=" * 70)

print(f"FPS: {fps:.2f}")
print(f"Frames: {frames_to_process}")


# ============================================================
# GET FIRST FRAME
# ============================================================

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    start_frame
)

success, first_frame = cap.read()

if not success:
    raise RuntimeError(
        "Could not read starting frame"
    )


# ============================================================
# COURT
#
# We're only detecting the court once because this camera
# is fixed.
# ============================================================

print()
print("Detecting court...")

court_result = detect_court(
    first_frame
)

print("Court detected.")


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

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(
        *"mp4v"
    ),
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError(
        "Could not create output video"
    )


# ============================================================
# RESET VIDEO
# ============================================================

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    start_frame
)


rows = []

previous_far_x = None
previous_near_x = None


# ============================================================
# PROCESS FRAMES
# ============================================================

print()
print("Detecting players...")
print()


for local_frame in range(
    frames_to_process
):

    success, frame = cap.read()

    if not success:
        break


    frame_number = (
        start_frame
        +
        local_frame
    )

    time_seconds = (
        START_SECONDS
        +
        local_frame / fps
    )


    # ========================================================
    # FAR PLAYER
    #
    # Use the crop that already gave us 94.4% detection.
    # ========================================================

    far_crop_x1 = 0
    far_crop_y1 = 120
    far_crop_x2 = 1000
    far_crop_y2 = 520

    far_crop = frame[
        far_crop_y1:far_crop_y2,
        far_crop_x1:far_crop_x2
    ]


    far_results = model.predict(
        far_crop,
        classes=[0],
        conf=0.10,
        imgsz=1280,
        verbose=False
    )


    far_candidates = []


    for box in far_results[0].boxes:

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        confidence = float(
            box.conf[0]
        )


        # Crop -> full frame

        x1 += far_crop_x1
        x2 += far_crop_x1

        y1 += far_crop_y1
        y2 += far_crop_y1


        box_height = y2 - y1

        centre_x = (
            x1 + x2
        ) / 2


        # Broad far-player filters

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


        score = confidence


        # Prefer continuity once we know where the
        # far player was in the previous frame.

        if previous_far_x is not None:

            distance = abs(
                centre_x
                -
                previous_far_x
            )

            score -= (
                distance
                /
                800.0
            )


        far_candidates.append(
            {
                "score": score,
                "confidence": confidence,

                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),

                "centre_x":
                    float(centre_x)
            }
        )


    if far_candidates:

        far_player = max(
            far_candidates,
            key=lambda d:
                d["score"]
        )

        previous_far_x = (
            far_player[
                "centre_x"
            ]
        )

    else:

        far_player = None


    # ========================================================
    # NEAR PLAYER
    #
    # Near player is large enough that the full frame works
    # very well.
    # ========================================================

    near_results = model.predict(
        frame,
        classes=[0],
        conf=0.25,
        imgsz=1280,
        verbose=False
    )


    near_candidates = []


    for box in near_results[0].boxes:

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        confidence = float(
            box.conf[0]
        )

        box_height = y2 - y1

        centre_x = (
            x1 + x2
        ) / 2


        # Near player should be substantially larger
        # than the far player.

        if box_height < 120:
            continue


        # For this camera, the near player occupies the
        # right side of the image.
        #
        # Still deliberately broad.

        if centre_x < width * 0.45:
            continue


        score = confidence


        if previous_near_x is not None:

            distance = abs(
                centre_x
                -
                previous_near_x
            )

            score -= (
                distance
                /
                1200.0
            )


        near_candidates.append(
            {
                "score": score,
                "confidence": confidence,

                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),

                "centre_x":
                    float(centre_x)
            }
        )


    if near_candidates:

        near_player = max(
            near_candidates,
            key=lambda d:
                d["score"]
        )

        previous_near_x = (
            near_player[
                "centre_x"
            ]
        )

    else:

        near_player = None


    # ========================================================
    # SAVE FAR PLAYER
    # ========================================================

    if far_player is not None:

        rows.append(
            {
                "frame":
                    frame_number,

                "time_seconds":
                    time_seconds,

                "player":
                    "far",

                "detected":
                    True,

                "confidence":
                    far_player[
                        "confidence"
                    ],

                "x1":
                    far_player["x1"],

                "y1":
                    far_player["y1"],

                "x2":
                    far_player["x2"],

                "y2":
                    far_player["y2"]
            }
        )

    else:

        rows.append(
            {
                "frame":
                    frame_number,

                "time_seconds":
                    time_seconds,

                "player":
                    "far",

                "detected":
                    False,

                "confidence":
                    np.nan,

                "x1": np.nan,
                "y1": np.nan,
                "x2": np.nan,
                "y2": np.nan
            }
        )


    # ========================================================
    # SAVE NEAR PLAYER
    # ========================================================

    if near_player is not None:

        rows.append(
            {
                "frame":
                    frame_number,

                "time_seconds":
                    time_seconds,

                "player":
                    "near",

                "detected":
                    True,

                "confidence":
                    near_player[
                        "confidence"
                    ],

                "x1":
                    near_player["x1"],

                "y1":
                    near_player["y1"],

                "x2":
                    near_player["x2"],

                "y2":
                    near_player["y2"]
            }
        )

    else:

        rows.append(
            {
                "frame":
                    frame_number,

                "time_seconds":
                    time_seconds,

                "player":
                    "near",

                "detected":
                    False,

                "confidence":
                    np.nan,

                "x1": np.nan,
                "y1": np.nan,
                "x2": np.nan,
                "y2": np.nan
            }
        )


    # ========================================================
    # DRAW FAR PLAYER
    # ========================================================

    if far_player is not None:

        x1 = int(
            round(
                far_player["x1"]
            )
        )

        y1 = int(
            round(
                far_player["y1"]
            )
        )

        x2 = int(
            round(
                far_player["x2"]
            )
        )

        y2 = int(
            round(
                far_player["y2"]
            )
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 255),
            2
        )


        cv2.putText(
            frame,
            (
                f"FAR "
                f"{far_player['confidence']:.2f}"
            ),
            (
                x1,
                max(
                    25,
                    y1 - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2
        )


    # ========================================================
    # DRAW NEAR PLAYER
    # ========================================================

    if near_player is not None:

        x1 = int(
            round(
                near_player["x1"]
            )
        )

        y1 = int(
            round(
                near_player["y1"]
            )
        )

        x2 = int(
            round(
                near_player["x2"]
            )
        )

        y2 = int(
            round(
                near_player["y2"]
            )
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            3
        )


        cv2.putText(
            frame,
            (
                f"NEAR "
                f"{near_player['confidence']:.2f}"
            ),
            (
                x1,
                max(
                    25,
                    y1 - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


    writer.write(
        frame
    )


    if local_frame % 50 == 0:

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
    OUTPUT_CSV,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

far_df = df[
    df["player"] == "far"
]

near_df = df[
    df["player"] == "near"
]


far_detected = int(
    far_df["detected"].sum()
)

near_detected = int(
    near_df["detected"].sum()
)


both_frames = 0


for frame_number in range(
    start_frame,
    start_frame + frames_to_process
):

    frame_rows = df[
        df["frame"]
        ==
        frame_number
    ]

    far_ok = (
        (
            frame_rows["player"]
            == "far"
        )
        &
        frame_rows["detected"]
    ).any()

    near_ok = (
        (
            frame_rows["player"]
            == "near"
        )
        &
        frame_rows["detected"]
    ).any()


    if far_ok and near_ok:
        both_frames += 1


print()
print("RESULTS")
print("=" * 70)

print(
    f"Far player: "
    f"{far_detected}/"
    f"{len(far_df)} "
    f"("
    f"{100 * far_detected / len(far_df):.1f}%"
    f")"
)

print(
    f"Near player: "
    f"{near_detected}/"
    f"{len(near_df)} "
    f"("
    f"{100 * near_detected / len(near_df):.1f}%"
    f")"
)

print(
    f"Both players simultaneously: "
    f"{both_frames}/"
    f"{frames_to_process} "
    f"("
    f"{100 * both_frames / frames_to_process:.1f}%"
    f")"
)

print()
print(
    f"Saved CSV: {OUTPUT_CSV}"
)

print(
    f"Saved video: {OUTPUT_VIDEO}"
)