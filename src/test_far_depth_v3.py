import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from court_detector import detect_court


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "videos/test_match.mp4"

OUTPUT_VIDEO = "output/far_depth_v3.mp4"
OUTPUT_CSV = "output/far_depth_v3.csv"

START_SECONDS = 60
DURATION_SECONDS = 10

COURT_WIDTH = 8.23
COURT_LENGTH = 23.77

# Test physical depths relative to the far baseline.
#
# Negative = behind baseline
# 0        = exactly on baseline
# Positive = inside court
#
DEPTH_VALUES = np.arange(
    -2.0,
    2.01,
    0.10
)

# These larger steps will be drawn so the video
# doesn't become unreadable.
DISPLAY_DEPTHS = [
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
]

# Initial assumption:
# during this clip the player should be reasonably
# close to the baseline.
INITIAL_DEPTH_LIMIT = 1.5


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    raise RuntimeError(
        "Could not determine FPS"
    )

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

start_frame = int(
    START_SECONDS * fps
)

frames_to_process = int(
    DURATION_SECONDS * fps
)


print()
print("FAR PLAYER DEPTH V3")
print("=" * 70)

print(f"FPS: {fps:.2f}")
print(f"Frames: {frames_to_process}")
print(
    f"Depth candidates: "
    f"{len(DEPTH_VALUES)}"
)


# ============================================================
# FIRST FRAME
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
# ============================================================

print()
print("Detecting court...")

court_result = detect_court(
    first_frame
)

H_court_to_image = (
    court_result["homography"]
    .astype(np.float64)
)

H_image_to_court = np.linalg.inv(
    H_court_to_image
)

print("Court detected.")


# ============================================================
# TRANSFORMS
# ============================================================

def court_to_image(
    court_x,
    court_y
):

    point = np.array(
        [[[
            court_x,
            court_y
        ]]],
        dtype=np.float32
    )

    result = cv2.perspectiveTransform(
        point,
        H_court_to_image
    )

    return result[0, 0]


def image_to_court(
    image_x,
    image_y
):

    point = np.array(
        [[[
            image_x,
            image_y
        ]]],
        dtype=np.float32
    )

    result = cv2.perspectiveTransform(
        point,
        H_image_to_court
    )

    return result[0, 0]


# ============================================================
# FIND IMAGE POINT FOR A SPECIFIC COURT Y
#
# We know the player's image X.
#
# But perspective means court X is not identical to image X.
#
# So:
#
# 1. Project the entire horizontal court line at court_y.
# 2. Find where that projected line crosses player image X.
#
# This gives the image Y corresponding to a known physical
# depth.
# ============================================================

def projected_y_at_image_x(
    image_x,
    court_y
):

    left = court_to_image(
        -4.0,
        court_y
    )

    right = court_to_image(
        COURT_WIDTH + 4.0,
        court_y
    )


    dx = (
        right[0]
        -
        left[0]
    )

    if abs(dx) < 1e-6:
        return None


    t = (
        image_x
        -
        left[0]
    ) / dx


    image_y = (
        left[1]
        +
        t
        *
        (
            right[1]
            -
            left[1]
        )
    )


    return float(
        image_y
    )


# ============================================================
# IMAGE EVIDENCE
#
# We are looking for the player's contact with the floor.
#
# This isn't perfect by itself, which is why temporal
# continuity will also be used.
# ============================================================

def calculate_evidence(
    frame,
    player_x,
    candidate_y,
    box_width
):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    half_width = max(
        8,
        int(
            box_width * 0.75
        )
    )


    x1 = max(
        0,
        int(
            player_x
            -
            half_width
        )
    )

    x2 = min(
        width,
        int(
            player_x
            +
            half_width
        )
    )

    y1 = max(
        0,
        int(
            candidate_y - 4
        )
    )

    y2 = min(
        height,
        int(
            candidate_y + 5
        )
    )


    if (
        x2 <= x1
        or
        y2 <= y1
    ):
        return 0.0


    patch = gray[
        y1:y2,
        x1:x2
    ]


    # Vertical gradient is useful because shoes/floor
    # normally create a horizontal-ish boundary.

    gy = cv2.Sobel(
        patch,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )


    gradient_score = float(
        np.mean(
            np.abs(gy)
        )
    )


    return gradient_score


# ============================================================
# YOLO
# ============================================================

print()
print("Loading YOLO...")

model = YOLO(
    "yolo11n.pt"
)


# ============================================================
# PASS 1 — DETECTION + PHYSICAL CANDIDATES
# ============================================================

print()
print("PASS 1: collecting depth candidates...")
print()


cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    start_frame
)


frames = []

previous_player_x = None


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


    # --------------------------------------------------------
    # FAR CROP
    # --------------------------------------------------------

    crop_x1 = 0
    crop_y1 = 120
    crop_x2 = 1000
    crop_y2 = 520


    crop = frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]


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


        x1 += crop_x1
        x2 += crop_x1

        y1 += crop_y1
        y2 += crop_y1


        box_width = (
            x2 - x1
        )

        box_height = (
            y2 - y1
        )

        centre_x = (
            x1 + x2
        ) / 2


        # Same filters that already gave us 94.4%

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


        if previous_player_x is not None:

            score -= (
                abs(
                    centre_x
                    -
                    previous_player_x
                )
                /
                800.0
            )


        detections.append(
            {
                "score":
                    score,

                "confidence":
                    confidence,

                "x1":
                    float(x1),

                "y1":
                    float(y1),

                "x2":
                    float(x2),

                "y2":
                    float(y2),

                "centre_x":
                    float(centre_x),

                "box_width":
                    float(box_width)
            }
        )


    # --------------------------------------------------------
    # NO PLAYER
    # --------------------------------------------------------

    if not detections:

        frames.append(
            {
                "frame":
                    frame_number,

                "time":
                    time_seconds,

                "detection":
                    None,

                "candidates":
                    []
            }
        )

        continue


    detection = max(
        detections,
        key=lambda d:
            d["score"]
    )


    previous_player_x = (
        detection[
            "centre_x"
        ]
    )


    # ========================================================
    # PHYSICAL DEPTH CANDIDATES
    # ========================================================

    candidates = []


    for depth in DEPTH_VALUES:

        projected_y = (
            projected_y_at_image_x(
                detection[
                    "centre_x"
                ],
                depth
            )
        )


        if projected_y is None:
            continue


        evidence = calculate_evidence(
            frame,
            detection[
                "centre_x"
            ],
            projected_y,
            detection[
                "box_width"
            ]
        )


        # Difference from YOLO box bottom.
        #
        # This is useful information but NOT treated as
        # absolute truth.

        box_difference = (
            projected_y
            -
            detection["y2"]
        )


        candidates.append(
            {
                "depth":
                    float(depth),

                "pixel_x":
                    detection[
                        "centre_x"
                    ],

                "pixel_y":
                    projected_y,

                "box_difference":
                    float(
                        box_difference
                    ),

                "evidence":
                    evidence
            }
        )


    # --------------------------------------------------------
    # NORMALISE EVIDENCE
    # --------------------------------------------------------

    if candidates:

        values = np.array(
            [
                c["evidence"]
                for c
                in candidates
            ],
            dtype=np.float64
        )

        low = float(
            values.min()
        )

        high = float(
            values.max()
        )


        for candidate in candidates:

            if high > low:

                candidate[
                    "evidence_norm"
                ] = (
                    candidate[
                        "evidence"
                    ]
                    -
                    low
                ) / (
                    high
                    -
                    low
                )

            else:

                candidate[
                    "evidence_norm"
                ] = 0.5


    frames.append(
        {
            "frame":
                frame_number,

            "time":
                time_seconds,

            "detection":
                detection,

            "candidates":
                candidates
        }
    )


    if local_frame % 50 == 0:

        print(
            f"Processed "
            f"{local_frame}/"
            f"{frames_to_process}"
        )


cap.release()


# ============================================================
# INITIAL DEPTH
#
# Use evidence over the first second rather than trusting
# one frame.
# ============================================================

print()
print("INITIALISING DEPTH")
print("=" * 70)


initial_frames = int(
    fps * 1.0
)


depth_scores = {
    round(float(d), 2): 0.0
    for d in DEPTH_VALUES
}


for frame_info in frames[
    :initial_frames
]:

    for candidate in frame_info[
        "candidates"
    ]:

        depth = round(
            candidate["depth"],
            2
        )


        if (
            abs(depth)
            >
            INITIAL_DEPTH_LIMIT
        ):
            continue


        # Image evidence

        score = (
            candidate[
                "evidence_norm"
            ]
        )


        # Very weak preference for being near the box bottom.
        #
        # This is deliberately weak because we've already
        # established that the box bottom can be wrong.

        score -= (
            abs(
                candidate[
                    "box_difference"
                ]
            )
            *
            0.005
        )


        depth_scores[
            depth
        ] += score


initial_depth = max(
    depth_scores,
    key=depth_scores.get
)


print(
    f"Initial depth estimate: "
    f"{initial_depth:+.2f}m"
)


# ============================================================
# TEMPORAL TRAJECTORY
# ============================================================

print()
print("Selecting physical trajectory...")


selected = []

previous_depth = float(
    initial_depth
)

previous_velocity = 0.0


for frame_info in frames:

    candidates = frame_info[
        "candidates"
    ]


    if not candidates:

        selected.append(
            None
        )

        continue


    best = None
    best_score = -1e9


    predicted_depth = (
        previous_depth
        +
        previous_velocity
    )


    for candidate in candidates:

        depth = (
            candidate[
                "depth"
            ]
        )


        score = 0.0


        # --------------------------------------------
        # IMAGE EVIDENCE
        # --------------------------------------------

        score += (
            candidate[
                "evidence_norm"
            ]
            *
            1.25
        )


        # --------------------------------------------
        # PREVIOUS POSITION
        # --------------------------------------------

        movement = abs(
            depth
            -
            previous_depth
        )

        score -= (
            movement
            *
            7.0
        )


        # --------------------------------------------
        # VELOCITY PREDICTION
        # --------------------------------------------

        prediction_error = abs(
            depth
            -
            predicted_depth
        )

        score -= (
            prediction_error
            *
            3.0
        )


        # --------------------------------------------
        # PHYSICAL FRAME-TO-FRAME LIMIT
        #
        # At 50fps, moving >30cm in depth in one frame
        # would imply an absurd tennis-player speed.
        # --------------------------------------------

        if movement > 0.30:

            score -= 100.0


        # --------------------------------------------
        # BOX BOTTOM — WEAK CLUE ONLY
        # --------------------------------------------

        score -= (
            abs(
                candidate[
                    "box_difference"
                ]
            )
            *
            0.01
        )


        if score > best_score:

            best_score = score
            best = candidate


    if best is None:

        selected.append(
            None
        )

        continue


    new_depth = (
        best["depth"]
    )


    measured_velocity = (
        new_depth
        -
        previous_depth
    )


    previous_velocity = (
        0.85
        *
        previous_velocity
        +
        0.15
        *
        measured_velocity
    )


    previous_depth = (
        new_depth
    )


    selected.append(
        best.copy()
    )


# ============================================================
# CSV
# ============================================================

rows = []


for frame_info, result in zip(
    frames,
    selected
):

    detection = frame_info[
        "detection"
    ]


    rows.append(
        {
            "frame":
                frame_info["frame"],

            "time_seconds":
                frame_info["time"],

            "detected":
                result is not None,

            "confidence":
                (
                    detection[
                        "confidence"
                    ]
                    if detection is not None
                    else np.nan
                ),

            "player_image_x":
                (
                    detection[
                        "centre_x"
                    ]
                    if detection is not None
                    else np.nan
                ),

            "box_bottom_y":
                (
                    detection[
                        "y2"
                    ]
                    if detection is not None
                    else np.nan
                ),

            "depth":
                (
                    result[
                        "depth"
                    ]
                    if result is not None
                    else np.nan
                ),

            "selected_image_y":
                (
                    result[
                        "pixel_y"
                    ]
                    if result is not None
                    else np.nan
                ),

            "box_difference":
                (
                    result[
                        "box_difference"
                    ]
                    if result is not None
                    else np.nan
                ),

            "evidence":
                (
                    result[
                        "evidence_norm"
                    ]
                    if result is not None
                    else np.nan
                )
        }
    )


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

valid = df[
    df["detected"]
]


print()
print("RESULTS")
print("=" * 70)

print(
    f"Detected: "
    f"{len(valid)}/"
    f"{len(df)} "
    f"("
    f"{100 * len(valid) / len(df):.1f}%"
    f")"
)


if len(valid) > 0:

    print()
    print("SELECTED DEPTH")

    print(
        f"Mean: "
        f"{valid['depth'].mean():+.2f}m"
    )

    print(
        f"Median: "
        f"{valid['depth'].median():+.2f}m"
    )

    print(
        f"Min: "
        f"{valid['depth'].min():+.2f}m"
    )

    print(
        f"Max: "
        f"{valid['depth'].max():+.2f}m"
    )


# ============================================================
# DEBUG VIDEO
# ============================================================

print()
print("Creating debug video...")


cap = cv2.VideoCapture(
    VIDEO_PATH
)

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    start_frame
)


writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(
        *"mp4v"
    ),
    fps,
    (width, height)
)


for i, (
    frame_info,
    result
) in enumerate(
    zip(
        frames,
        selected
    )
):

    success, frame = cap.read()

    if not success:
        break


    detection = frame_info[
        "detection"
    ]


    if detection is not None:

        # --------------------------------------------
        # YOLO BOX
        # --------------------------------------------

        x1 = int(
            round(
                detection["x1"]
            )
        )

        y1 = int(
            round(
                detection["y1"]
            )
        )

        x2 = int(
            round(
                detection["x2"]
            )
        )

        y2 = int(
            round(
                detection["y2"]
            )
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2
        )


        player_x = (
            detection[
                "centre_x"
            ]
        )


        # ====================================================
        # DRAW REFERENCE DEPTHS
        # ====================================================

        for depth in DISPLAY_DEPTHS:

            projected_y = (
                projected_y_at_image_x(
                    player_x,
                    depth
                )
            )

            if projected_y is None:
                continue


            px = int(
                round(
                    player_x
                )
            )

            py = int(
                round(
                    projected_y
                )
            )


            # Baseline gets a larger marker

            if abs(depth) < 0.01:

                radius = 7
                thickness = 2

            else:

                radius = 4
                thickness = 1


            cv2.circle(
                frame,
                (px, py),
                radius,
                (255, 0, 255),
                thickness
            )


            cv2.putText(
                frame,
                f"{depth:+.1f}m",
                (
                    px + 10,
                    py + 4
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 0, 255),
                1
            )


        # ====================================================
        # BOX BOTTOM
        # ====================================================

        box_point = (
            int(
                round(
                    player_x
                )
            ),
            int(
                round(
                    detection["y2"]
                )
            )
        )


        cv2.circle(
            frame,
            box_point,
            5,
            (0, 0, 255),
            -1
        )


    # ========================================================
    # SELECTED POSITION
    # ========================================================

    if result is not None:

        selected_point = (
            int(
                round(
                    result[
                        "pixel_x"
                    ]
                )
            ),
            int(
                round(
                    result[
                        "pixel_y"
                    ]
                )
            )
        )


        cv2.circle(
            frame,
            selected_point,
            8,
            (0, 255, 0),
            -1
        )


        cv2.putText(
            frame,
            (
                f"SELECTED "
                f"{result['depth']:+.2f}m"
            ),
            (
                selected_point[0] + 15,
                selected_point[1] - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )


    # ========================================================
    # TIME
    # ========================================================

    cv2.putText(
        frame,
        (
            f"{frame_info['time']:.2f}s"
        ),
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    writer.write(
        frame
    )


cap.release()
writer.release()


print()
print(
    f"Saved CSV: {OUTPUT_CSV}"
)

print(
    f"Saved video: {OUTPUT_VIDEO}"
)

print()
print("DONE")