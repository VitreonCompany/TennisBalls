import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from court_detector import detect_court


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "videos/test_match.mp4"

OUTPUT_VIDEO = "output/player_tracking_v2.mp4"
OUTPUT_CSV = "output/player_tracks_v2.csv"

START_SECONDS = 60
DURATION_SECONDS = 10

COURT_WIDTH = 8.23
COURT_LENGTH = 23.77

# Far-player ground search
GROUND_OFFSETS = np.arange(-3, 31, 2)

# Use first 0.75 sec to initialise far-player depth
INITIALISATION_SECONDS = 0.75

# Pose ankle confidence
MIN_ANKLE_CONF = 0.35


# ============================================================
# VIDEO
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

initialisation_frames = max(
    5,
    int(INITIALISATION_SECONDS * fps)
)


print()
print("COMBINED PLAYER TRACKER V2")
print("=" * 70)

print(f"FPS: {fps:.2f}")
print(f"Video: {width} x {height}")
print(f"Frames: {frames_to_process}")
print(
    f"Far initialisation: "
    f"{initialisation_frames} frames"
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
# COURT DETECTION
# ============================================================

print()
print("Detecting court...")

court_result = detect_court(
    first_frame
)

H_image_to_court = np.linalg.inv(
    court_result["homography"]
).astype(np.float64)

print("Court detected.")


# ============================================================
# COORDINATE CONVERSION
# ============================================================

def image_to_court(x, y):

    point = np.array(
        [[[x, y]]],
        dtype=np.float32
    )

    result = cv2.perspectiveTransform(
        point,
        H_image_to_court
    )

    return result[0, 0]


# ============================================================
# MODELS
# ============================================================

print()
print("Loading models...")

person_model = YOLO(
    "yolo11n.pt"
)

pose_model = YOLO(
    "yolo11n-pose.pt"
)

print("Models loaded.")


# ============================================================
# IMAGE EVIDENCE FOR FAR PLAYER
# ============================================================

def image_evidence(
    gray,
    centre_x,
    candidate_y,
    box_width
):

    search_half_width = max(
        6,
        int(box_width * 0.65)
    )

    x1 = max(
        0,
        int(centre_x - search_half_width)
    )

    x2 = min(
        gray.shape[1],
        int(centre_x + search_half_width)
    )

    y1 = max(
        1,
        int(candidate_y - 3)
    )

    y2 = min(
        gray.shape[0] - 1,
        int(candidate_y + 4)
    )

    if x2 <= x1 or y2 <= y1:
        return 0.0

    patch = gray[
        y1:y2,
        x1:x2
    ]

    gx = cv2.Sobel(
        patch,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        patch,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        gx,
        gy
    )

    return float(
        np.mean(magnitude)
    )


# ============================================================
# STORAGE
#
# First pass:
# collect all detections.
#
# We do this because the far player needs multi-frame
# initialisation.
# ============================================================

frame_data = []

previous_far_x = None
previous_near_x = None


# ============================================================
# RESET VIDEO
# ============================================================

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    start_frame
)


print()
print("PASS 1: detecting both players...")
print()


# ============================================================
# PASS 1
# ============================================================

for local_frame in range(
    frames_to_process
):

    success, frame = cap.read()

    if not success:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

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
    # FAR PLAYER DETECTION
    # ========================================================

    far_crop_x1 = 0
    far_crop_y1 = 120
    far_crop_x2 = 1000
    far_crop_y2 = 520

    far_crop = frame[
        far_crop_y1:far_crop_y2,
        far_crop_x1:far_crop_x2
    ]


    far_results = person_model.predict(
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

        x1 += far_crop_x1
        x2 += far_crop_x1

        y1 += far_crop_y1
        y2 += far_crop_y1

        box_width = x2 - x1
        box_height = y2 - y1

        centre_x = (
            x1 + x2
        ) / 2


        if centre_x > width * 0.55:
            continue

        if box_height < 20:
            continue

        if box_height > 180:
            continue

        if y2 < 300 or y2 > 520:
            continue


        score = confidence


        if previous_far_x is not None:

            score -= (
                abs(
                    centre_x
                    -
                    previous_far_x
                )
                /
                800.0
            )


        far_candidates.append(
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
    # FAR GROUND CANDIDATES
    # ========================================================

    far_ground_candidates = []


    if far_player is not None:

        centre_x = (
            far_player[
                "centre_x"
            ]
        )

        box_bottom = (
            far_player[
                "y2"
            ]
        )

        box_width = (
            far_player[
                "box_width"
            ]
        )


        for offset in GROUND_OFFSETS:

            candidate_y = (
                box_bottom
                +
                offset
            )


            if (
                candidate_y < 0
                or
                candidate_y >= height
            ):
                continue


            court = image_to_court(
                centre_x,
                candidate_y
            )

            court_x = float(
                court[0]
            )

            court_y = float(
                court[1]
            )


            if court_x < -8:
                continue

            if court_x > COURT_WIDTH + 8:
                continue

            if court_y < -10:
                continue

            if court_y > 12:
                continue


            evidence = image_evidence(
                gray,
                centre_x,
                candidate_y,
                box_width
            )


            far_ground_candidates.append(
                {
                    "offset":
                        float(offset),

                    "pixel_x":
                        float(centre_x),

                    "pixel_y":
                        float(candidate_y),

                    "court_x":
                        court_x,

                    "court_y":
                        court_y,

                    "evidence":
                        evidence
                }
            )


        # Normalise image evidence

        if far_ground_candidates:

            evidence_values = np.array(
                [
                    c["evidence"]
                    for c
                    in far_ground_candidates
                ],
                dtype=np.float64
            )

            e_min = (
                evidence_values.min()
            )

            e_max = (
                evidence_values.max()
            )


            for candidate in (
                far_ground_candidates
            ):

                if e_max > e_min:

                    candidate[
                        "evidence_norm"
                    ] = (
                        candidate[
                            "evidence"
                        ]
                        -
                        e_min
                    ) / (
                        e_max
                        -
                        e_min
                    )

                else:

                    candidate[
                        "evidence_norm"
                    ] = 0.5


    # ========================================================
    # NEAR PLAYER POSE
    # ========================================================

    near_crop_x1 = int(
        width * 0.45
    )

    near_crop_y1 = int(
        height * 0.20
    )

    near_crop_x2 = width
    near_crop_y2 = height


    near_crop = frame[
        near_crop_y1:near_crop_y2,
        near_crop_x1:near_crop_x2
    ]


    near_results = pose_model.predict(
        near_crop,
        conf=0.20,
        imgsz=1280,
        verbose=False
    )


    near_candidates = []


    for result in near_results:

        if result.boxes is None:
            continue

        if result.keypoints is None:
            continue


        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        keypoints_xy = (
            result.keypoints.xy
            .cpu()
            .numpy()
        )


        if result.keypoints.conf is not None:

            keypoints_conf = (
                result.keypoints.conf
                .cpu()
                .numpy()
            )

        else:

            keypoints_conf = None


        for detection_index in range(
            len(boxes)
        ):

            x1, y1, x2, y2 = (
                boxes[
                    detection_index
                ]
            )


            x1 += near_crop_x1
            x2 += near_crop_x1

            y1 += near_crop_y1
            y2 += near_crop_y1


            box_height = y2 - y1

            centre_x = (
                x1 + x2
            ) / 2


            if box_height < 120:
                continue

            if centre_x < width * 0.45:
                continue


            confidence = float(
                confidences[
                    detection_index
                ]
            )


            score = (
                confidence
                +
                box_height
                /
                1000.0
            )


            if previous_near_x is not None:

                score -= (
                    abs(
                        centre_x
                        -
                        previous_near_x
                    )
                    /
                    1200.0
                )


            kp = keypoints_xy[
                detection_index
            ].copy()

            kp[:, 0] += (
                near_crop_x1
            )

            kp[:, 1] += (
                near_crop_y1
            )


            if keypoints_conf is not None:

                kp_conf = (
                    keypoints_conf[
                        detection_index
                    ].copy()
                )

            else:

                kp_conf = (
                    np.ones(
                        len(kp),
                        dtype=np.float32
                    )
                )


            near_candidates.append(
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

                    "keypoints":
                        kp,

                    "keypoint_conf":
                        kp_conf
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
    # NEAR FOOT / GROUND INFORMATION
    # ========================================================

    near_ground = None

    near_left_foot = None
    near_right_foot = None


    if near_player is not None:

        kp = near_player[
            "keypoints"
        ]

        kp_conf = near_player[
            "keypoint_conf"
        ]


        # COCO:
        # 15 = left ankle
        # 16 = right ankle

        left = kp[15]
        right = kp[16]

        left_conf = float(
            kp_conf[15]
        )

        right_conf = float(
            kp_conf[16]
        )


        left_valid = (
            left_conf
            >=
            MIN_ANKLE_CONF
        )

        right_valid = (
            right_conf
            >=
            MIN_ANKLE_CONF
        )


        # --------------------------------------------
        # LEFT FOOT COURT POSITION
        # --------------------------------------------

        if left_valid:

            left_court = (
                image_to_court(
                    left[0],
                    left[1]
                )
            )

            near_left_foot = {
                "pixel_x":
                    float(left[0]),

                "pixel_y":
                    float(left[1]),

                "court_x":
                    float(
                        left_court[0]
                    ),

                "court_y":
                    float(
                        left_court[1]
                    ),

                "confidence":
                    left_conf
            }


        # --------------------------------------------
        # RIGHT FOOT COURT POSITION
        # --------------------------------------------

        if right_valid:

            right_court = (
                image_to_court(
                    right[0],
                    right[1]
                )
            )

            near_right_foot = {
                "pixel_x":
                    float(right[0]),

                "pixel_y":
                    float(right[1]),

                "court_x":
                    float(
                        right_court[0]
                    ),

                "court_y":
                    float(
                        right_court[1]
                    ),

                "confidence":
                    right_conf
            }


        # --------------------------------------------
        # PLAYER POSITION
        #
        # If both feet are available, calculate the
        # midpoint IN COURT SPACE.
        #
        # This is better than averaging pixels first.
        # --------------------------------------------

        if (
            near_left_foot is not None
            and
            near_right_foot is not None
        ):

            near_ground = {
                "court_x":
                    (
                        near_left_foot[
                            "court_x"
                        ]
                        +
                        near_right_foot[
                            "court_x"
                        ]
                    )
                    /
                    2,

                "court_y":
                    (
                        near_left_foot[
                            "court_y"
                        ]
                        +
                        near_right_foot[
                            "court_y"
                        ]
                    )
                    /
                    2,

                "pixel_x":
                    (
                        near_left_foot[
                            "pixel_x"
                        ]
                        +
                        near_right_foot[
                            "pixel_x"
                        ]
                    )
                    /
                    2,

                "pixel_y":
                    max(
                        near_left_foot[
                            "pixel_y"
                        ],
                        near_right_foot[
                            "pixel_y"
                        ]
                    ),

                "method":
                    "two_ankles"
            }


        elif near_left_foot is not None:

            near_ground = {
                "court_x":
                    near_left_foot[
                        "court_x"
                    ],

                "court_y":
                    near_left_foot[
                        "court_y"
                    ],

                "pixel_x":
                    near_left_foot[
                        "pixel_x"
                    ],

                "pixel_y":
                    near_left_foot[
                        "pixel_y"
                    ],

                "method":
                    "left_ankle"
            }


        elif near_right_foot is not None:

            near_ground = {
                "court_x":
                    near_right_foot[
                        "court_x"
                    ],

                "court_y":
                    near_right_foot[
                        "court_y"
                    ],

                "pixel_x":
                    near_right_foot[
                        "pixel_x"
                    ],

                "pixel_y":
                    near_right_foot[
                        "pixel_y"
                    ],

                "method":
                    "right_ankle"
            }


        else:

            # Last-resort box fallback

            box_x = (
                near_player["x1"]
                +
                near_player["x2"]
            ) / 2

            box_y = (
                near_player["y2"]
            )

            court = image_to_court(
                box_x,
                box_y
            )

            near_ground = {
                "court_x":
                    float(
                        court[0]
                    ),

                "court_y":
                    float(
                        court[1]
                    ),

                "pixel_x":
                    float(box_x),

                "pixel_y":
                    float(box_y),

                "method":
                    "box_fallback"
            }


    # ========================================================
    # STORE FRAME
    # ========================================================

    frame_data.append(
        {
            "frame":
                frame_number,

            "time_seconds":
                time_seconds,

            "far_player":
                far_player,

            "far_candidates":
                far_ground_candidates,

            "near_player":
                near_player,

            "near_ground":
                near_ground,

            "near_left_foot":
                near_left_foot,

            "near_right_foot":
                near_right_foot
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
# FAR PLAYER INITIALISATION
# ============================================================

print()
print("INITIALISING FAR PLAYER")
print("=" * 70)


initial_y_values = []


for data in frame_data[
    :initialisation_frames
]:

    candidates = data[
        "far_candidates"
    ]

    if not candidates:
        continue


    ranked = sorted(
        candidates,
        key=lambda c:
            c["evidence_norm"],
        reverse=True
    )


    for candidate in ranked[:5]:

        if (
            -3.0
            <= candidate["court_y"]
            <= 4.0
        ):

            weight = (
                0.25
                +
                candidate[
                    "evidence_norm"
                ]
            )

            initial_y_values.append(
                (
                    candidate[
                        "court_y"
                    ],
                    weight
                )
            )


if not initial_y_values:

    raise RuntimeError(
        "Could not initialise far player"
    )


initial_y_values.sort(
    key=lambda item:
        item[0]
)


total_weight = sum(
    weight
    for _, weight
    in initial_y_values
)

running_weight = 0.0
initial_far_y = None


for value, weight in initial_y_values:

    running_weight += weight

    if (
        running_weight
        >=
        total_weight / 2
    ):

        initial_far_y = value
        break


print(
    f"Initial far court-Y: "
    f"{initial_far_y:.2f}m"
)


# ============================================================
# FAR PLAYER TEMPORAL SELECTION
# ============================================================

print()
print("Selecting far-player trajectory...")


far_selected = []

previous_court = None
previous_velocity = None
previous_ground_y = None


for i, data in enumerate(
    frame_data
):

    candidates = data[
        "far_candidates"
    ]


    if not candidates:

        far_selected.append(
            None
        )

        continue


    for candidate in candidates:

        score = 0.0


        # Image evidence

        score += (
            candidate[
                "evidence_norm"
            ]
            * 1.5
        )


        # Initialisation consensus

        if i < initialisation_frames:

            score -= (
                abs(
                    candidate[
                        "court_y"
                    ]
                    -
                    initial_far_y
                )
                *
                2.5
            )


        # Pixel continuity

        if previous_ground_y is not None:

            score -= (
                abs(
                    candidate[
                        "pixel_y"
                    ]
                    -
                    previous_ground_y
                )
                *
                0.05
            )


        # Court continuity

        if previous_court is not None:

            candidate_position = np.array(
                [
                    candidate[
                        "court_x"
                    ],
                    candidate[
                        "court_y"
                    ]
                ],
                dtype=np.float64
            )

            distance = np.linalg.norm(
                candidate_position
                -
                previous_court
            )


            score -= (
                distance
                *
                4.0
            )


            if distance > 0.75:

                score -= 20.0


        # Velocity prediction

        if (
            previous_court is not None
            and
            previous_velocity is not None
        ):

            predicted = (
                previous_court
                +
                previous_velocity
            )

            candidate_position = np.array(
                [
                    candidate[
                        "court_x"
                    ],
                    candidate[
                        "court_y"
                    ]
                ],
                dtype=np.float64
            )

            prediction_error = (
                np.linalg.norm(
                    candidate_position
                    -
                    predicted
                )
            )

            score -= (
                prediction_error
                *
                2.0
            )


        # Weak offset prior

        score -= (
            abs(
                candidate[
                    "offset"
                ]
                -
                12
            )
            *
            0.005
        )


        candidate[
            "score"
        ] = score


    best = max(
        candidates,
        key=lambda c:
            c["score"]
    )


    current_court = np.array(
        [
            best["court_x"],
            best["court_y"]
        ],
        dtype=np.float64
    )


    if previous_court is not None:

        measured_velocity = (
            current_court
            -
            previous_court
        )


        if previous_velocity is None:

            previous_velocity = (
                measured_velocity
            )

        else:

            previous_velocity = (
                0.85
                *
                previous_velocity
                +
                0.15
                *
                measured_velocity
            )


    previous_court = (
        current_court
    )

    previous_ground_y = (
        best["pixel_y"]
    )


    far_selected.append(
        best.copy()
    )


# ============================================================
# BUILD FINAL CSV
# ============================================================

rows = []


for data, far in zip(
    frame_data,
    far_selected
):

    near = data[
        "near_ground"
    ]

    left = data[
        "near_left_foot"
    ]

    right = data[
        "near_right_foot"
    ]

    near_player = data[
        "near_player"
    ]

    far_player = data[
        "far_player"
    ]


    row = {
        "frame":
            data["frame"],

        "time_seconds":
            data["time_seconds"],


        # --------------------------------------------
        # NEAR
        # --------------------------------------------

        "near_detected":
            near is not None,

        "near_confidence":
            (
                near_player[
                    "confidence"
                ]
                if near_player
                is not None
                else np.nan
            ),

        "near_x":
            (
                near["court_x"]
                if near is not None
                else np.nan
            ),

        "near_y":
            (
                near["court_y"]
                if near is not None
                else np.nan
            ),

        "near_ground_method":
            (
                near["method"]
                if near is not None
                else ""
            ),


        # --------------------------------------------
        # NEAR LEFT FOOT
        # --------------------------------------------

        "near_left_foot_x":
            (
                left["court_x"]
                if left is not None
                else np.nan
            ),

        "near_left_foot_y":
            (
                left["court_y"]
                if left is not None
                else np.nan
            ),


        # --------------------------------------------
        # NEAR RIGHT FOOT
        # --------------------------------------------

        "near_right_foot_x":
            (
                right["court_x"]
                if right is not None
                else np.nan
            ),

        "near_right_foot_y":
            (
                right["court_y"]
                if right is not None
                else np.nan
            ),


        # --------------------------------------------
        # FAR
        # --------------------------------------------

        "far_detected":
            far is not None,

        "far_confidence":
            (
                far_player[
                    "confidence"
                ]
                if far_player
                is not None
                else np.nan
            ),

        "far_x":
            (
                far["court_x"]
                if far is not None
                else np.nan
            ),

        "far_y":
            (
                far["court_y"]
                if far is not None
                else np.nan
            ),

        "far_ground_offset":
            (
                far["offset"]
                if far is not None
                else np.nan
            )
    }


    rows.append(
        row
    )


df = pd.DataFrame(
    rows
)


df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

near_count = int(
    df[
        "near_detected"
    ].sum()
)

far_count = int(
    df[
        "far_detected"
    ].sum()
)

both_count = int(
    (
        df["near_detected"]
        &
        df["far_detected"]
    ).sum()
)


print()
print("FINAL TRACK RESULTS")
print("=" * 70)

print(
    f"Near: "
    f"{near_count}/"
    f"{len(df)} "
    f"("
    f"{100 * near_count / len(df):.1f}%"
    f")"
)

print(
    f"Far: "
    f"{far_count}/"
    f"{len(df)} "
    f"("
    f"{100 * far_count / len(df):.1f}%"
    f")"
)

print(
    f"Both: "
    f"{both_count}/"
    f"{len(df)} "
    f"("
    f"{100 * both_count / len(df):.1f}%"
    f")"
)


if far_count > 0:

    far_valid = df[
        df["far_detected"]
    ]

    print()
    print("FAR COURT Y")

    print(
        f"Mean: "
        f"{far_valid['far_y'].mean():.2f}m"
    )

    print(
        f"Median: "
        f"{far_valid['far_y'].median():.2f}m"
    )

    print(
        f"Min: "
        f"{far_valid['far_y'].min():.2f}m"
    )

    print(
        f"Max: "
        f"{far_valid['far_y'].max():.2f}m"
    )


if near_count > 0:

    near_valid = df[
        df["near_detected"]
    ]

    print()
    print("NEAR COURT Y")

    print(
        f"Mean: "
        f"{near_valid['near_y'].mean():.2f}m"
    )

    print(
        f"Median: "
        f"{near_valid['near_y'].median():.2f}m"
    )

    print(
        f"Min: "
        f"{near_valid['near_y'].min():.2f}m"
    )

    print(
        f"Max: "
        f"{near_valid['near_y'].max():.2f}m"
    )


# ============================================================
# TOP-DOWN COURT
# ============================================================

def draw_top_down(
    near_position,
    far_position
):

    court_w = 220
    court_h = 500

    margin = 25


    image = np.zeros(
        (
            court_h + margin * 2,
            court_w + margin * 2,
            3
        ),
        dtype=np.uint8
    )


    def court_to_minimap(
        court_x,
        court_y
    ):

        px = (
            margin
            +
            court_x
            /
            COURT_WIDTH
            *
            court_w
        )

        py = (
            margin
            +
            court_y
            /
            COURT_LENGTH
            *
            court_h
        )

        return (
            int(round(px)),
            int(round(py))
        )


    # Outer singles court

    cv2.rectangle(
        image,
        (
            margin,
            margin
        ),
        (
            margin + court_w,
            margin + court_h
        ),
        (255, 255, 255),
        2
    )


    # Net

    net_y = int(
        margin
        +
        court_h / 2
    )

    cv2.line(
        image,
        (
            margin,
            net_y
        ),
        (
            margin + court_w,
            net_y
        ),
        (255, 255, 255),
        2
    )


    # Service lines

    far_service_y = (
        5.485
    )

    near_service_y = (
        COURT_LENGTH
        -
        5.485
    )


    for service_y in [
        far_service_y,
        near_service_y
    ]:

        _, py = court_to_minimap(
            0,
            service_y
        )

        cv2.line(
            image,
            (
                margin,
                py
            ),
            (
                margin + court_w,
                py
            ),
            (180, 180, 180),
            1
        )


    # Centre service line

    centre_x = int(
        margin
        +
        court_w / 2
    )

    y1 = court_to_minimap(
        0,
        far_service_y
    )[1]

    y2 = court_to_minimap(
        0,
        near_service_y
    )[1]


    cv2.line(
        image,
        (
            centre_x,
            y1
        ),
        (
            centre_x,
            y2
        ),
        (180, 180, 180),
        1
    )


    # Far player

    if far_position is not None:

        point = court_to_minimap(
            far_position[0],
            far_position[1]
        )

        cv2.circle(
            image,
            point,
            8,
            (255, 0, 255),
            -1
        )


    # Near player

    if near_position is not None:

        point = court_to_minimap(
            near_position[0],
            near_position[1]
        )

        cv2.circle(
            image,
            point,
            8,
            (0, 255, 255),
            -1
        )


    return image


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


for i, row in df.iterrows():

    success, frame = cap.read()

    if not success:
        break


    data = frame_data[i]


    # ========================================================
    # DRAW NEAR
    # ========================================================

    near_player = data[
        "near_player"
    ]

    near_ground = data[
        "near_ground"
    ]


    if (
        near_player is not None
        and
        near_ground is not None
    ):

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


        point = (
            int(
                round(
                    near_ground[
                        "pixel_x"
                    ]
                )
            ),
            int(
                round(
                    near_ground[
                        "pixel_y"
                    ]
                )
            )
        )


        cv2.circle(
            frame,
            point,
            8,
            (0, 255, 0),
            -1
        )


        text = (
            f"NEAR "
            f"({row['near_x']:.2f}, "
            f"{row['near_y']:.2f})m"
        )


        cv2.putText(
            frame,
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


    # ========================================================
    # DRAW FAR
    # ========================================================

    far_player = data[
        "far_player"
    ]

    far = far_selected[i]


    if (
        far_player is not None
        and
        far is not None
    ):

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


        point = (
            int(
                round(
                    far["pixel_x"]
                )
            ),
            int(
                round(
                    far["pixel_y"]
                )
            )
        )


        cv2.circle(
            frame,
            point,
            7,
            (0, 255, 0),
            -1
        )


        text = (
            f"FAR "
            f"({row['far_x']:.2f}, "
            f"{row['far_y']:.2f})m"
        )


        cv2.putText(
            frame,
            text,
            (
                x1,
                max(
                    30,
                    y1 - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2
        )


    # ========================================================
    # MINIMAP
    # ========================================================

    if row["near_detected"]:

        near_position = (
            row["near_x"],
            row["near_y"]
        )

    else:

        near_position = None


    if row["far_detected"]:

        far_position = (
            row["far_x"],
            row["far_y"]
        )

    else:

        far_position = None


    minimap = draw_top_down(
        near_position,
        far_position
    )


    map_h, map_w = (
        minimap.shape[:2]
    )


    # Resize if video is smaller than expected

    max_map_height = int(
        height * 0.45
    )

    if map_h > max_map_height:

        scale = (
            max_map_height
            /
            map_h
        )

        minimap = cv2.resize(
            minimap,
            None,
            fx=scale,
            fy=scale
        )

        map_h, map_w = (
            minimap.shape[:2]
        )


    # Top-left corner

    map_x = 20
    map_y = 20


    frame[
        map_y:
        map_y + map_h,

        map_x:
        map_x + map_w
    ] = minimap


    cv2.putText(
        frame,
        f"{row['time_seconds']:.2f}s",
        (
            map_x,
            map_y + map_h + 30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
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