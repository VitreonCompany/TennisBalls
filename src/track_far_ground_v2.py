import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from court_detector import detect_court


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "videos/test_match.mp4"

VIDEO_OUTPUT = "output/far_ground_tracking_v2.mp4"
CSV_OUTPUT = "output/far_ground_tracking_v2.csv"

START_SECONDS = 60
DURATION_SECONDS = 10

COURT_WIDTH = 8.23

# Candidate ground positions relative to YOLO box bottom
GROUND_OFFSETS = np.arange(-3, 31, 2)

# NEW:
# Don't trust the first frame.
# Use this many seconds to establish the initial trajectory.
INITIALISATION_SECONDS = 0.75


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    raise RuntimeError("Could not determine FPS")

start_frame = int(START_SECONDS * fps)
frames_to_process = int(DURATION_SECONDS * fps)

initialisation_frames = max(
    5,
    int(INITIALISATION_SECONDS * fps)
)

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    start_frame
)

success, first_frame = cap.read()

if not success:
    raise RuntimeError("Could not read starting frame")

height, width = first_frame.shape[:2]


print()
print("TEMPORAL FAR-GROUND TRACKER V2")
print("=" * 70)

print(f"FPS: {fps:.2f}")
print(f"Frames: {frames_to_process}")
print(
    f"Initialisation window: "
    f"{initialisation_frames} frames "
    f"({INITIALISATION_SECONDS:.2f}s)"
)


# ============================================================
# COURT DETECTION
# ============================================================

print()
print("Detecting court...")

court_result = detect_court(first_frame)

H_image_to_court = np.linalg.inv(
    court_result["homography"]
).astype(np.float64)

far_baseline = court_result[
    "far_baseline"
].astype(np.float64)

print("Court detected.")


# ============================================================
# COORDINATE FUNCTIONS
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


def baseline_y_at_x(x):

    x1, y1, x2, y2 = far_baseline

    if abs(x2 - x1) < 1e-6:
        return (y1 + y2) / 2

    return (
        y1
        +
        (x - x1)
        * (y2 - y1)
        / (x2 - x1)
    )


# ============================================================
# IMAGE EVIDENCE
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
# LOAD YOLO
# ============================================================

print()
print("Loading YOLO...")

model = YOLO("yolo11n.pt")


# ============================================================
# FIRST PASS
#
# Get YOLO detections and candidate ground points for every
# frame FIRST.
#
# We deliberately do NOT track yet.
# ============================================================

print()
print("PASS 1: collecting candidate ground points...")
print()


cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    start_frame
)

frame_data = []

previous_detection_x = None


for local_frame in range(frames_to_process):

    success, frame = cap.read()

    if not success:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # FAR PLAYER CROP
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

        if previous_detection_x is not None:

            dx = abs(
                centre_x
                - previous_detection_x
            )

            score -= dx / 800.0


        detections.append(
            {
                "score": score,
                "confidence": confidence,

                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),

                "centre_x": float(centre_x),

                "box_width": float(box_width),
                "box_height": float(box_height)
            }
        )


    # --------------------------------------------------------
    # NO PLAYER
    # --------------------------------------------------------

    if not detections:

        frame_data.append(
            {
                "frame":
                    start_frame + local_frame,

                "time_seconds":
                    START_SECONDS
                    + local_frame / fps,

                "detected": False,

                "player": None,
                "candidates": []
            }
        )

        continue


    # --------------------------------------------------------
    # SELECT PLAYER
    # --------------------------------------------------------

    player = max(
        detections,
        key=lambda d:
            d["score"]
    )

    previous_detection_x = player[
        "centre_x"
    ]


    centre_x = player[
        "centre_x"
    ]

    box_width = player[
        "box_width"
    ]

    box_bottom = player[
        "y2"
    ]


    # --------------------------------------------------------
    # CREATE GROUND CANDIDATES
    # --------------------------------------------------------

    candidates = []


    for offset in GROUND_OFFSETS:

        candidate_y = (
            box_bottom + offset
        )

        if candidate_y < 0:
            continue

        if candidate_y >= height:
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


        # Broad physical region only.

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


        candidates.append(
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


    # --------------------------------------------------------
    # NORMALISE EVIDENCE WITHIN FRAME
    # --------------------------------------------------------

    if candidates:

        values = np.array(
            [
                c["evidence"]
                for c in candidates
            ],
            dtype=np.float64
        )

        e_min = values.min()
        e_max = values.max()


        for candidate in candidates:

            if e_max > e_min:

                candidate[
                    "evidence_norm"
                ] = (
                    candidate["evidence"]
                    - e_min
                ) / (
                    e_max - e_min
                )

            else:

                candidate[
                    "evidence_norm"
                ] = 0.5


    frame_data.append(
        {
            "frame":
                start_frame + local_frame,

            "time_seconds":
                START_SECONDS
                + local_frame / fps,

            "detected": True,

            "player": player,

            "candidates": candidates
        }
    )


    if local_frame % 50 == 0:

        print(
            f"Collected "
            f"{local_frame}/"
            f"{frames_to_process}"
        )


cap.release()


# ============================================================
# INITIALISATION
#
# This is the key change.
#
# Look at ALL candidates from the first 0.75 seconds.
#
# Rather than allowing the first strong edge to initialise the
# tracker, find the depth region supported across many frames.
# ============================================================

print()
print("INITIALISING TRACK")
print("=" * 70)


initial_y_values = []


for data in frame_data[
    :initialisation_frames
]:

    if not data["detected"]:
        continue

    candidates = data[
        "candidates"
    ]

    if not candidates:
        continue


    # Keep multiple credible candidates from each frame.
    #
    # This means one incorrect high-gradient line cannot
    # dominate the whole initialisation.

    ranked = sorted(
        candidates,
        key=lambda c:
            c["evidence_norm"],
        reverse=True
    )


    for candidate in ranked[:5]:

        # We know absolutely nothing exact about the player's
        # position yet, so this is intentionally broad.
        #
        # We're only removing obviously absurd far-player
        # positions during initialisation.

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
        "Could not establish initial ground region"
    )


# ============================================================
# WEIGHTED MEDIAN
# ============================================================

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

initial_court_y = None


for value, weight in initial_y_values:

    running_weight += weight

    if (
        running_weight
        >= total_weight / 2
    ):

        initial_court_y = value
        break


print(
    f"Initial court-Y estimate: "
    f"{initial_court_y:.2f}m"
)

print(
    f"Evidence samples used: "
    f"{len(initial_y_values)}"
)


# ============================================================
# SECOND PASS:
# TEMPORAL TRACKING
# ============================================================

print()
print("PASS 2: selecting smooth trajectory...")
print()


selected = []

previous_court = None
previous_velocity = None
previous_ground_y = None


for i, data in enumerate(frame_data):

    if (
        not data["detected"]
        or
        not data["candidates"]
    ):

        selected.append(None)
        continue


    candidates = data[
        "candidates"
    ]


    # ========================================================
    # SCORE
    # ========================================================

    for candidate in candidates:

        score = 0.0


        # ----------------------------------------------------
        # IMAGE EVIDENCE
        # ----------------------------------------------------

        score += (
            candidate[
                "evidence_norm"
            ]
            * 1.5
        )


        # ----------------------------------------------------
        # INITIALISATION REGION
        #
        # During the initialisation window, strongly favour
        # candidates near the consensus estimate.
        # ----------------------------------------------------

        if i < initialisation_frames:

            initial_error = abs(
                candidate["court_y"]
                -
                initial_court_y
            )

            score -= (
                initial_error
                * 2.5
            )


        # ----------------------------------------------------
        # TEMPORAL PIXEL CONTINUITY
        # ----------------------------------------------------

        if previous_ground_y is not None:

            pixel_jump = abs(
                candidate["pixel_y"]
                -
                previous_ground_y
            )

            score -= (
                pixel_jump
                * 0.05
            )


        # ----------------------------------------------------
        # COURT CONTINUITY
        # ----------------------------------------------------

        if previous_court is not None:

            dx = (
                candidate["court_x"]
                -
                previous_court[0]
            )

            dy = (
                candidate["court_y"]
                -
                previous_court[1]
            )

            distance = np.sqrt(
                dx * dx
                +
                dy * dy
            )


            # Stronger than V1.

            score -= (
                distance
                * 4.0
            )


            # Hard protection against ridiculous
            # frame-to-frame movement.

            if distance > 0.75:

                score -= 20.0


        # ----------------------------------------------------
        # VELOCITY PREDICTION
        # ----------------------------------------------------

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


            prediction_error = np.linalg.norm(
                candidate_position
                -
                predicted
            )


            score -= (
                prediction_error
                * 2.0
            )


        # ----------------------------------------------------
        # Very weak offset prior.
        # ----------------------------------------------------

        score -= (
            abs(
                candidate["offset"]
                - 12
            )
            * 0.005
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


    # ========================================================
    # VELOCITY UPDATE
    # ========================================================

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
                * previous_velocity
                +
                0.15
                * measured_velocity
            )


    previous_court = current_court

    previous_ground_y = best[
        "pixel_y"
    ]


    selected.append(
        best.copy()
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

rows = []


for data, result in zip(
    frame_data,
    selected
):

    if result is None:

        rows.append(
            {
                "frame":
                    data["frame"],

                "time_seconds":
                    data[
                        "time_seconds"
                    ],

                "detected": False,

                "confidence": np.nan,

                "box_bottom_y": np.nan,

                "ground_x": np.nan,
                "ground_y": np.nan,

                "offset": np.nan,

                "court_x": np.nan,
                "court_y": np.nan,

                "image_evidence":
                    np.nan
            }
        )

        continue


    player = data[
        "player"
    ]


    rows.append(
        {
            "frame":
                data["frame"],

            "time_seconds":
                data[
                    "time_seconds"
                ],

            "detected": True,

            "confidence":
                player[
                    "confidence"
                ],

            "box_bottom_y":
                player["y2"],

            "ground_x":
                result[
                    "pixel_x"
                ],

            "ground_y":
                result[
                    "pixel_y"
                ],

            "offset":
                result[
                    "offset"
                ],

            "court_x":
                result[
                    "court_x"
                ],

            "court_y":
                result[
                    "court_y"
                ],

            "image_evidence":
                result[
                    "evidence_norm"
                ]
        }
    )


df = pd.DataFrame(
    rows
)

df.to_csv(
    CSV_OUTPUT,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

detected = df[
    df["detected"] == True
]


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

print(
    f"Detection rate: "
    f"{100 * len(detected) / len(df):.1f}%"
)


if len(detected) > 0:

    print()
    print("GROUND OFFSET")

    print(
        f"Mean: "
        f"{detected['offset'].mean():.1f}px"
    )

    print(
        f"Median: "
        f"{detected['offset'].median():.1f}px"
    )

    print(
        f"Min: "
        f"{detected['offset'].min():.1f}px"
    )

    print(
        f"Max: "
        f"{detected['offset'].max():.1f}px"
    )


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


    suspicious = detected[
        detected["court_y"] < -1.5
    ]


    print()
    print(
        "Frames below -1.5m: "
        f"{len(suspicious)} / "
        f"{len(detected)} "
        f"("
        f"{100 * len(suspicious) / len(detected):.1f}%"
        f")"
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
    VIDEO_OUTPUT,
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


    if row["detected"]:

        data = frame_data[i]

        player = data[
            "player"
        ]


        x1 = int(
            round(
                player["x1"]
            )
        )

        y1 = int(
            round(
                player["y1"]
            )
        )

        x2 = int(
            round(
                player["x2"]
            )
        )

        y2 = int(
            round(
                player["y2"]
            )
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2
        )


        # Raw YOLO bottom

        raw_point = (
            int(
                round(
                    player[
                        "centre_x"
                    ]
                )
            ),
            y2
        )

        cv2.circle(
            frame,
            raw_point,
            5,
            (0, 0, 255),
            -1
        )


        # Ground

        ground_point = (
            int(
                round(
                    row[
                        "ground_x"
                    ]
                )
            ),
            int(
                round(
                    row[
                        "ground_y"
                    ]
                )
            )
        )

        cv2.circle(
            frame,
            ground_point,
            7,
            (0, 255, 0),
            -1
        )


        # Baseline

        baseline_y = baseline_y_at_x(
            row[
                "ground_x"
            ]
        )

        cv2.circle(
            frame,
            (
                int(
                    round(
                        row[
                            "ground_x"
                        ]
                    )
                ),
                int(
                    round(
                        baseline_y
                    )
                )
            ),
            5,
            (255, 0, 0),
            -1
        )


        text = (
            f"court "
            f"({row['court_x']:.2f}, "
            f"{row['court_y']:.2f})m "
            f"offset={row['offset']:+.0f}px"
        )


        cv2.putText(
            frame,
            text,
            (
                x1,
                max(
                    25,
                    y1 - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


    # Mark initialisation period

    if i < initialisation_frames:

        cv2.putText(
            frame,
            "INITIALISING",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


    writer.write(
        frame
    )


cap.release()
writer.release()


print()
print(
    f"Saved CSV: {CSV_OUTPUT}"
)

print(
    f"Saved video: {VIDEO_OUTPUT}"
)

print()
print("DONE")