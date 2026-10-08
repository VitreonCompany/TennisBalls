import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "videos/test_match.mp4"

OUTPUT_IMAGE = "output/near_feet_grid.jpg"

START_SECONDS = 60
DURATION_SECONDS = 10

# One sample every half second
SAMPLE_INTERVAL_SECONDS = 0.5


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


print()
print("NEAR PLAYER FEET TEST")
print("=" * 70)

print(f"FPS: {fps:.2f}")
print(f"Video size: {width} x {height}")


# ============================================================
# LOAD POSE MODEL
# ============================================================

print()
print("Loading pose model...")

model = YOLO(
    "yolo11n-pose.pt"
)


# ============================================================
# SAMPLE TIMES
# ============================================================

sample_times = np.arange(
    START_SECONDS,
    START_SECONDS + DURATION_SECONDS,
    SAMPLE_INTERVAL_SECONDS
)

print(
    f"Samples requested: "
    f"{len(sample_times)}"
)

print()


# ============================================================
# COCO POSE KEYPOINTS
#
# 15 = left ankle
# 16 = right ankle
# ============================================================

LEFT_ANKLE = 15
RIGHT_ANKLE = 16


# ============================================================
# STORAGE
# ============================================================

crops = []

successful_samples = 0


# ============================================================
# PROCESS SAMPLES
# ============================================================

for sample_number, time_seconds in enumerate(
    sample_times,
    start=1
):

    print(
        f"Sample "
        f"{sample_number:02d}/"
        f"{len(sample_times):02d} "
        f"at {time_seconds:.2f}s"
    )


    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        time_seconds * 1000
    )

    success, frame = cap.read()

    if not success:

        print("  Could not read frame")
        continue


    # ========================================================
    # NEAR PLAYER CROP
    #
    # We deliberately use the right side of the frame.
    # This gives pose estimation more pixels for the player.
    # ========================================================

    crop_x1 = int(
        width * 0.45
    )

    crop_y1 = int(
        height * 0.20
    )

    crop_x2 = width
    crop_y2 = height


    crop = frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]


    # ========================================================
    # POSE DETECTION
    # ========================================================

    results = model.predict(
        crop,
        conf=0.20,
        imgsz=1280,
        verbose=False
    )


    candidates = []


    for result in results:

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


        # Some Ultralytics versions expose
        # keypoint confidence separately.

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


            # Crop -> full image

            x1 += crop_x1
            x2 += crop_x1

            y1 += crop_y1
            y2 += crop_y1


            box_height = y2 - y1

            centre_x = (
                x1 + x2
            ) / 2


            # ------------------------------------------------
            # Near-player filtering
            # ------------------------------------------------

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
                box_height / 1000.0
            )


            candidates.append(
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

                    "keypoints":
                        keypoints_xy[
                            detection_index
                        ].copy(),

                    "keypoint_conf":
                        (
                            keypoints_conf[
                                detection_index
                            ].copy()
                            if keypoints_conf
                            is not None
                            else None
                        )
                }
            )


    # ========================================================
    # NO PLAYER
    # ========================================================

    if not candidates:

        print(
            "  No near player detected"
        )

        continue


    # ========================================================
    # SELECT NEAR PLAYER
    # ========================================================

    player = max(
        candidates,
        key=lambda d:
            d["score"]
    )


    x1 = player["x1"]
    y1 = player["y1"]
    x2 = player["x2"]
    y2 = player["y2"]


    # ========================================================
    # KEYPOINTS
    #
    # Convert crop coordinates back to full-frame coordinates.
    # ========================================================

    keypoints = player[
        "keypoints"
    ]


    keypoints[:, 0] += crop_x1
    keypoints[:, 1] += crop_y1


    left_ankle = keypoints[
        LEFT_ANKLE
    ]

    right_ankle = keypoints[
        RIGHT_ANKLE
    ]


    # ========================================================
    # KEYPOINT CONFIDENCE
    # ========================================================

    if (
        player["keypoint_conf"]
        is not None
    ):

        left_conf = float(
            player[
                "keypoint_conf"
            ][LEFT_ANKLE]
        )

        right_conf = float(
            player[
                "keypoint_conf"
            ][RIGHT_ANKLE]
        )

    else:

        left_conf = 1.0
        right_conf = 1.0


    # ========================================================
    # BOX BOTTOM
    # ========================================================

    box_bottom = np.array(
        [
            (x1 + x2) / 2,
            y2
        ],
        dtype=np.float64
    )


    # ========================================================
    # CHOOSE POSE GROUND POINT
    #
    # If both ankles are reliable:
    #
    # x = midpoint between ankles
    # y = LOWER ankle
    #
    # Important:
    # averaging ankle Y can move the ground point upward
    # when one foot is raised during a stroke.
    #
    # The lower ankle is normally the better ground-contact
    # estimate.
    # ========================================================

    MIN_ANKLE_CONF = 0.35


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


    if left_valid and right_valid:

        ground_x = (
            left_ankle[0]
            +
            right_ankle[0]
        ) / 2

        ground_y = max(
            left_ankle[1],
            right_ankle[1]
        )

        ground_method = (
            "both ankles"
        )


    elif left_valid:

        ground_x = (
            left_ankle[0]
        )

        ground_y = (
            left_ankle[1]
        )

        ground_method = (
            "left ankle"
        )


    elif right_valid:

        ground_x = (
            right_ankle[0]
        )

        ground_y = (
            right_ankle[1]
        )

        ground_method = (
            "right ankle"
        )


    else:

        ground_x = (
            box_bottom[0]
        )

        ground_y = (
            box_bottom[1]
        )

        ground_method = (
            "box fallback"
        )


    pose_ground = np.array(
        [
            ground_x,
            ground_y
        ],
        dtype=np.float64
    )


    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"  detection="
        f"{player['confidence']:.3f}"
    )

    print(
        f"  left ankle="
        f"({left_ankle[0]:.1f}, "
        f"{left_ankle[1]:.1f}) "
        f"conf={left_conf:.3f}"
    )

    print(
        f"  right ankle="
        f"({right_ankle[0]:.1f}, "
        f"{right_ankle[1]:.1f}) "
        f"conf={right_conf:.3f}"
    )

    print(
        f"  method="
        f"{ground_method}"
    )

    print(
        f"  box bottom="
        f"({box_bottom[0]:.1f}, "
        f"{box_bottom[1]:.1f})"
    )

    print(
        f"  pose ground="
        f"({pose_ground[0]:.1f}, "
        f"{pose_ground[1]:.1f})"
    )


    # ========================================================
    # CREATE ENLARGED PLAYER CROP
    # ========================================================

    margin_x = 70
    margin_top = 50
    margin_bottom = 80


    view_x1 = max(
        0,
        int(x1 - margin_x)
    )

    view_y1 = max(
        0,
        int(y1 - margin_top)
    )

    view_x2 = min(
        width,
        int(x2 + margin_x)
    )

    view_y2 = min(
        height,
        int(y2 + margin_bottom)
    )


    view = frame[
        view_y1:view_y2,
        view_x1:view_x2
    ].copy()


    # ========================================================
    # CONVERT POINTS TO LOCAL CROP COORDINATES
    # ========================================================

    local_x1 = int(
        round(
            x1 - view_x1
        )
    )

    local_y1 = int(
        round(
            y1 - view_y1
        )
    )

    local_x2 = int(
        round(
            x2 - view_x1
        )
    )

    local_y2 = int(
        round(
            y2 - view_y1
        )
    )


    local_box_bottom = (
        int(
            round(
                box_bottom[0]
                -
                view_x1
            )
        ),
        int(
            round(
                box_bottom[1]
                -
                view_y1
            )
        )
    )


    local_ground = (
        int(
            round(
                pose_ground[0]
                -
                view_x1
            )
        ),
        int(
            round(
                pose_ground[1]
                -
                view_y1
            )
        )
    )


    local_left_ankle = (
        int(
            round(
                left_ankle[0]
                -
                view_x1
            )
        ),
        int(
            round(
                left_ankle[1]
                -
                view_y1
            )
        )
    )


    local_right_ankle = (
        int(
            round(
                right_ankle[0]
                -
                view_x1
            )
        ),
        int(
            round(
                right_ankle[1]
                -
                view_y1
            )
        )
    )


    # ========================================================
    # DRAW
    # ========================================================

    # Yellow = player box

    cv2.rectangle(
        view,
        (
            local_x1,
            local_y1
        ),
        (
            local_x2,
            local_y2
        ),
        (0, 255, 255),
        2
    )


    # Red = bounding-box bottom

    cv2.circle(
        view,
        local_box_bottom,
        6,
        (0, 0, 255),
        -1
    )


    # Left ankle

    if left_valid:

        cv2.circle(
            view,
            local_left_ankle,
            6,
            (255, 0, 255),
            -1
        )


    # Right ankle

    if right_valid:

        cv2.circle(
            view,
            local_right_ankle,
            6,
            (255, 255, 0),
            -1
        )


    # Green = selected pose ground

    cv2.circle(
        view,
        local_ground,
        8,
        (0, 255, 0),
        2
    )


    # ========================================================
    # LABEL
    # ========================================================

    cv2.putText(
        view,
        f"{time_seconds:.2f}s",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        view,
        ground_method,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )


    crops.append(
        view
    )

    successful_samples += 1


cap.release()


# ============================================================
# BUILD CONTACT SHEET
# ============================================================

if not crops:

    raise RuntimeError(
        "No successful near-player samples"
    )


# Standardise crop size for grid

CELL_WIDTH = 400
CELL_HEIGHT = 400


cells = []


for crop in crops:

    h, w = crop.shape[:2]

    scale = min(
        CELL_WIDTH / w,
        CELL_HEIGHT / h
    )

    new_width = int(
        w * scale
    )

    new_height = int(
        h * scale
    )


    resized = cv2.resize(
        crop,
        (
            new_width,
            new_height
        )
    )


    cell = np.zeros(
        (
            CELL_HEIGHT,
            CELL_WIDTH,
            3
        ),
        dtype=np.uint8
    )


    x_offset = (
        CELL_WIDTH
        -
        new_width
    ) // 2

    y_offset = (
        CELL_HEIGHT
        -
        new_height
    ) // 2


    cell[
        y_offset:
        y_offset + new_height,

        x_offset:
        x_offset + new_width
    ] = resized


    cells.append(
        cell
    )


# ============================================================
# GRID
# ============================================================

columns = 4

rows = int(
    np.ceil(
        len(cells)
        /
        columns
    )
)


grid = np.zeros(
    (
        rows * CELL_HEIGHT,
        columns * CELL_WIDTH,
        3
    ),
    dtype=np.uint8
)


for i, cell in enumerate(cells):

    row = i // columns
    column = i % columns

    y1 = (
        row
        *
        CELL_HEIGHT
    )

    x1 = (
        column
        *
        CELL_WIDTH
    )


    grid[
        y1:
        y1 + CELL_HEIGHT,

        x1:
        x1 + CELL_WIDTH
    ] = cell


cv2.imwrite(
    OUTPUT_IMAGE,
    grid
)


# ============================================================
# DONE
# ============================================================

print()
print("RESULTS")
print("=" * 70)

print(
    f"Successful samples: "
    f"{successful_samples}/"
    f"{len(sample_times)}"
)

print()
print("LEGEND")
print("=" * 70)

print(
    "Yellow box = pose player detection"
)

print(
    "Red dot = bounding-box bottom"
)

print(
    "Magenta = left ankle"
)

print(
    "Cyan = right ankle"
)

print(
    "Green circle = chosen ground point"
)

print()
print(
    f"Saved: {OUTPUT_IMAGE}"
)