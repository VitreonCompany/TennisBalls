import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "videos/test_match.mp4"
OUTPUT_PATH = "output/far_feet_grid.jpg"

START_SECONDS = 60
DURATION_SECONDS = 10

# Number of samples across the 10 seconds
NUM_SAMPLES = 20

# Far-player search crop
CROP_X1 = 0
CROP_Y1 = 120
CROP_X2 = 1000
CROP_Y2 = 520

# How much extra image to include around detected player
PAD_LEFT_RIGHT = 35
PAD_TOP = 20
PAD_BOTTOM = 45

# Enlargement of each player crop
DISPLAY_SCALE = 4


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    raise RuntimeError("Could not determine video FPS")

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)


print()
print("FAR PLAYER FEET INSPECTION")
print("=" * 70)

print(f"FPS: {fps:.2f}")
print(f"Video size: {width} x {height}")
print(f"Samples requested: {NUM_SAMPLES}")


# ============================================================
# LOAD YOLO
# ============================================================

print()
print("Loading YOLO...")

model = YOLO("yolo11n.pt")


# ============================================================
# SAMPLE TIMES
# ============================================================

sample_times = np.linspace(
    START_SECONDS,
    START_SECONDS + DURATION_SECONDS,
    NUM_SAMPLES,
    endpoint=False
)


tiles = []


# ============================================================
# PROCESS EACH SAMPLE
# ============================================================

for sample_number, time_seconds in enumerate(
    sample_times,
    start=1
):

    print(
        f"Sample {sample_number:02d}/{NUM_SAMPLES} "
        f"at {time_seconds:.2f}s"
    )


    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    frame_number = int(
        time_seconds * fps
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )

    success, frame = cap.read()

    if not success:
        print("  Could not read frame")
        continue


    # --------------------------------------------------------
    # FAR COURT CROP
    # --------------------------------------------------------

    search_crop = frame[
        CROP_Y1:CROP_Y2,
        CROP_X1:CROP_X2
    ]


    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    results = model.predict(
        search_crop,
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


        # Convert to full-frame coordinates

        x1 += CROP_X1
        x2 += CROP_X1

        y1 += CROP_Y1
        y2 += CROP_Y1


        box_height = y2 - y1
        centre_x = (
            x1 + x2
        ) / 2


        # Broad filters for far player

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


        candidates.append(
            {
                "confidence": confidence,

                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,

                "height": box_height,

                "centre_x": centre_x
            }
        )


    if not candidates:

        print("  No player detected")
        continue


    # Highest confidence plausible player

    best = max(
        candidates,
        key=lambda d:
            d["confidence"]
    )


    x1 = best["x1"]
    y1 = best["y1"]
    x2 = best["x2"]
    y2 = best["y2"]

    confidence = best[
        "confidence"
    ]

    box_height = best[
        "height"
    ]


    # ========================================================
    # CREATE PADDED PLAYER CROP
    # ========================================================

    px1 = max(
        0,
        int(
            np.floor(
                x1 - PAD_LEFT_RIGHT
            )
        )
    )

    py1 = max(
        0,
        int(
            np.floor(
                y1 - PAD_TOP
            )
        )
    )

    px2 = min(
        width,
        int(
            np.ceil(
                x2 + PAD_LEFT_RIGHT
            )
        )
    )

    py2 = min(
        height,
        int(
            np.ceil(
                y2 + PAD_BOTTOM
            )
        )
    )


    player_crop = frame[
        py1:py2,
        px1:px2
    ].copy()


    # ========================================================
    # YOLO BOX POSITION INSIDE PLAYER CROP
    # ========================================================

    local_x1 = int(
        round(x1 - px1)
    )

    local_y1 = int(
        round(y1 - py1)
    )

    local_x2 = int(
        round(x2 - px1)
    )

    local_y2 = int(
        round(y2 - py1)
    )


    # Draw box

    cv2.rectangle(
        player_crop,
        (
            local_x1,
            local_y1
        ),
        (
            local_x2,
            local_y2
        ),
        (0, 255, 255),
        1
    )


    # ========================================================
    # RAW YOLO BOTTOM
    # ========================================================

    centre_x = int(
        round(
            (
                local_x1
                +
                local_x2
            )
            / 2
        )
    )

    raw_bottom_y = local_y2


    cv2.circle(
        player_crop,
        (
            centre_x,
            raw_bottom_y
        ),
        2,
        (0, 0, 255),
        -1
    )


    # Draw horizontal line at YOLO bottom

    cv2.line(
        player_crop,
        (
            max(
                0,
                local_x1 - 15
            ),
            raw_bottom_y
        ),
        (
            min(
                player_crop.shape[1] - 1,
                local_x2 + 15
            ),
            raw_bottom_y
        ),
        (0, 0, 255),
        1
    )


    # ========================================================
    # DRAW REFERENCE LINES BELOW YOLO BOX
    #
    # These let us visually inspect where the actual feet are.
    #
    # +5 / +10 / +15 / +20 / +25 pixels
    # ========================================================

    for offset in [
        5,
        10,
        15,
        20,
        25
    ]:

        test_y = (
            raw_bottom_y
            +
            offset
        )

        if (
            test_y
            >= player_crop.shape[0]
        ):
            continue


        cv2.line(
            player_crop,
            (
                max(
                    0,
                    local_x1 - 10
                ),
                test_y
            ),
            (
                min(
                    player_crop.shape[1] - 1,
                    local_x2 + 25
                ),
                test_y
            ),
            (255, 0, 255),
            1
        )


        cv2.putText(
            player_crop,
            f"+{offset}",
            (
                min(
                    player_crop.shape[1] - 25,
                    local_x2 + 5
                ),
                test_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.25,
            (255, 0, 255),
            1,
            cv2.LINE_AA
        )


    # ========================================================
    # ENLARGE
    # ========================================================

    enlarged = cv2.resize(
        player_crop,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_NEAREST
    )


    # ========================================================
    # ADD HEADER
    # ========================================================

    header_height = 55

    tile = np.zeros(
        (
            enlarged.shape[0]
            + header_height,
            enlarged.shape[1],
            3
        ),
        dtype=np.uint8
    )

    tile[
        header_height:,
        :
    ] = enlarged


    cv2.putText(
        tile,
        f"{time_seconds:.1f}s",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        tile,
        (
            f"conf {confidence:.2f} "
            f"h {box_height:.0f}px"
        ),
        (8, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    tiles.append(
        tile
    )


cap.release()


# ============================================================
# CHECK
# ============================================================

if not tiles:
    raise RuntimeError(
        "No far-player crops were generated"
    )


print()
print(
    f"Successfully created "
    f"{len(tiles)} crops."
)


# ============================================================
# NORMALISE TILE SIZE
# ============================================================

max_width = max(
    tile.shape[1]
    for tile in tiles
)

max_height = max(
    tile.shape[0]
    for tile in tiles
)


normalised_tiles = []


for tile in tiles:

    canvas = np.zeros(
        (
            max_height,
            max_width,
            3
        ),
        dtype=np.uint8
    )

    canvas[
        :tile.shape[0],
        :tile.shape[1]
    ] = tile

    normalised_tiles.append(
        canvas
    )


# ============================================================
# CREATE GRID
#
# 4 columns x 5 rows for 20 samples.
# ============================================================

columns = 4

rows = int(
    np.ceil(
        len(normalised_tiles)
        / columns
    )
)


blank_tile = np.zeros(
    (
        max_height,
        max_width,
        3
    ),
    dtype=np.uint8
)


while (
    len(normalised_tiles)
    <
    rows * columns
):

    normalised_tiles.append(
        blank_tile.copy()
    )


grid_rows = []


for row_index in range(rows):

    start = (
        row_index
        *
        columns
    )

    end = (
        start
        +
        columns
    )

    row = cv2.hconcat(
        normalised_tiles[
            start:end
        ]
    )

    grid_rows.append(
        row
    )


grid = cv2.vconcat(
    grid_rows
)


# ============================================================
# SAVE
# ============================================================

cv2.imwrite(
    OUTPUT_PATH,
    grid
)


print()
print("LEGEND")
print("=" * 70)

print(
    "Yellow box = YOLO person detection"
)

print(
    "Red line/dot = YOLO box bottom"
)

print(
    "Magenta lines = +5, +10, +15, +20, +25 pixels"
)

print()
print(
    f"Saved: {OUTPUT_PATH}"
)