import cv2
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_CSV = "output/far_ground_tracking.csv"

OUTPUT_CSV = "output/far_ground_tracking_smoothed.csv"
OUTPUT_VIDEO = "output/far_ground_tracking_smoothed.mp4"

VIDEO_PATH = "videos/test_match.mp4"

START_SECONDS = 60
DURATION_SECONDS = 10


# ============================================================
# PHYSICAL LIMITS
#
# These are deliberately generous.
#
# We are NOT saying a tennis player actually moves this fast.
# We simply want to catch obviously impossible tracking jumps.
# ============================================================

MAX_SPEED_MPS = 12.0

# Additional tolerance because our measurements are noisy.
MAX_FRAME_JUMP_METRES = 0.45

# Short missing/bad sections can be interpolated.
MAX_INTERPOLATION_GAP = 12


# ============================================================
# LOAD TRACK
# ============================================================

print()
print("FAR TRACK SMOOTHING")
print("=" * 70)

df = pd.read_csv(INPUT_CSV)

print(f"Rows loaded: {len(df)}")


# ============================================================
# VIDEO INFO
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

print(f"FPS: {fps:.2f}")


# ============================================================
# CREATE WORKING COLUMNS
# ============================================================

df["raw_court_x"] = df["court_x"]
df["raw_court_y"] = df["court_y"]

df["valid"] = (
    df["detected"].astype(bool)
    &
    df["court_x"].notna()
    &
    df["court_y"].notna()
)

df["outlier"] = False


# ============================================================
# PASS 1
#
# FORWARD SPEED / JUMP CHECK
# ============================================================

previous_good_index = None


for i in range(len(df)):

    if not df.at[i, "valid"]:
        continue

    if previous_good_index is None:

        previous_good_index = i
        continue


    frame_gap = (
        df.at[i, "frame"]
        -
        df.at[
            previous_good_index,
            "frame"
        ]
    )


    if frame_gap <= 0:
        continue


    time_gap = frame_gap / fps


    x1 = df.at[
        previous_good_index,
        "court_x"
    ]

    y1 = df.at[
        previous_good_index,
        "court_y"
    ]

    x2 = df.at[i, "court_x"]
    y2 = df.at[i, "court_y"]


    distance = np.sqrt(
        (x2 - x1) ** 2
        +
        (y2 - y1) ** 2
    )


    allowed_distance = (
        MAX_SPEED_MPS
        * time_gap
        +
        MAX_FRAME_JUMP_METRES
    )


    if distance > allowed_distance:

        df.at[i, "outlier"] = True

        continue


    previous_good_index = i


# ============================================================
# PASS 2
#
# THREE-POINT CONSISTENCY
#
# This catches a classic pattern:
#
# good
# BAD SPIKE
# good
#
# If the point before and after are close to each other,
# but the middle point is far away, the middle point is almost
# certainly a tracking error.
# ============================================================

for i in range(
    1,
    len(df) - 1
):

    if not df.at[i, "valid"]:
        continue

    if not df.at[i - 1, "valid"]:
        continue

    if not df.at[i + 1, "valid"]:
        continue


    prev_point = np.array(
        [
            df.at[i - 1, "court_x"],
            df.at[i - 1, "court_y"]
        ]
    )

    current_point = np.array(
        [
            df.at[i, "court_x"],
            df.at[i, "court_y"]
        ]
    )

    next_point = np.array(
        [
            df.at[i + 1, "court_x"],
            df.at[i + 1, "court_y"]
        ]
    )


    neighbour_distance = np.linalg.norm(
        next_point - prev_point
    )

    current_from_prev = np.linalg.norm(
        current_point - prev_point
    )

    current_from_next = np.linalg.norm(
        current_point - next_point
    )


    # Neighbours agree with each other,
    # while current point is far from both.

    if (
        neighbour_distance < 0.70
        and
        current_from_prev > 0.80
        and
        current_from_next > 0.80
    ):

        df.at[i, "outlier"] = True


# ============================================================
# REMOVE OUTLIERS
# ============================================================

outlier_count = int(
    df["outlier"].sum()
)


print()
print("OUTLIER DETECTION")
print("=" * 70)

print(
    f"Outliers found: "
    f"{outlier_count}"
)


df.loc[
    df["outlier"],
    "court_x"
] = np.nan

df.loc[
    df["outlier"],
    "court_y"
] = np.nan


# ============================================================
# INTERPOLATE SHORT GAPS
#
# This uses BOTH the previous and future good positions.
#
# That's the important difference from the previous
# forward-only tracker.
# ============================================================

df["court_x"] = (
    df["court_x"]
    .interpolate(
        method="linear",
        limit=MAX_INTERPOLATION_GAP,
        limit_direction="both",
        limit_area="inside"
    )
)

df["court_y"] = (
    df["court_y"]
    .interpolate(
        method="linear",
        limit=MAX_INTERPOLATION_GAP,
        limit_direction="both",
        limit_area="inside"
    )
)


# ============================================================
# LIGHT TEMPORAL SMOOTHING
#
# Centered rolling median uses:
#
# previous frames
# current frame
# future frames
#
# Median is much more resistant to spikes than an average.
# ============================================================

df["smooth_court_x"] = (
    df["court_x"]
    .rolling(
        window=5,
        center=True,
        min_periods=1
    )
    .median()
)

df["smooth_court_y"] = (
    df["court_y"]
    .rolling(
        window=5,
        center=True,
        min_periods=1
    )
    .median()
)


# ============================================================
# CALCULATE SMOOTHED SPEED
# ============================================================

df["speed_mps"] = np.nan


for i in range(
    1,
    len(df)
):

    x1 = df.at[
        i - 1,
        "smooth_court_x"
    ]

    y1 = df.at[
        i - 1,
        "smooth_court_y"
    ]

    x2 = df.at[
        i,
        "smooth_court_x"
    ]

    y2 = df.at[
        i,
        "smooth_court_y"
    ]


    if (
        pd.isna(x1)
        or pd.isna(y1)
        or pd.isna(x2)
        or pd.isna(y2)
    ):
        continue


    distance = np.sqrt(
        (x2 - x1) ** 2
        +
        (y2 - y1) ** 2
    )


    df.at[
        i,
        "speed_mps"
    ] = distance * fps


# ============================================================
# SUMMARY
# ============================================================

valid_raw = df[
    df["raw_court_y"].notna()
]

valid_smooth = df[
    df["smooth_court_y"].notna()
]


print()
print("RAW COURT Y")
print("=" * 70)

if len(valid_raw) > 0:

    print(
        f"Mean: "
        f"{valid_raw['raw_court_y'].mean():.2f}m"
    )

    print(
        f"Median: "
        f"{valid_raw['raw_court_y'].median():.2f}m"
    )

    print(
        f"Min: "
        f"{valid_raw['raw_court_y'].min():.2f}m"
    )

    print(
        f"Max: "
        f"{valid_raw['raw_court_y'].max():.2f}m"
    )


print()
print("SMOOTHED COURT Y")
print("=" * 70)

if len(valid_smooth) > 0:

    print(
        f"Mean: "
        f"{valid_smooth['smooth_court_y'].mean():.2f}m"
    )

    print(
        f"Median: "
        f"{valid_smooth['smooth_court_y'].median():.2f}m"
    )

    print(
        f"Min: "
        f"{valid_smooth['smooth_court_y'].min():.2f}m"
    )

    print(
        f"Max: "
        f"{valid_smooth['smooth_court_y'].max():.2f}m"
    )


valid_speed = df[
    df["speed_mps"].notna()
]


if len(valid_speed) > 0:

    print()
    print("SMOOTHED SPEED")
    print("=" * 70)

    print(
        f"Mean: "
        f"{valid_speed['speed_mps'].mean():.2f} m/s"
    )

    print(
        f"95th percentile: "
        f"{valid_speed['speed_mps'].quantile(0.95):.2f} m/s"
    )

    print(
        f"Max: "
        f"{valid_speed['speed_mps'].max():.2f} m/s"
    )


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print()
print(
    f"Saved CSV: {OUTPUT_CSV}"
)


# ============================================================
# CREATE DEBUG VIDEO
#
# We cannot convert the smoothed court coordinates back to
# image pixels yet without using the court homography again.
#
# So for this debug video we show:
#
# - original detected ground point
# - whether that frame was rejected
# - raw vs smoothed court coordinates
# ============================================================

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError(
        "Could not create output video"
    )


cap.set(
    cv2.CAP_PROP_POS_MSEC,
    START_SECONDS * 1000
)


for i in range(
    min(
        len(df),
        int(DURATION_SECONDS * fps)
    )
):

    success, frame = cap.read()

    if not success:
        break


    # --------------------------------------------------------
    # Draw original ground point if available
    # --------------------------------------------------------

    gx = df.at[i, "ground_x"]
    gy = df.at[i, "ground_y"]


    if (
        not pd.isna(gx)
        and
        not pd.isna(gy)
    ):

        point = (
            int(round(gx)),
            int(round(gy))
        )


        if df.at[i, "outlier"]:

            # Rejected measurement

            cv2.circle(
                frame,
                point,
                8,
                (0, 0, 255),
                2
            )

            cv2.line(
                frame,
                (
                    point[0] - 8,
                    point[1] - 8
                ),
                (
                    point[0] + 8,
                    point[1] + 8
                ),
                (0, 0, 255),
                2
            )

            cv2.line(
                frame,
                (
                    point[0] + 8,
                    point[1] - 8
                ),
                (
                    point[0] - 8,
                    point[1] + 8
                ),
                (0, 0, 255),
                2
            )

        else:

            cv2.circle(
                frame,
                point,
                6,
                (0, 255, 0),
                -1
            )


    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    raw_x = df.at[
        i,
        "raw_court_x"
    ]

    raw_y = df.at[
        i,
        "raw_court_y"
    ]

    smooth_x = df.at[
        i,
        "smooth_court_x"
    ]

    smooth_y = df.at[
        i,
        "smooth_court_y"
    ]


    if (
        not pd.isna(raw_x)
        and
        not pd.isna(raw_y)
    ):

        raw_text = (
            f"RAW: "
            f"({raw_x:.2f}, "
            f"{raw_y:.2f})m"
        )

    else:

        raw_text = "RAW: missing"


    if (
        not pd.isna(smooth_x)
        and
        not pd.isna(smooth_y)
    ):

        smooth_text = (
            f"SMOOTH: "
            f"({smooth_x:.2f}, "
            f"{smooth_y:.2f})m"
        )

    else:

        smooth_text = (
            "SMOOTH: missing"
        )


    cv2.putText(
        frame,
        raw_text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        smooth_text,
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    if df.at[i, "outlier"]:

        cv2.putText(
            frame,
            "OUTLIER REJECTED",
            (30, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    writer.write(frame)


cap.release()
writer.release()


print(
    f"Saved video: {OUTPUT_VIDEO}"
)

print()
print("DONE")