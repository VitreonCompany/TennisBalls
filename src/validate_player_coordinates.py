import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "videos/test_match.mp4"
HOMOGRAPHY_PATH = "data/homography.npy"
MODEL_PATH = "yolo11n.pt"

OUTPUT_DIR = Path("output")

# ------------------------------------------------------------
# TEST SECTIONS
#
# Each number is the start time in seconds.
#
# 60  = 1:00
# 180 = 3:00
# 360 = 6:00
# 600 = 10:00
# ------------------------------------------------------------

TEST_START_TIMES = [
    60,
    180,
    360,
    600,
]

TEST_DURATION = 10


# ============================================================
# YOLO SETTINGS
# ============================================================

CONFIDENCE = 0.10
IMAGE_SIZE = 1280


# ============================================================
# FAR PLAYER CROP
# ============================================================

FAR_CROP_X1 = 0
FAR_CROP_Y1 = 120
FAR_CROP_X2 = 1000
FAR_CROP_Y2 = 520


# ============================================================
# TENNIS COURT DIMENSIONS
# ============================================================

COURT_LENGTH = 23.77
HALF_COURT_LENGTH = COURT_LENGTH / 2

SINGLES_WIDTH = 8.23
HALF_SINGLES_WIDTH = SINGLES_WIDTH / 2

DOUBLES_WIDTH = 10.97
HALF_DOUBLES_WIDTH = DOUBLES_WIDTH / 2

SERVICE_LINE_DISTANCE_FROM_NET = 6.40


# ============================================================
# TOP-DOWN COURT DISPLAY SETTINGS
# ============================================================

COURT_VIEW_WIDTH = 600
COURT_VIEW_HEIGHT = 900

COURT_MARGIN_X = 80
COURT_MARGIN_Y = 70

COURT_DRAW_WIDTH = (
    COURT_VIEW_WIDTH -
    COURT_MARGIN_X * 2
)

COURT_DRAW_HEIGHT = (
    COURT_VIEW_HEIGHT -
    COURT_MARGIN_Y * 2
)


# ============================================================
# LOAD HOMOGRAPHY
# ============================================================

print()
print("Loading homography...")

H = np.load(
    HOMOGRAPHY_PATH
)

print("Homography loaded:")

print(
    np.array2string(
        H,
        precision=5,
        suppress_small=True
    )
)


# ============================================================
# LOAD YOLO
# ============================================================

print()
print("Loading YOLO model...")

model = YOLO(
    MODEL_PATH
)

print("YOLO loaded.")


# ============================================================
# PIXEL -> COURT COORDINATES
# ============================================================

def pixel_to_court(x, y):

    point = np.array(
        [[[x, y]]],
        dtype=np.float32
    )

    transformed = cv2.perspectiveTransform(
        point,
        H
    )[0][0]

    return (
        float(transformed[0]),
        float(transformed[1])
    )


# ============================================================
# COURT COORDINATE -> DISPLAY PIXEL
# ============================================================

def court_to_display(
    court_x,
    court_y
):

    normalized_x = (
        court_x +
        HALF_DOUBLES_WIDTH
    ) / DOUBLES_WIDTH

    normalized_y = (
        court_y +
        HALF_COURT_LENGTH
    ) / COURT_LENGTH

    display_x = int(
        COURT_MARGIN_X +
        normalized_x *
        COURT_DRAW_WIDTH
    )

    display_y = int(
        COURT_MARGIN_Y +
        normalized_y *
        COURT_DRAW_HEIGHT
    )

    return (
        display_x,
        display_y
    )


# ============================================================
# DRAW TOP-DOWN TENNIS COURT
# ============================================================

def draw_top_down_court():

    court = np.zeros(
        (
            COURT_VIEW_HEIGHT,
            COURT_VIEW_WIDTH,
            3
        ),
        dtype=np.uint8
    )

    court[:] = (
        30,
        30,
        30
    )

    # --------------------------------------------------------
    # Court surface
    # --------------------------------------------------------

    top_left = court_to_display(
        -HALF_DOUBLES_WIDTH,
        -HALF_COURT_LENGTH
    )

    bottom_right = court_to_display(
        HALF_DOUBLES_WIDTH,
        HALF_COURT_LENGTH
    )

    cv2.rectangle(
        court,
        top_left,
        bottom_right,
        (65, 120, 65),
        -1
    )

    line_colour = (
        255,
        255,
        255
    )

    line_thickness = 2

    # --------------------------------------------------------
    # Doubles sidelines
    # --------------------------------------------------------

    for x in [
        -HALF_DOUBLES_WIDTH,
        HALF_DOUBLES_WIDTH
    ]:

        p1 = court_to_display(
            x,
            -HALF_COURT_LENGTH
        )

        p2 = court_to_display(
            x,
            HALF_COURT_LENGTH
        )

        cv2.line(
            court,
            p1,
            p2,
            line_colour,
            line_thickness
        )

    # --------------------------------------------------------
    # Baselines
    # --------------------------------------------------------

    for y in [
        -HALF_COURT_LENGTH,
        HALF_COURT_LENGTH
    ]:

        p1 = court_to_display(
            -HALF_DOUBLES_WIDTH,
            y
        )

        p2 = court_to_display(
            HALF_DOUBLES_WIDTH,
            y
        )

        cv2.line(
            court,
            p1,
            p2,
            line_colour,
            line_thickness
        )

    # --------------------------------------------------------
    # Singles sidelines
    # --------------------------------------------------------

    for x in [
        -HALF_SINGLES_WIDTH,
        HALF_SINGLES_WIDTH
    ]:

        p1 = court_to_display(
            x,
            -HALF_COURT_LENGTH
        )

        p2 = court_to_display(
            x,
            HALF_COURT_LENGTH
        )

        cv2.line(
            court,
            p1,
            p2,
            line_colour,
            line_thickness
        )

    # --------------------------------------------------------
    # Service lines
    # --------------------------------------------------------

    for y in [
        -SERVICE_LINE_DISTANCE_FROM_NET,
        SERVICE_LINE_DISTANCE_FROM_NET
    ]:

        p1 = court_to_display(
            -HALF_SINGLES_WIDTH,
            y
        )

        p2 = court_to_display(
            HALF_SINGLES_WIDTH,
            y
        )

        cv2.line(
            court,
            p1,
            p2,
            line_colour,
            line_thickness
        )

    # --------------------------------------------------------
    # Centre service line
    # --------------------------------------------------------

    p1 = court_to_display(
        0,
        -SERVICE_LINE_DISTANCE_FROM_NET
    )

    p2 = court_to_display(
        0,
        SERVICE_LINE_DISTANCE_FROM_NET
    )

    cv2.line(
        court,
        p1,
        p2,
        line_colour,
        line_thickness
    )

    # --------------------------------------------------------
    # Net
    # --------------------------------------------------------

    net_left = court_to_display(
        -HALF_DOUBLES_WIDTH,
        0
    )

    net_right = court_to_display(
        HALF_DOUBLES_WIDTH,
        0
    )

    cv2.line(
        court,
        net_left,
        net_right,
        (220, 220, 220),
        4
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    cv2.putText(
        court,
        "FAR PLAYER SIDE",
        (180, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        court,
        "NEAR PLAYER SIDE",
        (
            170,
            COURT_VIEW_HEIGHT - 25
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return court


# ============================================================
# DETECT NEAR PLAYER
# ============================================================

def detect_near_player(frame):

    results = model(
        frame,
        classes=[0],
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False
    )[0]

    candidates = []

    for box in results.boxes:

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Current V1 ground-point estimate:
        #
        # bottom-centre of bounding box
        # ----------------------------------------------------

        foot_x = (
            x1 + x2
        ) / 2

        foot_y = y2

        court_x, court_y = (
            pixel_to_court(
                foot_x,
                foot_y
            )
        )

        # ----------------------------------------------------
        # Near-player region
        # ----------------------------------------------------

        if (
            11.885
            <
            court_y
            <
            30
        ):

            candidates.append(
                {
                    "confidence":
                        float(box.conf[0]),

                    "x1":
                        float(x1),

                    "y1":
                        float(y1),

                    "x2":
                        float(x2),

                    "y2":
                        float(y2),

                    "foot_x":
                        float(foot_x),

                    "foot_y":
                        float(foot_y),

                    "court_x":
                        court_x,

                    "court_y":
                        court_y
                }
            )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item:
            item["confidence"]
    )


# ============================================================
# DETECT FAR PLAYER
# ============================================================

def detect_far_player(frame):

    height, width = (
        frame.shape[:2]
    )

    crop_x1 = max(
        0,
        min(
            FAR_CROP_X1,
            width
        )
    )

    crop_y1 = max(
        0,
        min(
            FAR_CROP_Y1,
            height
        )
    )

    crop_x2 = max(
        crop_x1 + 1,
        min(
            FAR_CROP_X2,
            width
        )
    )

    crop_y2 = max(
        crop_y1 + 1,
        min(
            FAR_CROP_Y2,
            height
        )
    )

    crop = frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]

    results = model(
        crop,
        classes=[0],
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False
    )[0]

    candidates = []

    for box in results.boxes:

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Convert crop coordinates back to original image
        # ----------------------------------------------------

        x1 += crop_x1
        x2 += crop_x1

        y1 += crop_y1
        y2 += crop_y1

        # ----------------------------------------------------
        # Current V1 ground-point estimate
        # ----------------------------------------------------

        foot_x = (
            x1 + x2
        ) / 2

        foot_y = y2

        court_x, court_y = (
            pixel_to_court(
                foot_x,
                foot_y
            )
        )

        # ----------------------------------------------------
        # Far-player region
        # ----------------------------------------------------

        if (
            -15
            <
            court_y
            <
            11.885
        ):

            candidates.append(
                {
                    "confidence":
                        float(box.conf[0]),

                    "x1":
                        float(x1),

                    "y1":
                        float(y1),

                    "x2":
                        float(x2),

                    "y2":
                        float(y2),

                    "foot_x":
                        float(foot_x),

                    "foot_y":
                        float(foot_y),

                    "court_x":
                        court_x,

                    "court_y":
                        court_y
                }
            )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item:
            item["confidence"]
    )


# ============================================================
# DRAW PLAYER ON ORIGINAL VIDEO
# ============================================================

def draw_player_on_frame(
    frame,
    player,
    label,
    colour
):

    if player is None:
        return

    x1 = int(
        player["x1"]
    )

    y1 = int(
        player["y1"]
    )

    x2 = int(
        player["x2"]
    )

    y2 = int(
        player["y2"]
    )

    foot_x = int(
        player["foot_x"]
    )

    foot_y = int(
        player["foot_y"]
    )

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        colour,
        2
    )

    # --------------------------------------------------------
    # Ground point
    # --------------------------------------------------------

    cv2.circle(
        frame,
        (
            foot_x,
            foot_y
        ),
        8,
        colour,
        -1
    )

    # --------------------------------------------------------
    # Ground-point crosshair
    # --------------------------------------------------------

    cv2.line(
        frame,
        (
            foot_x - 15,
            foot_y
        ),
        (
            foot_x + 15,
            foot_y
        ),
        colour,
        2
    )

    cv2.line(
        frame,
        (
            foot_x,
            foot_y - 15
        ),
        (
            foot_x,
            foot_y + 15
        ),
        colour,
        2
    )

    # --------------------------------------------------------
    # Coordinate text
    # --------------------------------------------------------

    text = (
        f"{label}  "
        f"X={player['court_x']:.2f}m  "
        f"Y={player['court_y']:.2f}m  "
        f"conf={player['confidence']:.2f}"
    )

    text_y = max(
        30,
        y1 - 10
    )

    cv2.putText(
        frame,
        text,
        (
            x1,
            text_y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        colour,
        2
    )


# ============================================================
# DRAW PLAYER ON TOP-DOWN COURT
# ============================================================

def draw_player_on_court(
    court,
    player,
    label,
    colour
):

    if player is None:
        return

    court_x = (
        player["court_x"]
    )

    court_y = (
        player["court_y"]
    )

    display_x, display_y = (
        court_to_display(
            court_x,
            court_y
        )
    )

    # --------------------------------------------------------
    # Allow some room outside the baseline because players
    # often stand behind the regulation court.
    # --------------------------------------------------------

    if (
        -100
        <=
        display_x
        <=
        COURT_VIEW_WIDTH + 100
        and
        -150
        <=
        display_y
        <=
        COURT_VIEW_HEIGHT + 150
    ):

        cv2.circle(
            court,
            (
                display_x,
                display_y
            ),
            12,
            colour,
            -1
        )

        cv2.circle(
            court,
            (
                display_x,
                display_y
            ),
            16,
            (255, 255, 255),
            2
        )

        text = (
            f"{label} "
            f"({court_x:.2f}, "
            f"{court_y:.2f})"
        )

        cv2.putText(
            court,
            text,
            (
                display_x + 18,
                display_y - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2
        )


# ============================================================
# COMBINE VIDEO + COURT
# ============================================================

def combine_views(
    frame,
    court
):

    target_height = 720

    # --------------------------------------------------------
    # Resize video
    # --------------------------------------------------------

    frame_scale = (
        target_height /
        frame.shape[0]
    )

    frame_width = int(
        frame.shape[1] *
        frame_scale
    )

    frame_resized = cv2.resize(
        frame,
        (
            frame_width,
            target_height
        )
    )

    # --------------------------------------------------------
    # Resize top-down court
    # --------------------------------------------------------

    court_scale = (
        target_height /
        court.shape[0]
    )

    court_width = int(
        court.shape[1] *
        court_scale
    )

    court_resized = cv2.resize(
        court,
        (
            court_width,
            target_height
        )
    )

    return np.hstack(
        (
            frame_resized,
            court_resized
        )
    )


# ============================================================
# PROCESS ONE TEST SECTION
# ============================================================

def process_test(
    cap,
    fps,
    start_time,
    duration,
    test_number
):

    output_path = (
        OUTPUT_DIR /
        f"player_coordinate_validation_{test_number}.mp4"
    )

    print()
    print("=" * 70)

    print(
        f"TEST {test_number}"
    )

    print("=" * 70)

    print(
        f"Start time: "
        f"{start_time:.2f}s"
    )

    print(
        f"Duration: "
        f"{duration:.2f}s"
    )

    print(
        f"Output: "
        f"{output_path}"
    )

    # --------------------------------------------------------
    # Jump to requested point in video
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        start_time * 1000
    )

    frames_to_process = int(
        duration * fps
    )

    print(
        f"Frames to process: "
        f"{frames_to_process}"
    )

    writer = None

    processed = 0

    near_detected_count = 0
    far_detected_count = 0

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    while (
        processed
        <
        frames_to_process
    ):

        success, frame = (
            cap.read()
        )

        if not success:
            break

        current_time = (
            start_time +
            processed / fps
        )

        # ----------------------------------------------------
        # Detect players
        # ----------------------------------------------------

        near_player = (
            detect_near_player(
                frame
            )
        )

        far_player = (
            detect_far_player(
                frame
            )
        )

        if (
            near_player
            is not None
        ):
            near_detected_count += 1

        if (
            far_player
            is not None
        ):
            far_detected_count += 1

        # ----------------------------------------------------
        # Original-video display
        # ----------------------------------------------------

        display_frame = (
            frame.copy()
        )

        draw_player_on_frame(
            display_frame,
            near_player,
            "P1 NEAR",
            (0, 255, 255)
        )

        draw_player_on_frame(
            display_frame,
            far_player,
            "P2 FAR",
            (255, 100, 50)
        )

        # ----------------------------------------------------
        # Time + test number
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            (
                f"TEST {test_number}  "
                f"Time: {current_time:.2f}s"
            ),
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Top-down court
        # ----------------------------------------------------

        court = (
            draw_top_down_court()
        )

        draw_player_on_court(
            court,
            near_player,
            "P1",
            (0, 255, 255)
        )

        draw_player_on_court(
            court,
            far_player,
            "P2",
            (255, 100, 50)
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        combined = (
            combine_views(
                display_frame,
                court
            )
        )

        # ----------------------------------------------------
        # Create writer
        # ----------------------------------------------------

        if writer is None:

            output_height, output_width = (
                combined.shape[:2]
            )

            fourcc = (
                cv2.VideoWriter_fourcc(
                    *"mp4v"
                )
            )

            writer = (
                cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    fps,
                    (
                        output_width,
                        output_height
                    )
                )
            )

        writer.write(
            combined
        )

        processed += 1

        if (
            processed % 50 == 0
            or
            processed == frames_to_process
        ):

            print(
                f"Processed "
                f"{processed}/"
                f"{frames_to_process}"
            )

    # --------------------------------------------------------
    # Close writer
    # --------------------------------------------------------

    if writer is not None:
        writer.release()

    # --------------------------------------------------------
    # Detection statistics
    # --------------------------------------------------------

    if processed > 0:

        near_rate = (
            near_detected_count /
            processed *
            100
        )

        far_rate = (
            far_detected_count /
            processed *
            100
        )

    else:

        near_rate = 0
        far_rate = 0

    print()
    print(
        f"P1 near detection rate: "
        f"{near_rate:.1f}%"
    )

    print(
        f"P2 far detection rate: "
        f"{far_rate:.1f}%"
    )

    print(
        f"Saved: "
        f"{output_path}"
    )

    return {
        "test": test_number,
        "start_time": start_time,
        "processed": processed,
        "near_rate": near_rate,
        "far_rate": far_rate,
        "output": output_path
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{VIDEO_PATH}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:

        cap.release()

        raise RuntimeError(
            "Could not determine video FPS."
        )

    video_duration = (
        frame_count /
        fps
    )

    print()
    print("=" * 70)
    print("PLAYER COORDINATE VALIDATION")
    print("=" * 70)

    print(
        f"Video FPS: "
        f"{fps:.2f}"
    )

    print(
        f"Video duration: "
        f"{video_duration:.2f}s"
    )

    print(
        f"Number of tests: "
        f"{len(TEST_START_TIMES)}"
    )

    print(
        f"Duration per test: "
        f"{TEST_DURATION}s"
    )

    results = []

    # --------------------------------------------------------
    # Run all requested tests
    # --------------------------------------------------------

    for test_number, start_time in enumerate(
        TEST_START_TIMES,
        start=1
    ):

        if start_time >= video_duration:

            print()
            print(
                f"Skipping test "
                f"{test_number}: "
                f"{start_time}s is outside video."
            )

            continue

        remaining_time = (
            video_duration -
            start_time
        )

        duration = min(
            TEST_DURATION,
            remaining_time
        )

        result = process_test(
            cap,
            fps,
            start_time,
            duration,
            test_number
        )

        results.append(
            result
        )

    cap.release()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)

    for result in results:

        minutes = int(
            result["start_time"]
            // 60
        )

        seconds = int(
            result["start_time"]
            % 60
        )

        print()
        print(
            f"TEST {result['test']} "
            f"({minutes:02d}:{seconds:02d})"
        )

        print(
            f"  P1 near: "
            f"{result['near_rate']:.1f}%"
        )

        print(
            f"  P2 far:  "
            f"{result['far_rate']:.1f}%"
        )

        print(
            f"  File: "
            f"{result['output']}"
        )

    if results:

        average_near = np.mean(
            [
                result["near_rate"]
                for result in results
            ]
        )

        average_far = np.mean(
            [
                result["far_rate"]
                for result in results
            ]
        )

        print()
        print("-" * 70)

        print(
            f"AVERAGE P1 NEAR DETECTION: "
            f"{average_near:.1f}%"
        )

        print(
            f"AVERAGE P2 FAR DETECTION:  "
            f"{average_far:.1f}%"
        )

    print()
    print("=" * 70)
    print("FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()