import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import math


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "videos/test_match.mp4"
MODEL_PATH = "yolo11n.pt"

OUTPUT_DIR = Path("output")


# ============================================================
# TEST SECTIONS
#
# Same four sections used in the coordinate validation.
#
# 60  = 01:00
# 180 = 03:00
# 360 = 06:00
# 600 = 10:00
# ============================================================

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
# TRACKING SETTINGS
# ============================================================

# Number of frames we allow a player to be missing before
# declaring the track genuinely lost.
MAX_MISSED_FRAMES = 25

# Maximum normalised centre movement allowed between detections.
#
# This is deliberately generous because tennis players move fast.
MAX_DISTANCE_RATIO = 0.18

# If matching becomes uncertain, reject the candidate rather
# than switching player identity.
MAX_MATCH_SCORE = 2.20

# Detection must have at least this height.
MIN_BOX_HEIGHT = 18

# Ignore enormous boxes that are clearly not useful players.
MAX_BOX_HEIGHT_RATIO = 0.75


# ============================================================
# DISPLAY COLOURS
#
# OpenCV uses BGR.
# ============================================================

P1_COLOUR = (0, 255, 255)       # yellow
P2_COLOUR = (255, 100, 50)      # blue/orange-ish

DETECTED_COLOUR = (0, 255, 0)
PREDICTED_COLOUR = (0, 165, 255)
LOST_COLOUR = (0, 0, 255)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# ============================================================
# TRACK CLASS
# ============================================================

class PlayerTrack:

    def __init__(
        self,
        player_id,
        colour
    ):

        self.player_id = player_id
        self.colour = colour

        self.initialized = False

        self.box = None

        self.center_x = None
        self.center_y = None

        self.previous_center_x = None
        self.previous_center_y = None

        self.velocity_x = 0.0
        self.velocity_y = 0.0

        self.box_width = None
        self.box_height = None

        self.confidence = None

        self.missed_frames = 0

        self.status = "UNINITIALIZED"

        self.detected_frames = 0
        self.predicted_frames = 0
        self.lost_frames = 0

        self.total_frames = 0


    # ========================================================
    # UPDATE FROM REAL YOLO DETECTION
    # ========================================================

    def update_detection(
        self,
        detection
    ):

        new_x = detection["center_x"]
        new_y = detection["center_y"]

        if self.initialized:

            self.previous_center_x = (
                self.center_x
            )

            self.previous_center_y = (
                self.center_y
            )

            measured_velocity_x = (
                new_x -
                self.center_x
            )

            measured_velocity_y = (
                new_y -
                self.center_y
            )

            # Smooth velocity slightly.
            self.velocity_x = (
                0.65 * self.velocity_x
                +
                0.35 * measured_velocity_x
            )

            self.velocity_y = (
                0.65 * self.velocity_y
                +
                0.35 * measured_velocity_y
            )

        else:

            self.previous_center_x = new_x
            self.previous_center_y = new_y

            self.velocity_x = 0.0
            self.velocity_y = 0.0

        self.center_x = new_x
        self.center_y = new_y

        self.box = (
            detection["x1"],
            detection["y1"],
            detection["x2"],
            detection["y2"]
        )

        self.box_width = (
            detection["width"]
        )

        self.box_height = (
            detection["height"]
        )

        self.confidence = (
            detection["confidence"]
        )

        self.missed_frames = 0

        self.initialized = True

        self.status = "DETECTED"

        self.detected_frames += 1
        self.total_frames += 1


    # ========================================================
    # UPDATE WHEN YOLO TEMPORARILY MISSES PLAYER
    # ========================================================

    def update_missing(
        self,
        frame_width,
        frame_height
    ):

        self.total_frames += 1

        if not self.initialized:

            self.status = "LOST"
            self.lost_frames += 1
            return

        self.missed_frames += 1

        if (
            self.missed_frames
            <=
            MAX_MISSED_FRAMES
        ):

            # Predict using recent velocity.
            self.center_x += (
                self.velocity_x
            )

            self.center_y += (
                self.velocity_y
            )

            # Gradually reduce velocity during prediction.
            self.velocity_x *= 0.92
            self.velocity_y *= 0.92

            # Keep prediction within image.
            self.center_x = max(
                0,
                min(
                    frame_width - 1,
                    self.center_x
                )
            )

            self.center_y = max(
                0,
                min(
                    frame_height - 1,
                    self.center_y
                )
            )

            if (
                self.box_width is not None
                and
                self.box_height is not None
            ):

                half_w = (
                    self.box_width / 2
                )

                half_h = (
                    self.box_height / 2
                )

                self.box = (
                    self.center_x - half_w,
                    self.center_y - half_h,
                    self.center_x + half_w,
                    self.center_y + half_h
                )

            self.status = "PREDICTED"

            self.predicted_frames += 1

        else:

            self.status = "LOST"

            self.lost_frames += 1


    # ========================================================
    # PREDICT NEXT CENTRE
    # ========================================================

    def predicted_center(self):

        if not self.initialized:

            return None

        return (
            self.center_x + self.velocity_x,
            self.center_y + self.velocity_y
        )


# ============================================================
# YOLO PERSON DETECTION
# ============================================================

def detect_people(
    model,
    frame
):

    frame_height, frame_width = (
        frame.shape[:2]
    )

    results = model.predict(
        frame,
        classes=[0],
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False
    )[0]

    detections = []

    for box in results.boxes:

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        confidence = float(
            box.conf[0]
        )

        width = (
            x2 - x1
        )

        height = (
            y2 - y1
        )

        if height < MIN_BOX_HEIGHT:
            continue

        if (
            height
            >
            frame_height
            *
            MAX_BOX_HEIGHT_RATIO
        ):
            continue

        center_x = (
            x1 + x2
        ) / 2

        center_y = (
            y1 + y2
        ) / 2

        detections.append(
            {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),

                "width": float(width),
                "height": float(height),

                "center_x": float(center_x),
                "center_y": float(center_y),

                "confidence": confidence
            }
        )

    return detections


# ============================================================
# INITIAL PLAYER SELECTION
# ============================================================

def initialize_players(
    detections,
    frame_width,
    frame_height
):

    """
    Select the two most plausible tennis players.

    IMPORTANT:

    This logic is ONLY used to initialise identity.

    Once initialized, identity is maintained temporally.

    For this fixed camera:
      - far player appears smaller / higher in image
      - near player appears larger / lower in image

    We do NOT use these rules every frame.
    """

    if len(detections) < 2:
        return None, None

    candidates = []

    for detection in detections:

        cx = detection["center_x"]
        cy = detection["center_y"]

        h = detection["height"]

        # Ignore people very high in image.
        if cy < frame_height * 0.20:
            continue

        # Ignore extreme image edges.
        if cx < frame_width * 0.02:
            continue

        if cx > frame_width * 0.98:
            continue

        candidates.append(
            detection
        )

    if len(candidates) < 2:
        return None, None

    # --------------------------------------------------------
    # Near player:
    #
    # Combination of:
    #   lower in image
    #   larger bounding box
    # --------------------------------------------------------

    near_scores = []

    for detection in candidates:

        vertical_score = (
            detection["center_y"]
            /
            frame_height
        )

        size_score = (
            detection["height"]
            /
            frame_height
        )

        score = (
            vertical_score
            +
            size_score * 1.5
        )

        near_scores.append(
            (
                score,
                detection
            )
        )

    near_scores.sort(
        key=lambda item:
            item[0],
        reverse=True
    )

    near_player = (
        near_scores[0][1]
    )

    # --------------------------------------------------------
    # Far player:
    #
    # Remove near player first.
    #
    # Prefer people:
    #   higher in image
    #   smaller than near player
    #   reasonably central to court image
    # --------------------------------------------------------

    far_candidates = []

    for detection in candidates:

        if detection is near_player:
            continue

        # Far player should normally be smaller
        # than the initialized near player.
        if (
            detection["height"]
            >
            near_player["height"] * 0.85
        ):
            continue

        vertical_score = (
            detection["center_y"]
            /
            frame_height
        )

        size_score = (
            detection["height"]
            /
            frame_height
        )

        # Lower score = more likely far player.
        score = (
            vertical_score
            +
            size_score * 0.75
        )

        far_candidates.append(
            (
                score,
                detection
            )
        )

    if not far_candidates:
        return None, None

    far_candidates.sort(
        key=lambda item:
            item[0]
    )

    far_player = (
        far_candidates[0][1]
    )

    return (
        near_player,
        far_player
    )


# ============================================================
# MATCH SCORE
# ============================================================

def calculate_match_score(
    track,
    detection,
    frame_width,
    frame_height
):

    predicted = (
        track.predicted_center()
    )

    if predicted is None:
        return float("inf")

    predicted_x, predicted_y = (
        predicted
    )

    dx = (
        detection["center_x"]
        -
        predicted_x
    )

    dy = (
        detection["center_y"]
        -
        predicted_y
    )

    distance = math.sqrt(
        dx * dx +
        dy * dy
    )

    diagonal = math.sqrt(
        frame_width * frame_width
        +
        frame_height * frame_height
    )

    distance_ratio = (
        distance /
        diagonal
    )

    # --------------------------------------------------------
    # Reject impossible jumps.
    # --------------------------------------------------------

    if (
        distance_ratio
        >
        MAX_DISTANCE_RATIO
    ):

        return float("inf")

    # --------------------------------------------------------
    # Size consistency
    # --------------------------------------------------------

    if (
        track.box_height is not None
        and
        track.box_height > 0
    ):

        height_ratio = (
            detection["height"]
            /
            track.box_height
        )

        height_penalty = abs(
            math.log(
                max(
                    height_ratio,
                    0.01
                )
            )
        )

    else:

        height_penalty = 0.0

    if (
        track.box_width is not None
        and
        track.box_width > 0
    ):

        width_ratio = (
            detection["width"]
            /
            track.box_width
        )

        width_penalty = abs(
            math.log(
                max(
                    width_ratio,
                    0.01
                )
            )
        )

    else:

        width_penalty = 0.0

    # --------------------------------------------------------
    # Confidence bonus
    # --------------------------------------------------------

    confidence_bonus = (
        detection["confidence"]
        *
        0.20
    )

    # --------------------------------------------------------
    # Final score
    #
    # LOWER = BETTER
    # --------------------------------------------------------

    score = (
        distance_ratio * 8.0
        +
        height_penalty * 0.65
        +
        width_penalty * 0.25
        -
        confidence_bonus
    )

    return score


# ============================================================
# ASSIGN DETECTIONS TO TRACKS
# ============================================================

def assign_detections(
    p1_track,
    p2_track,
    detections,
    frame_width,
    frame_height
):

    """
    Jointly assign detections to P1 and P2.

    A single detection can NEVER be assigned to both players.
    """

    if len(detections) == 0:

        return None, None

    # --------------------------------------------------------
    # Build score table
    # --------------------------------------------------------

    p1_scores = []

    p2_scores = []

    for index, detection in enumerate(
        detections
    ):

        score_p1 = (
            calculate_match_score(
                p1_track,
                detection,
                frame_width,
                frame_height
            )
        )

        score_p2 = (
            calculate_match_score(
                p2_track,
                detection,
                frame_width,
                frame_height
            )
        )

        p1_scores.append(
            (
                score_p1,
                index
            )
        )

        p2_scores.append(
            (
                score_p2,
                index
            )
        )

    # --------------------------------------------------------
    # Search all possible assignment combinations.
    #
    # This is small because YOLO normally only finds a handful
    # of people.
    # --------------------------------------------------------

    best_total_score = (
        float("inf")
    )

    best_p1_index = None
    best_p2_index = None

    p1_options = [
        None
    ]

    p2_options = [
        None
    ]

    for score, index in p1_scores:

        if (
            math.isfinite(score)
            and
            score <= MAX_MATCH_SCORE
        ):

            p1_options.append(
                index
            )

    for score, index in p2_scores:

        if (
            math.isfinite(score)
            and
            score <= MAX_MATCH_SCORE
        ):

            p2_options.append(
                index
            )

    # Penalty for leaving a track unmatched.
    unmatched_penalty = 1.80

    for p1_index in p1_options:

        for p2_index in p2_options:

            if (
                p1_index is not None
                and
                p2_index is not None
                and
                p1_index == p2_index
            ):
                continue

            if p1_index is None:

                p1_score = (
                    unmatched_penalty
                )

            else:

                p1_score = (
                    calculate_match_score(
                        p1_track,
                        detections[p1_index],
                        frame_width,
                        frame_height
                    )
                )

            if p2_index is None:

                p2_score = (
                    unmatched_penalty
                )

            else:

                p2_score = (
                    calculate_match_score(
                        p2_track,
                        detections[p2_index],
                        frame_width,
                        frame_height
                    )
                )

            total_score = (
                p1_score +
                p2_score
            )

            if (
                total_score
                <
                best_total_score
            ):

                best_total_score = (
                    total_score
                )

                best_p1_index = (
                    p1_index
                )

                best_p2_index = (
                    p2_index
                )

    if best_p1_index is None:

        p1_detection = None

    else:

        p1_detection = (
            detections[
                best_p1_index
            ]
        )

    if best_p2_index is None:

        p2_detection = None

    else:

        p2_detection = (
            detections[
                best_p2_index
            ]
        )

    return (
        p1_detection,
        p2_detection
    )


# ============================================================
# DRAW TRACK
# ============================================================

def draw_track(
    frame,
    track
):

    if (
        not track.initialized
        or
        track.box is None
    ):

        return

    if track.status == "LOST":
        return

    x1, y1, x2, y2 = (
        track.box
    )

    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    # --------------------------------------------------------
    # Solid box = actual YOLO detection
    #
    # Thinner box = prediction
    # --------------------------------------------------------

    if track.status == "DETECTED":

        thickness = 3

    else:

        thickness = 1

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        track.colour,
        thickness
    )

    # --------------------------------------------------------
    # Centre point
    # --------------------------------------------------------

    cx = int(
        round(
            track.center_x
        )
    )

    cy = int(
        round(
            track.center_y
        )
    )

    cv2.circle(
        frame,
        (cx, cy),
        6,
        track.colour,
        -1
    )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    if track.status == "DETECTED":

        status_text = (
            "DETECTED"
        )

    elif track.status == "PREDICTED":

        status_text = (
            f"PREDICTED "
            f"{track.missed_frames}"
        )

    else:

        status_text = (
            "LOST"
        )

    if track.confidence is not None:

        confidence_text = (
            f"{track.confidence:.2f}"
        )

    else:

        confidence_text = "--"

    text = (
        f"{track.player_id}  "
        f"{status_text}  "
        f"conf={confidence_text}"
    )

    text_y = max(
        25,
        y1 - 10
    )

    cv2.putText(
        frame,
        text,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        track.colour,
        2
    )


# ============================================================
# DRAW STATUS PANEL
# ============================================================

def draw_status_panel(
    frame,
    p1_track,
    p2_track,
    test_number,
    current_time
):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (15, 15),
        (620, 125),
        BLACK,
        -1
    )

    cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0,
        frame
    )

    cv2.putText(
        frame,
        (
            f"TEST {test_number}   "
            f"TIME {current_time:.2f}s"
        ),
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        WHITE,
        2
    )

    cv2.putText(
        frame,
        (
            f"P1: {p1_track.status}   "
            f"missed={p1_track.missed_frames}"
        ),
        (30, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        P1_COLOUR,
        2
    )

    cv2.putText(
        frame,
        (
            f"P2: {p2_track.status}   "
            f"missed={p2_track.missed_frames}"
        ),
        (30, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        P2_COLOUR,
        2
    )


# ============================================================
# PROCESS ONE TEST
# ============================================================

def process_test(
    model,
    cap,
    fps,
    frame_width,
    frame_height,
    start_time,
    duration,
    test_number
):

    output_path = (
        OUTPUT_DIR
        /
        f"identity_tracking_{test_number}.mp4"
    )

    print()
    print("=" * 70)
    print(
        f"TEST {test_number}"
    )
    print("=" * 70)

    print(
        f"Start: {start_time:.2f}s"
    )

    print(
        f"Duration: {duration:.2f}s"
    )

    print(
        f"Output: {output_path}"
    )

    start_frame = int(
        start_time * fps
    )

    frames_to_process = int(
        duration * fps
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            frame_width,
            frame_height
        )
    )

    if not writer.isOpened():

        raise RuntimeError(
            f"Could not create "
            f"{output_path}"
        )

    # --------------------------------------------------------
    # Create persistent tracks.
    #
    # P1 = near player at initialization
    # P2 = far player at initialization
    # --------------------------------------------------------

    p1_track = PlayerTrack(
        "P1",
        P1_COLOUR
    )

    p2_track = PlayerTrack(
        "P2",
        P2_COLOUR
    )

    initialized = False

    initialization_frame = None

    processed = 0

    # ========================================================
    # FRAME LOOP
    # ========================================================

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
            start_time
            +
            processed / fps
        )

        detections = detect_people(
            model,
            frame
        )

        # ====================================================
        # INITIALIZATION
        # ====================================================

        if not initialized:

            near_detection, far_detection = (
                initialize_players(
                    detections,
                    frame_width,
                    frame_height
                )
            )

            if (
                near_detection is not None
                and
                far_detection is not None
            ):

                p1_track.update_detection(
                    near_detection
                )

                p2_track.update_detection(
                    far_detection
                )

                initialized = True

                initialization_frame = (
                    processed
                )

                print(
                    f"Players initialized at "
                    f"{current_time:.2f}s"
                )

            else:

                # Not enough information yet.
                p1_track.total_frames += 1
                p2_track.total_frames += 1

                p1_track.status = (
                    "WAITING"
                )

                p2_track.status = (
                    "WAITING"
                )

        # ====================================================
        # NORMAL TRACKING
        # ====================================================

        else:

            (
                p1_detection,
                p2_detection
            ) = assign_detections(
                p1_track,
                p2_track,
                detections,
                frame_width,
                frame_height
            )

            # ------------------------------------------------
            # P1
            # ------------------------------------------------

            if p1_detection is not None:

                p1_track.update_detection(
                    p1_detection
                )

            else:

                p1_track.update_missing(
                    frame_width,
                    frame_height
                )

            # ------------------------------------------------
            # P2
            # ------------------------------------------------

            if p2_detection is not None:

                p2_track.update_detection(
                    p2_detection
                )

            else:

                p2_track.update_missing(
                    frame_width,
                    frame_height
                )

        # ====================================================
        # DRAW
        # ====================================================

        draw_track(
            frame,
            p1_track
        )

        draw_track(
            frame,
            p2_track
        )

        draw_status_panel(
            frame,
            p1_track,
            p2_track,
            test_number,
            current_time
        )

        # ----------------------------------------------------
        # Initialization warning
        # ----------------------------------------------------

        if not initialized:

            cv2.putText(
                frame,
                "WAITING TO INITIALIZE BOTH PLAYERS",
                (
                    30,
                    frame_height - 35
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        writer.write(
            frame
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

    writer.release()

    # ========================================================
    # STATISTICS
    # ========================================================

    print()
    print("TRACK RESULTS")
    print("-" * 70)

    if initialization_frame is None:

        print(
            "Players were never initialized."
        )

    else:

        print(
            f"Initialized at local frame: "
            f"{initialization_frame}"
        )

    def calculate_stats(
        track
    ):

        total = max(
            track.total_frames,
            1
        )

        detected_percent = (
            100.0
            *
            track.detected_frames
            /
            total
        )

        predicted_percent = (
            100.0
            *
            track.predicted_frames
            /
            total
        )

        lost_percent = (
            100.0
            *
            track.lost_frames
            /
            total
        )

        maintained_percent = (
            100.0
            *
            (
                track.detected_frames
                +
                track.predicted_frames
            )
            /
            total
        )

        return {
            "detected":
                detected_percent,

            "predicted":
                predicted_percent,

            "lost":
                lost_percent,

            "maintained":
                maintained_percent
        }

    p1_stats = (
        calculate_stats(
            p1_track
        )
    )

    p2_stats = (
        calculate_stats(
            p2_track
        )
    )

    print()
    print("P1")

    print(
        f"  YOLO detected: "
        f"{p1_stats['detected']:.1f}%"
    )

    print(
        f"  Predicted:     "
        f"{p1_stats['predicted']:.1f}%"
    )

    print(
        f"  Lost:          "
        f"{p1_stats['lost']:.1f}%"
    )

    print(
        f"  Track present: "
        f"{p1_stats['maintained']:.1f}%"
    )

    print()
    print("P2")

    print(
        f"  YOLO detected: "
        f"{p2_stats['detected']:.1f}%"
    )

    print(
        f"  Predicted:     "
        f"{p2_stats['predicted']:.1f}%"
    )

    print(
        f"  Lost:          "
        f"{p2_stats['lost']:.1f}%"
    )

    print(
        f"  Track present: "
        f"{p2_stats['maintained']:.1f}%"
    )

    print()
    print(
        f"Saved: {output_path}"
    )

    return {
        "test": test_number,
        "start_time": start_time,
        "p1": p1_stats,
        "p2": p2_stats,
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

    print()
    print("=" * 70)
    print("TENNIS PLAYER IDENTITY TRACKING TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

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

    frame_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:

        cap.release()

        raise RuntimeError(
            "Could not determine FPS."
        )

    video_duration = (
        frame_count /
        fps
    )

    print(
        f"Video: {VIDEO_PATH}"
    )

    print(
        f"Resolution: "
        f"{frame_width}x"
        f"{frame_height}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Duration: "
        f"{video_duration:.2f}s"
    )

    # --------------------------------------------------------
    # Load YOLO once
    # --------------------------------------------------------

    print()
    print("Loading YOLO...")

    model = YOLO(
        MODEL_PATH
    )

    print("YOLO loaded.")

    results = []

    # ========================================================
    # RUN FOUR TESTS
    # ========================================================

    for test_number, start_time in enumerate(
        TEST_START_TIMES,
        start=1
    ):

        if (
            start_time
            >=
            video_duration
        ):

            print(
                f"Skipping test "
                f"{test_number}: "
                f"outside video."
            )

            continue

        remaining = (
            video_duration
            -
            start_time
        )

        duration = min(
            TEST_DURATION,
            remaining
        )

        result = process_test(
            model,
            cap,
            fps,
            frame_width,
            frame_height,
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
    print("FINAL IDENTITY TRACKING SUMMARY")
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
            f"  P1 detected: "
            f"{result['p1']['detected']:.1f}%"
        )

        print(
            f"  P1 predicted: "
            f"{result['p1']['predicted']:.1f}%"
        )

        print(
            f"  P1 present: "
            f"{result['p1']['maintained']:.1f}%"
        )

        print(
            f"  P2 detected: "
            f"{result['p2']['detected']:.1f}%"
        )

        print(
            f"  P2 predicted: "
            f"{result['p2']['predicted']:.1f}%"
        )

        print(
            f"  P2 present: "
            f"{result['p2']['maintained']:.1f}%"
        )

        print(
            f"  File: "
            f"{result['output']}"
        )

    print()
    print("=" * 70)
    print("IMPORTANT")
    print("=" * 70)

    print(
        "High percentages do NOT prove identity accuracy."
    )

    print(
        "Watch the videos and check that P1 and P2 "
        "never jump to another person."
    )

    print()
    print("FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()