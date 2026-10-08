import cv2
import numpy as np
import math


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "videos/test_match.mp4"
OUTPUT_PATH = "output/court_detector.jpg"
POINTS_OUTPUT = "data/auto_court_points.npy"

# Development benchmark ONLY.
# Never used to calculate the court.
MANUAL_PATH = "data/court_points_8.npy"

COURT_WIDTH = 8.23
COURT_LENGTH = 23.77

DOUBLES_WIDTH = 10.97
ALLEY_WIDTH = (DOUBLES_WIDTH - COURT_WIDTH) / 2.0

SERVICE_DISTANCE = 5.485


# ============================================================
# BASIC GEOMETRY
# ============================================================

def line_from_segment(segment):

    x1, y1, x2, y2 = segment

    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1

    norm = math.hypot(a, b)

    if norm < 1e-9:
        return None

    return np.array(
        [a / norm, b / norm, c / norm],
        dtype=np.float64
    )


def line_intersection(line1, line2):

    a1, b1, c1 = line1
    a2, b2, c2 = line2

    det = a1 * b2 - a2 * b1

    if abs(det) < 1e-9:
        return None

    x = (
        b1 * c2 - b2 * c1
    ) / det

    y = (
        c1 * a2 - c2 * a1
    ) / det

    return np.array(
        [x, y],
        dtype=np.float64
    )


def segment_info(segment):

    x1, y1, x2, y2 = segment

    dx = x2 - x1
    dy = y2 - y1

    length = math.hypot(dx, dy)

    angle = math.degrees(
        math.atan2(dy, dx)
    )

    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    return length, angle, mid_x, mid_y


def angle_difference(a, b):

    difference = abs(a - b)

    while difference >= 180:
        difference -= 180

    return min(
        difference,
        180 - difference
    )


# ============================================================
# ORANGE COURT MASK
# ============================================================

def find_court_mask(frame):

    height, width = frame.shape[:2]

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    orange = cv2.inRange(
        hsv,
        np.array([5, 70, 70]),
        np.array([35, 255, 255])
    )

    orange = cv2.morphologyEx(
        orange,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
        iterations=2
    )

    orange = cv2.morphologyEx(
        orange,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            orange,
            connectivity=8
        )
    )

    best_label = None
    best_score = -1

    for i in range(1, num_labels):

        y = stats[i, cv2.CC_STAT_TOP]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        bottom = y + h

        if area < 10000:
            continue

        if bottom < height * 0.55:
            continue

        score = area + bottom * 100

        if score > best_score:
            best_score = score
            best_label = i

    if best_label is None:
        raise RuntimeError(
            "Could not isolate playing surface"
        )

    mask = np.zeros_like(orange)

    mask[
        labels == best_label
    ] = 255

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    largest = max(
        contours,
        key=cv2.contourArea
    )

    mask[:] = 0

    cv2.drawContours(
        mask,
        [largest],
        -1,
        255,
        thickness=cv2.FILLED
    )

    return mask, hsv


# ============================================================
# COURT LINE SEGMENTS
# ============================================================

def find_line_segments(
    hsv,
    court_mask
):

    expanded = cv2.dilate(
        court_mask,
        np.ones((15, 15), np.uint8),
        iterations=1
    )

    white = cv2.inRange(
        hsv,
        np.array([0, 0, 120]),
        np.array([180, 120, 255])
    )

    white = cv2.bitwise_and(
        white,
        expanded
    )

    white = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8)
    )

    edges = cv2.Canny(
        white,
        50,
        150
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360,
        threshold=25,
        minLineLength=50,
        maxLineGap=40
    )

    if lines is None:
        raise RuntimeError(
            "No court lines detected"
        )

    return (
        lines.reshape(-1, 4),
        white,
        edges
    )


# ============================================================
# NEAR BASELINE
# ============================================================

def find_near_baseline(
    segments,
    height
):

    candidates = []

    for segment in segments:

        length, angle, mx, my = (
            segment_info(segment)
        )

        if my < height * 0.55:
            continue

        if not (-25 <= angle <= -3):
            continue

        if length < 150:
            continue

        score = (
            length
            +
            my * 0.5
        )

        candidates.append(
            (
                score,
                segment.copy(),
                angle
            )
        )

    if not candidates:
        raise RuntimeError(
            "Near baseline not found"
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return (
        candidates[0][1],
        candidates[0][2]
    )


# ============================================================
# BASELINE INTERSECTION CLUSTERS
# ============================================================

def baseline_intersections(
    segments,
    baseline_segment,
    baseline_angle,
    width,
    height
):

    baseline_line = line_from_segment(
        baseline_segment
    )

    candidates = []

    for segment in segments:

        length, angle, mx, my = (
            segment_info(segment)
        )

        if length < 80:
            continue

        diff = angle_difference(
            angle,
            baseline_angle
        )

        if diff < 8:
            continue

        line = line_from_segment(
            segment
        )

        if line is None:
            continue

        p = line_intersection(
            baseline_line,
            line
        )

        if p is None:
            continue

        x, y = p

        if not (
            -100 <= x <= width + 100
        ):
            continue

        if not (
            height * 0.45
            <= y
            <= height * 0.95
        ):
            continue

        candidates.append(
            {
                "point": p,
                "length": length,
                "segment": segment
            }
        )

    # --------------------------------------------
    # Cluster nearby intersections
    # --------------------------------------------

    candidates.sort(
        key=lambda item:
            item["point"][0]
    )

    clusters = []

    CLUSTER_DISTANCE = 12

    for candidate in candidates:

        placed = False

        for cluster in clusters:

            xs = [
                item["point"][0]
                for item in cluster
            ]

            weights = [
                item["length"]
                for item in cluster
            ]

            cluster_x = np.average(
                xs,
                weights=weights
            )

            if abs(
                candidate["point"][0]
                - cluster_x
            ) < CLUSTER_DISTANCE:

                cluster.append(candidate)
                placed = True
                break

        if not placed:
            clusters.append(
                [candidate]
            )

    results = []

    for cluster in clusters:

        weights = np.array(
            [
                item["length"]
                for item in cluster
            ]
        )

        points = np.array(
            [
                item["point"]
                for item in cluster
            ]
        )

        point = np.average(
            points,
            axis=0,
            weights=weights
        )

        strength = weights.sum()

        if (
            0 <= point[0] <= width
            and
            height * 0.50
            <= point[1]
            <= height * 0.90
        ):

            results.append(
                {
                    "point": point,
                    "strength": strength
                }
            )

    results.sort(
        key=lambda item:
            item["point"][0]
    )

    return results


# ============================================================
# IDENTIFY NEAR BASELINE LANDMARKS
#
# Current strategy:
#
# left-most cluster(s) = left doubles area
# inner strong cluster = left singles
# right-most cluster(s) = right doubles area
#
# We use tennis geometry to infer right singles.
# ============================================================

def find_near_corners(clusters):

    if len(clusters) < 3:
        raise RuntimeError(
            "Not enough baseline intersections"
        )

    # --------------------------------------------------------
    # LEFT DOUBLES
    #
    # There may be two Hough edges around the same painted
    # doubles line. Average the first two if they are close.
    # --------------------------------------------------------

    left_candidates = clusters[:2]

    if (
        len(left_candidates) >= 2
        and
        abs(
            left_candidates[1]["point"][0]
            -
            left_candidates[0]["point"][0]
        ) < 40
    ):

        left_doubles = np.average(
            np.array(
                [
                    left_candidates[0]["point"],
                    left_candidates[1]["point"]
                ]
            ),
            axis=0,
            weights=[
                left_candidates[0]["strength"],
                left_candidates[1]["strength"]
            ]
        )

        start_index = 2

    else:

        left_doubles = (
            left_candidates[0]["point"]
        )

        start_index = 1

    # --------------------------------------------------------
    # RIGHT DOUBLES
    #
    # Same idea at right edge.
    # --------------------------------------------------------

    right_candidates = clusters[-2:]

    if (
        len(right_candidates) >= 2
        and
        abs(
            right_candidates[1]["point"][0]
            -
            right_candidates[0]["point"][0]
        ) < 40
    ):

        right_doubles = np.average(
            np.array(
                [
                    right_candidates[0]["point"],
                    right_candidates[1]["point"]
                ]
            ),
            axis=0,
            weights=[
                right_candidates[0]["strength"],
                right_candidates[1]["strength"]
            ]
        )

        end_index = len(clusters) - 2

    else:

        right_doubles = (
            right_candidates[-1]["point"]
        )

        end_index = len(clusters) - 1

    # --------------------------------------------------------
    # LEFT SINGLES
    #
    # Look for strongest interior intersection.
    # --------------------------------------------------------

    interior = clusters[
        start_index:end_index
    ]

    if not interior:
        raise RuntimeError(
            "No interior singles intersection detected"
        )

    left_singles_cluster = max(
        interior,
        key=lambda item:
            item["strength"]
    )

    p7 = (
        left_singles_cluster["point"]
    )

    # --------------------------------------------------------
    # INFER P8 USING 1D PROJECTIVE GEOMETRY
    #
    # Known physical positions across doubles baseline:
    #
    # left doubles  = 0.00m
    # left singles  = 1.37m
    # right singles = 9.60m
    # right doubles = 10.97m
    # --------------------------------------------------------

    baseline_vector = (
        right_doubles
        -
        left_doubles
    )

    denominator = np.dot(
        baseline_vector,
        baseline_vector
    )

    if denominator < 1e-9:
        raise RuntimeError(
            "Invalid baseline geometry"
        )

    t_left_singles = np.dot(
        p7 - left_doubles,
        baseline_vector
    ) / denominator

    X2 = ALLEY_WIDTH
    T2 = t_left_singles

    X3 = DOUBLES_WIDTH
    T3 = 1.0

    A = np.array(
        [
            [
                X2,
                -T2 * X2
            ],
            [
                X3,
                -T3 * X3
            ]
        ],
        dtype=np.float64
    )

    B = np.array(
        [
            T2,
            T3
        ],
        dtype=np.float64
    )

    a, c = np.linalg.solve(
        A,
        B
    )

    X_RIGHT_SINGLES = (
        DOUBLES_WIDTH
        -
        ALLEY_WIDTH
    )

    t_right_singles = (
        a * X_RIGHT_SINGLES
    ) / (
        c * X_RIGHT_SINGLES
        + 1.0
    )

    p8 = (
        left_doubles
        +
        t_right_singles
        * baseline_vector
    )

    return (
        p7,
        p8,
        left_doubles,
        right_doubles
    )


# ============================================================
# VANISHING POINT
#
# Find intersections of likely longitudinal court segments.
# ============================================================

def find_sideline_vanishing_point(
    segments,
    width,
    height
):

    usable = []

    for segment in segments:

        length, angle, mx, my = (
            segment_info(segment)
        )

        if length < 70:
            continue

        # Broad family containing sidelines.
        if not (
            5 <= angle <= 35
        ):
            continue

        line = line_from_segment(
            segment
        )

        if line is None:
            continue

        usable.append(
            (
                line,
                length
            )
        )

    intersections = []

    for i in range(len(usable)):

        for j in range(
            i + 1,
            len(usable)
        ):

            p = line_intersection(
                usable[i][0],
                usable[j][0]
            )

            if p is None:
                continue

            x, y = p

            # Sideline VP for this camera is expected to
            # potentially sit outside the left of the image.
            # Keep this deliberately broad.

            if not (
                -width * 2
                <= x
                <= width * 1.5
            ):
                continue

            if not (
                -height
                <= y
                <= height * 1.5
            ):
                continue

            weight = (
                usable[i][1]
                *
                usable[j][1]
            )

            intersections.append(
                (
                    p,
                    weight
                )
            )

    if not intersections:
        raise RuntimeError(
            "Could not estimate sideline vanishing point"
        )

    # --------------------------------------------------------
    # RANSAC-like voting:
    # choose candidate with most nearby weighted support
    # --------------------------------------------------------

    best_point = None
    best_score = -1

    RADIUS = 80.0

    for candidate, _ in intersections:

        score = 0.0

        for point, weight in intersections:

            distance = np.linalg.norm(
                point - candidate
            )

            if distance < RADIUS:
                score += weight

        if score > best_score:

            best_score = score
            best_point = candidate

    # Refine by weighted average of nearby intersections.

    nearby_points = []
    nearby_weights = []

    for point, weight in intersections:

        if np.linalg.norm(
            point - best_point
        ) < RADIUS:

            nearby_points.append(point)
            nearby_weights.append(weight)

    vp = np.average(
        np.array(nearby_points),
        axis=0,
        weights=np.array(nearby_weights)
    )

    return vp


# ============================================================
# FAR BASELINE
# ============================================================

def find_far_baseline(
    segments,
    width,
    height
):

    candidates = []

    for segment in segments:

        length, angle, mx, my = (
            segment_info(segment)
        )

        # Far baseline is strongly foreshortened and close
        # to horizontal in this camera.

        if not (
            height * 0.34
            <= my
            <= height * 0.50
        ):
            continue

        if not (
            -5 <= angle <= 5
        ):
            continue

        if length < 120:
            continue

        score = (
            length
            -
            abs(angle) * 10
        )

        candidates.append(
            (
                score,
                segment.copy()
            )
        )

    if not candidates:
        raise RuntimeError(
            "Could not find far baseline"
        )

    candidates.sort(
        key=lambda item:
            item[0],
        reverse=True
    )

    return candidates[0][1]


# ============================================================
# GENERATE FULL SINGLES COURT
# ============================================================

def build_full_court(
    p1,
    p2,
    p7,
    p8
):

    world_outer = np.array(
        [
            [0.0, 0.0],
            [COURT_WIDTH, 0.0],
            [0.0, COURT_LENGTH],
            [COURT_WIDTH, COURT_LENGTH]
        ],
        dtype=np.float32
    )

    image_outer = np.array(
        [
            p1,
            p2,
            p7,
            p8
        ],
        dtype=np.float32
    )

    H = cv2.getPerspectiveTransform(
        world_outer,
        image_outer
    )

    world_points = np.array(
        [
            [0.0, 0.0],
            [COURT_WIDTH, 0.0],

            [0.0, SERVICE_DISTANCE],
            [COURT_WIDTH, SERVICE_DISTANCE],

            [
                0.0,
                COURT_LENGTH
                -
                SERVICE_DISTANCE
            ],

            [
                COURT_WIDTH,
                COURT_LENGTH
                -
                SERVICE_DISTANCE
            ],

            [0.0, COURT_LENGTH],
            [COURT_WIDTH, COURT_LENGTH]
        ],
        dtype=np.float32
    )

    projected = cv2.perspectiveTransform(
        world_points.reshape(
            -1,
            1,
            2
        ),
        H
    ).reshape(
        -1,
        2
    )

    return projected, H


# ============================================================
# COMPLETE DETECTOR
# ============================================================

def detect_court(frame):

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # Playing surface
    # --------------------------------------------------------

    court_mask, hsv = (
        find_court_mask(frame)
    )

    # --------------------------------------------------------
    # White lines
    # --------------------------------------------------------

    segments, white, edges = (
        find_line_segments(
            hsv,
            court_mask
        )
    )

    # --------------------------------------------------------
    # Near baseline
    # --------------------------------------------------------

    near_baseline_segment, near_angle = (
        find_near_baseline(
            segments,
            height
        )
    )

    # --------------------------------------------------------
    # Baseline intersections
    # --------------------------------------------------------

    clusters = baseline_intersections(
        segments,
        near_baseline_segment,
        near_angle,
        width,
        height
    )

    # --------------------------------------------------------
    # P7 + P8
    # --------------------------------------------------------

    (
        p7,
        p8,
        left_doubles,
        right_doubles
    ) = find_near_corners(
        clusters
    )

    # --------------------------------------------------------
    # Sideline vanishing point
    # --------------------------------------------------------

    vp = find_sideline_vanishing_point(
        segments,
        width,
        height
    )

    # --------------------------------------------------------
    # Far baseline
    # --------------------------------------------------------

    far_baseline_segment = (
        find_far_baseline(
            segments,
            width,
            height
        )
    )

    far_line = line_from_segment(
        far_baseline_segment
    )

    # --------------------------------------------------------
    # Lines from VP through P7/P8
    # --------------------------------------------------------

    left_sideline = (
        line_from_segment(
            [
                vp[0],
                vp[1],
                p7[0],
                p7[1]
            ]
        )
    )

    right_sideline = (
        line_from_segment(
            [
                vp[0],
                vp[1],
                p8[0],
                p8[1]
            ]
        )
    )

    # --------------------------------------------------------
    # P1 + P2
    # --------------------------------------------------------

    p1 = line_intersection(
        far_line,
        left_sideline
    )

    p2 = line_intersection(
        far_line,
        right_sideline
    )

    if p1 is None or p2 is None:
        raise RuntimeError(
            "Could not calculate far singles corners"
        )

    # --------------------------------------------------------
    # Full regulation singles court
    # --------------------------------------------------------

    points, H = build_full_court(
        p1,
        p2,
        p7,
        p8
    )

    return {
        "points": points,
        "homography": H,
        "vanishing_point": vp,
        "near_baseline": near_baseline_segment,
        "far_baseline": far_baseline_segment,
        "left_doubles": left_doubles,
        "right_doubles": right_doubles,
        "clusters": clusters,
        "court_mask": court_mask,
        "white_mask": white,
        "edges": edges
    }


# ============================================================
# DRAW RESULT
# ============================================================

def draw_result(
    frame,
    result
):

    output = frame.copy()

    points = result["points"]

    # Singles sidelines + cross-court lines

    lines_to_draw = [
        (0, 2),
        (2, 4),
        (4, 6),

        (1, 3),
        (3, 5),
        (5, 7),

        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7)
    ]

    for a, b in lines_to_draw:

        pa = tuple(
            np.round(
                points[a]
            ).astype(int)
        )

        pb = tuple(
            np.round(
                points[b]
            ).astype(int)
        )

        cv2.line(
            output,
            pa,
            pb,
            (255, 0, 255),
            3
        )

    # Points

    for i, point in enumerate(
        points,
        start=1
    ):

        x = int(round(point[0]))
        y = int(round(point[1]))

        cv2.circle(
            output,
            (x, y),
            8,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            output,
            str(i),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    # VP marker if visible-ish

    vp = result["vanishing_point"]

    print(
        f"Vanishing point: "
        f"({vp[0]:.1f}, {vp[1]:.1f})"
    )

    return output


# ============================================================
# RUN TEST
# ============================================================

if __name__ == "__main__":

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        60 * 1000
    )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError(
            "Could not read video frame"
        )

    print()
    print("ZERO-AI COURT DETECTOR")
    print("=" * 70)

    result = detect_court(
        frame
    )

    points = result["points"]

    print()
    print("AUTOMATIC COURT POINTS")
    print("=" * 70)

    for i, point in enumerate(
        points,
        start=1
    ):

        print(
            f"P{i}: "
            f"({point[0]:.1f}, "
            f"{point[1]:.1f})"
        )

    print()
    print(
        f"Left doubles: "
        f"{result['left_doubles']}"
    )

    print(
        f"Right doubles: "
        f"{result['right_doubles']}"
    )

    print(
        f"Vanishing point: "
        f"{result['vanishing_point']}"
    )

    print(
        f"Far baseline segment: "
        f"{result['far_baseline']}"
    )

    # --------------------------------------------------------
    # Save points
    # --------------------------------------------------------

    np.save(
        POINTS_OUTPUT,
        points
    )

    # --------------------------------------------------------
    # Draw
    # --------------------------------------------------------

    output = draw_result(
        frame,
        result
    )

    cv2.imwrite(
        OUTPUT_PATH,
        output
    )

    # --------------------------------------------------------
    # DEVELOPMENT BENCHMARK
    #
    # IMPORTANT:
    # manual points have NOT been used anywhere in detection.
    # --------------------------------------------------------

    try:

        manual = np.load(
            MANUAL_PATH
        ).astype(np.float32)

        errors = np.linalg.norm(
            points - manual,
            axis=1
        )

        print()
        print("HIDDEN MANUAL BENCHMARK")
        print("=" * 70)

        for i in range(8):

            print(
                f"P{i + 1}: "
                f"AUTO=("
                f"{points[i][0]:.1f}, "
                f"{points[i][1]:.1f}) | "
                f"MANUAL=("
                f"{manual[i][0]:.1f}, "
                f"{manual[i][1]:.1f}) | "
                f"ERROR={errors[i]:.1f}px"
            )

        print()
        print(
            f"Mean error: "
            f"{errors.mean():.1f}px"
        )

        print(
            f"Max error: "
            f"{errors.max():.1f}px"
        )

    except FileNotFoundError:

        pass

    print()
    print(
        f"Saved points: "
        f"{POINTS_OUTPUT}"
    )

    print(
        f"Saved image: "
        f"{OUTPUT_PATH}"
    )