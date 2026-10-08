import cv2
import numpy as np


# ============================================================
# FILES
# ============================================================

VIDEO_PATH = "videos/test_match.mp4"

AI_POINTS_PATH = "data/ai_court_points_v2.npy"
MANUAL_POINTS_PATH = "data/court_points_8.npy"

OUTPUT_PATH = "output/test_far_corners_geometry.jpg"


# ============================================================
# AUTOMATIC RESULTS FROM OUR PREVIOUS CV TESTS
#
# These are NOT manual court labels.
#
# VP = automatically estimated sideline vanishing point
# FAR_BASELINE_Y = automatically detected far baseline
# ============================================================

VP = np.array(
    [-287.17292619, 337.27677988],
    dtype=np.float64
)

FAR_BASELINE_Y = 447.0


# ============================================================
# LOAD FRAME
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

cap.set(
    cv2.CAP_PROP_POS_MSEC,
    60 * 1000
)

success, frame = cap.read()

cap.release()

if not success:
    raise RuntimeError("Could not read video frame")


# ============================================================
# LOAD AI POINTS
# ============================================================

ai_points = np.load(
    AI_POINTS_PATH
).astype(np.float64)


# Python indexing:
#
# point 7 = index 6
# point 8 = index 7

p7 = ai_points[6]
p8 = ai_points[7]


print()
print("INPUT")
print("=" * 60)

print(f"Vanishing point: {VP}")
print(f"Far baseline y: {FAR_BASELINE_Y:.1f}")

print(f"Point 7: {p7}")
print(f"Point 8: {p8}")


# ============================================================
# INTERSECT A LINE WITH HORIZONTAL FAR BASELINE
#
# The line runs:
#
# vanishing point -> near baseline corner
#
# and we find where it reaches y = 447.
# ============================================================

def intersect_with_horizontal_line(
    vanishing_point,
    point,
    target_y
):

    vx, vy = vanishing_point
    px, py = point

    if abs(py - vy) < 1e-9:
        raise RuntimeError(
            "Cannot intersect horizontal line."
        )

    t = (
        target_y - vy
    ) / (
        py - vy
    )

    x = vx + t * (px - vx)

    return np.array(
        [x, target_y],
        dtype=np.float64
    )


# ============================================================
# CALCULATE FAR CORNERS
# ============================================================

p1 = intersect_with_horizontal_line(
    VP,
    p7,
    FAR_BASELINE_Y
)

p2 = intersect_with_horizontal_line(
    VP,
    p8,
    FAR_BASELINE_Y
)


print()
print("CALCULATED FAR CORNERS")
print("=" * 60)

print(
    f"Point 1 = "
    f"({p1[0]:.1f}, {p1[1]:.1f})"
)

print(
    f"Point 2 = "
    f"({p2[0]:.1f}, {p2[1]:.1f})"
)


# ============================================================
# DRAW
# ============================================================

output = frame.copy()


# ------------------------------------------------------------
# Draw vanishing-point-to-corner lines.
#
# VP is outside the image, but OpenCV can still draw the line.
# ------------------------------------------------------------

vp_int = tuple(
    np.round(VP).astype(int)
)

p7_int = tuple(
    np.round(p7).astype(int)
)

p8_int = tuple(
    np.round(p8).astype(int)
)

p1_int = tuple(
    np.round(p1).astype(int)
)

p2_int = tuple(
    np.round(p2).astype(int)
)


# Sideline 1 -> 7
cv2.line(
    output,
    vp_int,
    p7_int,
    (255, 0, 255),
    3
)

# Sideline 2 -> 8
cv2.line(
    output,
    vp_int,
    p8_int,
    (255, 0, 255),
    3
)


# Far baseline
cv2.line(
    output,
    p1_int,
    p2_int,
    (0, 255, 255),
    4
)


# Near baseline
cv2.line(
    output,
    p7_int,
    p8_int,
    (0, 255, 255),
    3
)


# ============================================================
# DRAW POINTS
# ============================================================

points_to_draw = [
    (1, p1),
    (2, p2),
    (7, p7),
    (8, p8)
]


for number, point in points_to_draw:

    x = int(round(point[0]))
    y = int(round(point[1]))

    cv2.circle(
        output,
        (x, y),
        9,
        (0, 255, 255),
        -1
    )

    cv2.putText(
        output,
        str(number),
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


# ============================================================
# HIDDEN MANUAL BENCHMARK
#
# Manual points are used ONLY here for measuring accuracy.
# They do not affect the calculation above.
# ============================================================

manual = np.load(
    MANUAL_POINTS_PATH
).astype(np.float64)

manual_p1 = manual[0]
manual_p2 = manual[1]


error1 = np.linalg.norm(
    p1 - manual_p1
)

error2 = np.linalg.norm(
    p2 - manual_p2
)


print()
print("HIDDEN MANUAL BENCHMARK")
print("=" * 60)

print(
    f"Point 1:"
    f" AUTO=({p1[0]:.1f}, {p1[1]:.1f})"
    f" MANUAL=({manual_p1[0]:.1f}, {manual_p1[1]:.1f})"
    f" ERROR={error1:.1f}px"
)

print(
    f"Point 2:"
    f" AUTO=({p2[0]:.1f}, {p2[1]:.1f})"
    f" MANUAL=({manual_p2[0]:.1f}, {manual_p2[1]:.1f})"
    f" ERROR={error2:.1f}px"
)


print()
print(
    f"Mean error: "
    f"{(error1 + error2) / 2:.1f}px"
)


# ============================================================
# SAVE
# ============================================================

cv2.imwrite(
    OUTPUT_PATH,
    output
)

print()
print(f"Saved: {OUTPUT_PATH}")