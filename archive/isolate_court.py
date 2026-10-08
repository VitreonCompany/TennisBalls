import cv2
import numpy as np

VIDEO_PATH = "videos/test_match.mp4"

OUTPUT_MASK = "output/orange_court_mask.jpg"
OUTPUT_OVERLAY = "output/orange_court_overlay.jpg"

# --------------------------------------------------
# 1. Load our test frame
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, 60 * 1000)

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read frame")

height, width = frame.shape[:2]

print(f"Frame size: {width} x {height}")


# --------------------------------------------------
# 2. Convert image to HSV
# --------------------------------------------------

hsv = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2HSV
)


# --------------------------------------------------
# 3. Find orange regions
#
# This range is intentionally broad.
# --------------------------------------------------

lower_orange = np.array(
    [5, 70, 70],
    dtype=np.uint8
)

upper_orange = np.array(
    [35, 255, 255],
    dtype=np.uint8
)

orange = cv2.inRange(
    hsv,
    lower_orange,
    upper_orange
)


# --------------------------------------------------
# 4. Clean small gaps/noise
# --------------------------------------------------

large_kernel = np.ones(
    (7, 7),
    np.uint8
)

orange = cv2.morphologyEx(
    orange,
    cv2.MORPH_CLOSE,
    large_kernel,
    iterations=2
)

small_kernel = np.ones(
    (3, 3),
    np.uint8
)

orange = cv2.morphologyEx(
    orange,
    cv2.MORPH_OPEN,
    small_kernel
)


# --------------------------------------------------
# 5. Find connected orange regions
# --------------------------------------------------

num_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        orange,
        connectivity=8
    )
)

print()
print(f"Orange regions found: {num_labels - 1}")


# --------------------------------------------------
# 6. Choose the foreground tennis court
#
# We want:
# - a large orange region
# - one that reaches well into the lower part
#   of the image
#
# This should reject the orange/background region.
# --------------------------------------------------

best_label = None
best_score = -1

for i in range(1, num_labels):

    x = stats[
        i,
        cv2.CC_STAT_LEFT
    ]

    y = stats[
        i,
        cv2.CC_STAT_TOP
    ]

    w = stats[
        i,
        cv2.CC_STAT_WIDTH
    ]

    h = stats[
        i,
        cv2.CC_STAT_HEIGHT
    ]

    area = stats[
        i,
        cv2.CC_STAT_AREA
    ]

    bottom = y + h

    # Ignore small orange objects
    if area < 10000:
        continue

    # Our foreground court should extend
    # well into the lower half of the frame.
    if bottom < height * 0.55:
        continue

    # Prefer:
    # 1. large regions
    # 2. regions extending farther downward
    score = area + (bottom * 100)

    print(
        f"Candidate {i}: "
        f"area={area}, "
        f"bottom={bottom}, "
        f"score={score:.0f}"
    )

    if score > best_score:

        best_score = score
        best_label = i


if best_label is None:

    raise RuntimeError(
        "Could not identify foreground court"
    )


print()
print(
    f"Selected region: {best_label}"
)


# --------------------------------------------------
# 7. Keep ONLY that region
# --------------------------------------------------

clean = np.zeros_like(orange)

clean[
    labels == best_label
] = 255


# --------------------------------------------------
# 8. Find outer contour
# --------------------------------------------------

contours, _ = cv2.findContours(
    clean,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if not contours:

    raise RuntimeError(
        "No foreground court contour found"
    )

largest = max(
    contours,
    key=cv2.contourArea
)


# --------------------------------------------------
# 9. Fill the entire selected court region
#
# This removes holes caused by:
# - white lines
# - tennis balls
# - players
# - shadows
# --------------------------------------------------

clean[:] = 0

cv2.drawContours(
    clean,
    [largest],
    -1,
    255,
    thickness=cv2.FILLED
)


# --------------------------------------------------
# 10. Create visual overlay
#
# Everything outside our detected court becomes dark.
# --------------------------------------------------

overlay = frame.copy()

outside = clean == 0

overlay[outside] = (
    overlay[outside] * 0.20
).astype(np.uint8)


# --------------------------------------------------
# 11. Draw detected boundary
# --------------------------------------------------

cv2.drawContours(
    overlay,
    [largest],
    -1,
    (0, 255, 0),
    3
)


# --------------------------------------------------
# 12. Save results
# --------------------------------------------------

cv2.imwrite(
    OUTPUT_MASK,
    clean
)

cv2.imwrite(
    OUTPUT_OVERLAY,
    overlay
)

print()
print("Saved:")
print(OUTPUT_MASK)
print(OUTPUT_OVERLAY)