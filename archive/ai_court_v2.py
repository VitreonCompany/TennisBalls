import cv2
import base64
import json
import re
import numpy as np
from openai import OpenAI

VIDEO_PATH = "videos/test_match.mp4"

OUTPUT_IMAGE = "output/ai_court_v2.jpg"
OUTPUT_POINTS = "data/ai_court_points_v2.npy"

client = OpenAI()

# --------------------------------------------------
# LOAD FRAME
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, 60 * 1000)

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read video")

height, width = frame.shape[:2]

# --------------------------------------------------
# ENCODE IMAGE
# --------------------------------------------------

success, buffer = cv2.imencode(
    ".jpg",
    frame,
    [cv2.IMWRITE_JPEG_QUALITY, 95]
)

if not success:
    raise RuntimeError("Could not encode frame")

image64 = base64.b64encode(
    buffer
).decode("utf-8")


# --------------------------------------------------
# ASK STRONG VISION MODEL
# --------------------------------------------------

prompt = f"""
You are calibrating a tennis court from a single camera image.

IMAGE:
width = {width}
height = {height}

Your task is NOT to independently guess eight tennis landmarks.

Instead, reason about the complete regulation SINGLES court as one
perspective-projected rectangle.

A regulation singles court is:

width = 8.23 metres
length = 23.77 metres

The service lines are 5.485 metres from each baseline.

The camera is positioned at a very oblique angle.

IMPORTANT VISUAL FACTS:

- The playing surface is orange.
- Blue surrounds the court.
- There are other courts and structural lines in the background.
  IGNORE THEM.
- The net is NOT a baseline.
- The doubles sidelines are NOT the singles sidelines.
- Some far court lines are faint.
- Use perspective geometry and the known structure of a tennis court
  to infer faint or partially obscured lines.
- White painted tennis lines are the evidence to align against.

Identify ONLY these four SINGLES baseline corners:

A = FAR baseline / LEFT singles sideline
B = FAR baseline / RIGHT singles sideline
C = NEAR baseline / LEFT singles sideline
D = NEAR baseline / RIGHT singles sideline

LEFT and RIGHT refer to the corresponding singles sidelines of the
court, not simply the left/right edge of the image.

The expected geometric structure is:

A ---------------- B
|                  |
|                  |
|     FAR HALF     |
|                  |
------ NET ---------
|                  |
|     NEAR HALF    |
|                  |
C ---------------- D

Perspective may make this quadrilateral look extremely distorted.

Study the painted lines carefully.

Return pixel coordinates in the ORIGINAL {width}x{height} image.

Also return a confidence from 0 to 1 for each point.

Return ONLY JSON:

{{
  "A": {{"x": 0, "y": 0, "confidence": 0.0}},
  "B": {{"x": 0, "y": 0, "confidence": 0.0}},
  "C": {{"x": 0, "y": 0, "confidence": 0.0}},
  "D": {{"x": 0, "y": 0, "confidence": 0.0}}
}}
"""

response = client.responses.create(
    model="gpt-5.6-sol",
    reasoning={
        "effort": "high"
    },
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": prompt
                },
                {
                    "type": "input_image",
                    "image_url":
                        f"data:image/jpeg;base64,{image64}",
                    "detail": "high"
                }
            ]
        }
    ]
)

text = response.output_text.strip()

text = re.sub(
    r"^```json\s*",
    "",
    text
)

text = re.sub(
    r"\s*```$",
    "",
    text
)

result = json.loads(text)


# --------------------------------------------------
# FOUR AI CORNERS
# --------------------------------------------------

A = np.array(
    [result["A"]["x"], result["A"]["y"]],
    dtype=np.float32
)

B = np.array(
    [result["B"]["x"], result["B"]["y"]],
    dtype=np.float32
)

C = np.array(
    [result["C"]["x"], result["C"]["y"]],
    dtype=np.float32
)

D = np.array(
    [result["D"]["x"], result["D"]["y"]],
    dtype=np.float32
)

image_corners = np.array(
    [A, B, D, C],
    dtype=np.float32
)


# --------------------------------------------------
# REAL TENNIS COURT
# --------------------------------------------------

COURT_WIDTH = 8.23
COURT_LENGTH = 23.77
SERVICE = 5.485

world_corners = np.array([
    [0, 0],
    [COURT_WIDTH, 0],
    [COURT_WIDTH, COURT_LENGTH],
    [0, COURT_LENGTH]
], dtype=np.float32)


# --------------------------------------------------
# HOMOGRAPHY
# world court -> image
# --------------------------------------------------

H = cv2.getPerspectiveTransform(
    world_corners,
    image_corners
)


def world_to_image(points):

    points = np.array(
        [points],
        dtype=np.float32
    )

    return cv2.perspectiveTransform(
        points,
        H
    )[0]


# --------------------------------------------------
# CALCULATE ALL 8 POINTS
#
# Points 3-6 are NOT AI guesses.
# They are mathematically derived from the
# four corners and regulation court geometry.
# --------------------------------------------------

world_landmarks = np.array([

    # Far baseline
    [0, 0],
    [COURT_WIDTH, 0],

    # Far service line
    [0, SERVICE],
    [COURT_WIDTH, SERVICE],

    # Near service line
    [0, COURT_LENGTH - SERVICE],
    [COURT_WIDTH, COURT_LENGTH - SERVICE],

    # Near baseline
    [0, COURT_LENGTH],
    [COURT_WIDTH, COURT_LENGTH]

], dtype=np.float32)

points = world_to_image(
    world_landmarks
)

np.save(
    OUTPUT_POINTS,
    points
)


# --------------------------------------------------
# DRAW COMPLETE COURT
# --------------------------------------------------

output = frame.copy()

# Outer court
outer = world_to_image(
    world_corners
)

for i in range(4):

    p1 = tuple(
        np.round(outer[i]).astype(int)
    )

    p2 = tuple(
        np.round(outer[(i + 1) % 4]).astype(int)
    )

    cv2.line(
        output,
        p1,
        p2,
        (255, 0, 255),
        3
    )


# Service lines
for y in [
    SERVICE,
    COURT_LENGTH - SERVICE
]:

    line = world_to_image(
        np.array([
            [0, y],
            [COURT_WIDTH, y]
        ], dtype=np.float32)
    )

    cv2.line(
        output,
        tuple(np.round(line[0]).astype(int)),
        tuple(np.round(line[1]).astype(int)),
        (255, 0, 255),
        2
    )


# Centre service line
centre = world_to_image(
    np.array([
        [COURT_WIDTH / 2, SERVICE],
        [
            COURT_WIDTH / 2,
            COURT_LENGTH - SERVICE
        ]
    ], dtype=np.float32)
)

cv2.line(
    output,
    tuple(np.round(centre[0]).astype(int)),
    tuple(np.round(centre[1]).astype(int)),
    (255, 0, 255),
    2
)


# Draw landmarks
for i, (x, y) in enumerate(
    points,
    start=1
):

    x = int(round(x))
    y = int(round(y))

    cv2.circle(
        output,
        (x, y),
        7,
        (0, 255, 255),
        -1
    )

    cv2.putText(
        output,
        str(i),
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


cv2.imwrite(
    OUTPUT_IMAGE,
    output
)


# --------------------------------------------------
# PRINT AI CORNERS
# --------------------------------------------------

print()
print("AI CORNERS")
print("-" * 60)

for name in ["A", "B", "C", "D"]:

    p = result[name]

    print(
        f"{name}: "
        f"({p['x']}, {p['y']}) "
        f"confidence={p['confidence']}"
    )


# --------------------------------------------------
# BENCHMARK AGAINST YOUR MANUAL POINTS
#
# IMPORTANT:
# Manual points are NOT used to calculate anything.
# They are ONLY used to grade the automatic result.
# --------------------------------------------------

manual = np.load(
    "data/court_points_8.npy"
).astype(np.float32)

errors = np.linalg.norm(
    points - manual,
    axis=1
)

print()
print("8-POINT RESULTS")
print("-" * 60)

for i in range(8):

    print(
        f"Point {i + 1}: "
        f"AUTO=({points[i][0]:.0f}, "
        f"{points[i][1]:.0f}) | "
        f"MANUAL=({manual[i][0]:.0f}, "
        f"{manual[i][1]:.0f}) | "
        f"error={errors[i]:.1f}px"
    )

print()
print(
    f"Mean pixel error: "
    f"{errors.mean():.1f}px"
)

print(
    f"Max pixel error: "
    f"{errors.max():.1f}px"
)

print()
print(f"Saved: {OUTPUT_IMAGE}")
print(f"Saved: {OUTPUT_POINTS}")