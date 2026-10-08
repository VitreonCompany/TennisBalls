import cv2
import base64
import json
import re
import numpy as np
from openai import OpenAI


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "videos/test_match.mp4"

OUTPUT_IMAGE = "output/ai_court_v3.jpg"
OUTPUT_POINTS = "data/ai_court_points_v3.npy"

MANUAL_POINTS = "data/court_points_8.npy"

MODEL = "gpt-6-astra"


# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI()


# ============================================================
# LOAD TEST FRAME
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

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

height, width = frame.shape[:2]

print()
print(
    f"Image size: {width} x {height}"
)


# ============================================================
# ENCODE ORIGINAL IMAGE
# ============================================================

success, buffer = cv2.imencode(
    ".jpg",
    frame,
    [
        cv2.IMWRITE_JPEG_QUALITY,
        98
    ]
)

if not success:
    raise RuntimeError(
        "Could not encode frame"
    )

image64 = base64.b64encode(
    buffer
).decode("utf-8")


# ============================================================
# PROMPT
# ============================================================

prompt = f"""
You are an expert computer-vision system performing precise
tennis-court camera calibration.

You are looking at ONE original video frame.

IMAGE RESOLUTION:

width = {width} pixels
height = {height} pixels


============================================================
OBJECTIVE
============================================================

Locate EIGHT precise pixel landmarks on the MAIN FOREGROUND
REGULATION SINGLES TENNIS COURT.

You must solve the court from scratch.

Do not make a quick visual guess.

Before returning your answer, internally:

1. detect the court,
2. identify the correct painted tennis lines,
3. estimate all eight points,
4. reconstruct the implied court geometry,
5. check that the geometry is physically consistent,
6. re-inspect the original image,
7. identify any suspicious points,
8. revise them,
9. repeat your internal checking if necessary,
10. return only your final best estimate.

Do not output your intermediate reasoning.


============================================================
THE EIGHT REQUIRED LANDMARKS
============================================================

Point 1:

FAR baseline
intersection with
LEFT SINGLES sideline.


Point 2:

FAR baseline
intersection with
RIGHT SINGLES sideline.


Point 3:

FAR service line
intersection with
LEFT SINGLES sideline.


Point 4:

FAR service line
intersection with
RIGHT SINGLES sideline.


Point 5:

NEAR service line
intersection with
LEFT SINGLES sideline.


Point 6:

NEAR service line
intersection with
RIGHT SINGLES sideline.


Point 7:

NEAR baseline
intersection with
LEFT SINGLES sideline.


Point 8:

NEAR baseline
intersection with
RIGHT SINGLES sideline.


============================================================
REGULATION COURT GEOMETRY
============================================================

A regulation SINGLES tennis court is:

8.23 metres wide

23.77 metres long


Distance from each baseline to its service line:

5.485 metres


Therefore, in real-world court coordinates:

Point 1 = (0.000, 0.000)
Point 2 = (8.230, 0.000)

Point 3 = (0.000, 5.485)
Point 4 = (8.230, 5.485)

Point 5 = (0.000, 18.285)
Point 6 = (8.230, 18.285)

Point 7 = (0.000, 23.770)
Point 8 = (8.230, 23.770)


All eight points therefore belong to ONE perspective projection
of this known planar geometry.


============================================================
VERY IMPORTANT: SINGLES VS DOUBLES
============================================================

This court has doubles alleys.

DO NOT use the outside doubles sidelines.

We need the INNER SINGLES SIDELINES.

The required longitudinal relationships are:

1 -> 3 -> 5 -> 7

all lie on the SAME physical LEFT SINGLES SIDELINE.

And:

2 -> 4 -> 6 -> 8

all lie on the SAME physical RIGHT SINGLES SIDELINE.


Because of perspective, they will not necessarily appear
equally spaced in the image.


============================================================
CROSS-COURT STRUCTURE
============================================================

1 -> 2

is the FAR BASELINE.


3 -> 4

is the FAR SERVICE LINE.


5 -> 6

is the NEAR SERVICE LINE.


7 -> 8

is the NEAR BASELINE.


These four real-world lines are parallel.

Because of perspective they should belong to the same
vanishing-point family in the image.


============================================================
VISUAL SCENE INFORMATION
============================================================

The main playing surface is ORANGE.

The surrounding floor is BLUE/GREY.

The image contains distracting geometry including:

- another tennis court,
- walls,
- structural edges,
- net structures,
- background lines,
- doubles sidelines,
- court-surround boundaries.

IGNORE geometry that does not belong to the MAIN FOREGROUND
orange tennis court.


============================================================
THE NET
============================================================

Be particularly careful with the tennis net.

The net is NOT:

- a baseline,
- a service line,
- or one of the requested eight landmarks.

Do not mistake the net tape or net structure for a painted
court line.


============================================================
FAINT FAR COURT
============================================================

The FAR end of the court is strongly foreshortened.

Some painted lines may be faint, partially hidden or difficult
to distinguish.

Do not simply choose the clearest nearby line.

When a landmark is difficult:

A. identify visible fragments of the correct painted line,

B. identify the two perspective line families,

C. use the known regulation court dimensions,

D. use the other court intersections,

E. infer where the missing/faint intersection must lie,

F. then return the best pixel coordinate.


============================================================
GEOMETRIC SELF-CHECK
============================================================

After your initial estimate, perform an internal geometric
validation.

Check that:

1. Points 1, 3, 5, 7 follow one physical singles sideline.

2. Points 2, 4, 6, 8 follow the other physical singles sideline.

3. Lines 1-2, 3-4, 5-6 and 7-8 represent the four correct
   cross-court painted lines.

4. The implied projective transformation is compatible with a
   rectangular 8.23m x 23.77m tennis court.

5. The service-line positions are compatible with their known
   distance of 5.485m from each baseline.

6. None of the points accidentally use a doubles sideline.

7. None of the points belong to the background court.

8. None of the cross-court lines correspond to the tennis net.

9. Each final point lies at, or is geometrically inferred from,
   the appropriate painted-line intersection.


============================================================
SECOND VISUAL PASS
============================================================

Once you have a geometrically consistent solution, inspect the
original image AGAIN.

Do not accept geometry alone.

Compare the projected court against the actual visible white
paint.

If one end of the projected court drifts away from the painted
lines, correct the responsible landmarks and validate again.

Prefer agreement with BOTH:

- visible image evidence
- regulation projective geometry


============================================================
PRECISION
============================================================

Return coordinates in the ORIGINAL:

{width} x {height}

pixel image.

Do not return normalized coordinates.

Use integer pixel coordinates.

Estimate the CENTER of the painted line intersection rather
than an arbitrary edge of the white paint.


============================================================
CONFIDENCE
============================================================

For each point return confidence from:

0.0 to 1.0

Confidence should represent how certain you are that the
coordinate corresponds to the correct physical tennis
landmark.

A geometrically inferred faint point may have lower confidence
than a clearly visible intersection.


============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Do not use markdown.

Do not provide an explanation.

Do not expose your reasoning.

Use exactly this structure:

{{
    "points": [
        {{
            "id": 1,
            "x": 0,
            "y": 0,
            "confidence": 0.0
        }},
        {{
            "id": 2,
            "x": 0,
            "y": 0,
            "confidence": 0.0
        }},
        {{
            "id": 3,
            "x": 0,
            "y": 0,
            "confidence": 0.0
        }},
        {{
            "id": 4,
            "x": 0,
            "y": 0,
            "confidence": 0.0
        }},
        {{
            "id": 5,
            "x": 0,
            "y": 0,
            "confidence": 0.0
        }},
        {{
            "id": 6,
            "x": 0,
            "y": 0,
            "confidence": 0.0
        }},
        {{
            "id": 7,
            "x": 0,
            "y": 0,
            "confidence": 0.0
        }},
        {{
            "id": 8,
            "x": 0,
            "y": 0,
            "confidence": 0.0
        }}
    ]
}}
"""


# ============================================================
# CALL MODEL
# ============================================================

print()
print("Calling GPT-5.6 Sol...")
print("Fresh 8-point detection + self-check.")
print()

response = client.responses.create(

    model=MODEL,

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


# ============================================================
# READ RESPONSE
# ============================================================

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

try:

    result = json.loads(text)

except json.JSONDecodeError:

    print()
    print("MODEL RESPONSE:")
    print(text)

    raise RuntimeError(
        "Model did not return valid JSON"
    )


# ============================================================
# VALIDATE RESPONSE
# ============================================================

if "points" not in result:

    raise RuntimeError(
        "Response does not contain 'points'"
    )

returned_points = result["points"]

if len(returned_points) != 8:

    raise RuntimeError(
        f"Expected 8 points, got "
        f"{len(returned_points)}"
    )


returned_points.sort(
    key=lambda p: p["id"]
)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

new_points = np.array(

    [
        [
            p["x"],
            p["y"]
        ]

        for p in returned_points
    ],

    dtype=np.float32
)


# ============================================================
# SAVE RAW AI POINTS
# ============================================================

np.save(
    OUTPUT_POINTS,
    new_points
)


# ============================================================
# DRAW RESULT
# ============================================================

output = frame.copy()


# ------------------------------------------------------------
# LEFT SINGLES SIDELINE
# 1 -> 3 -> 5 -> 7
# ------------------------------------------------------------

left_indices = [
    0,
    2,
    4,
    6
]

for a, b in zip(
    left_indices[:-1],
    left_indices[1:]
):

    cv2.line(

        output,

        tuple(
            np.round(
                new_points[a]
            ).astype(int)
        ),

        tuple(
            np.round(
                new_points[b]
            ).astype(int)
        ),

        (255, 0, 255),

        3
    )


# ------------------------------------------------------------
# RIGHT SINGLES SIDELINE
# 2 -> 4 -> 6 -> 8
# ------------------------------------------------------------

right_indices = [
    1,
    3,
    5,
    7
]

for a, b in zip(
    right_indices[:-1],
    right_indices[1:]
):

    cv2.line(

        output,

        tuple(
            np.round(
                new_points[a]
            ).astype(int)
        ),

        tuple(
            np.round(
                new_points[b]
            ).astype(int)
        ),

        (255, 0, 255),

        3
    )


# ------------------------------------------------------------
# CROSS-COURT LINES
# ------------------------------------------------------------

cross_lines = [

    (0, 1),   # far baseline
    (2, 3),   # far service line
    (4, 5),   # near service line
    (6, 7)    # near baseline
]


for a, b in cross_lines:

    cv2.line(

        output,

        tuple(
            np.round(
                new_points[a]
            ).astype(int)
        ),

        tuple(
            np.round(
                new_points[b]
            ).astype(int)
        ),

        (255, 0, 255),

        3
    )


# ============================================================
# DRAW LANDMARKS
# ============================================================

for i, point in enumerate(
    new_points,
    start=1
):

    x = int(
        round(point[0])
    )

    y = int(
        round(point[1])
    )

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
        (x + 9, y - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


# ============================================================
# SAVE IMAGE
# ============================================================

cv2.imwrite(
    OUTPUT_IMAGE,
    output
)


# ============================================================
# PRINT AI RESULT
# ============================================================

print()
print("GPT-5.6 SOL RESULT")
print("=" * 65)

for p in returned_points:

    print(
        f"Point {p['id']}: "
        f"({p['x']}, {p['y']}) | "
        f"confidence="
        f"{p['confidence']}"
    )


# ============================================================
# MANUAL BENCHMARK
#
# CRITICAL:
#
# These manual points were NOT sent to the AI.
#
# They are loaded only AFTER the AI has completed its
# prediction so that we can objectively grade it.
# ============================================================

manual = np.load(
    MANUAL_POINTS
).astype(np.float32)


errors = np.linalg.norm(
    new_points - manual,
    axis=1
)


print()
print("HIDDEN MANUAL BENCHMARK")
print("=" * 65)


for i in range(8):

    print(

        f"Point {i + 1}: "

        f"AUTO=("
        f"{new_points[i][0]:.0f}, "
        f"{new_points[i][1]:.0f}) | "

        f"MANUAL=("
        f"{manual[i][0]:.0f}, "
        f"{manual[i][1]:.0f}) | "

        f"error="
        f"{errors[i]:.1f}px"
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


# ============================================================
# COMPARE TO V2 IF AVAILABLE
# ============================================================

try:

    v2 = np.load(
        "data/ai_court_points_v2.npy"
    ).astype(np.float32)

    v2_errors = np.linalg.norm(
        v2 - manual,
        axis=1
    )

    print()
    print("V2 vs V3")
    print("=" * 65)

    print(
        f"V2 mean error: "
        f"{v2_errors.mean():.1f}px"
    )

    print(
        f"V3 mean error: "
        f"{errors.mean():.1f}px"
    )

    print(
        f"V2 max error: "
        f"{v2_errors.max():.1f}px"
    )

    print(
        f"V3 max error: "
        f"{errors.max():.1f}px"
    )

except FileNotFoundError:

    pass


print()
print(f"Saved: {OUTPUT_IMAGE}")
print(f"Saved: {OUTPUT_POINTS}")