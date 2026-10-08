import cv2
import base64
import json
import re
import numpy as np
from openai import OpenAI

VIDEO_PATH = "videos/test_match.mp4"

client = OpenAI()

# ---------------------------------------------
# Extract the same frame we've been using
# ---------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, 60 * 1000)

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read video frame")

height, width = frame.shape[:2]

cv2.imwrite("output/ai_input_frame.jpg", frame)

# Convert image to base64
success, buffer = cv2.imencode(".jpg", frame)

if not success:
    raise RuntimeError("Could not encode frame")

base64_image = base64.b64encode(buffer).decode("utf-8")


# ---------------------------------------------
# Ask vision model to locate court landmarks
# ---------------------------------------------

prompt = f"""
You are analysing a tennis court image.

Image dimensions:
width = {width} pixels
height = {height} pixels

Identify the PIXEL coordinates of these 8 landmarks
on the ORANGE tennis court:

1. FAR baseline - left singles sideline intersection
2. FAR baseline - right singles sideline intersection
3. FAR service line - left singles sideline intersection
4. FAR service line - right singles sideline intersection
5. NEAR service line - left singles sideline intersection
6. NEAR service line - right singles sideline intersection
7. NEAR baseline - left singles sideline intersection
8. NEAR baseline - right singles sideline intersection

Important:

- Use the SINGLES sidelines, not the doubles sidelines.
- FAR means the end farther from the camera.
- NEAR means the end closest to the camera.
- Coordinates use image pixels:
  x increases left-to-right.
  y increases top-to-bottom.
- Some lines are faint. Use the known geometry of a regulation
  tennis court to reason about where obscured/faint intersections
  should lie.
- Do not use the net as a baseline or service line.
- Return your best estimate even when a landmark is difficult.

Return ONLY valid JSON in exactly this format:

{{
  "points": [
    {{"id": 1, "x": 0, "y": 0}},
    {{"id": 2, "x": 0, "y": 0}},
    {{"id": 3, "x": 0, "y": 0}},
    {{"id": 4, "x": 0, "y": 0}},
    {{"id": 5, "x": 0, "y": 0}},
    {{"id": 6, "x": 0, "y": 0}},
    {{"id": 7, "x": 0, "y": 0}},
    {{"id": 8, "x": 0, "y": 0}}
  ]
}}
"""

response = client.responses.create(
    model="gpt-6-luna",
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
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
            ]
        }
    ]
)

text = response.output_text.strip()

# Remove markdown fences if the model happens to use them
text = re.sub(r"^```json\s*", "", text)
text = re.sub(r"\s*```$", "", text)

data = json.loads(text)

ai_points = np.array(
    [[p["x"], p["y"]] for p in data["points"]],
    dtype=np.float32
)

np.save("data/ai_court_points.npy", ai_points)


# ---------------------------------------------
# Compare with YOUR manual landmarks
# ---------------------------------------------

manual_points = np.load("data/court_points_8.npy")

print()
print("AI vs manual:")
print("-" * 60)

errors = []

for i in range(8):

    ai = ai_points[i]
    manual = manual_points[i]

    error = np.linalg.norm(ai - manual)
    errors.append(error)

    print(
        f"Point {i + 1}: "
        f"AI=({ai[0]:.0f}, {ai[1]:.0f}) | "
        f"Manual=({manual[0]:.0f}, {manual[1]:.0f}) | "
        f"error={error:.1f}px"
    )


print()
print(f"Mean pixel error: {np.mean(errors):.1f}px")
print(f"Max pixel error:  {np.max(errors):.1f}px")


# ---------------------------------------------
# Draw AI predictions
# ---------------------------------------------

output = frame.copy()

for i, (x, y) in enumerate(ai_points):

    cv2.circle(
        output,
        (int(x), int(y)),
        8,
        (255, 0, 255),
        -1
    )

    cv2.putText(
        output,
        f"AI {i + 1}",
        (int(x) + 10, int(y) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2
    )


cv2.imwrite(
    "output/ai_court_points.jpg",
    output
)

print()
print("Saved:")
print("data/ai_court_points.npy")
print("output/ai_court_points.jpg")