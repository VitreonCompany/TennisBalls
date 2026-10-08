import cv2
import numpy as np

VIDEO_PATH = "videos/test_match.mp4"
OUTPUT_POINTS = "data/court_points_8.npy"
OUTPUT_IMAGE = "output/court_points_8.jpg"

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, 60 * 1000)

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read video frame.")

display = frame.copy()
points = []

names = [
    "1 FAR baseline - left singles corner",
    "2 FAR baseline - right singles corner",
    "3 FAR service line - left singles sideline",
    "4 FAR service line - right singles sideline",
    "5 NEAR service line - left singles sideline",
    "6 NEAR service line - right singles sideline",
    "7 NEAR baseline - left singles corner",
    "8 NEAR baseline - right singles corner",
]


def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(points) >= 8:
            return

        points.append((x, y))

        number = len(points)

        cv2.circle(display, (x, y), 7, (0, 0, 255), -1)

        cv2.putText(
            display,
            str(number),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        print(f"{names[number - 1]}: ({x}, {y})")

        if number < 8:
            print(f"NEXT: {names[number]}")
        else:
            print()
            print("All 8 points selected.")
            print("Press S to save or R to restart.")


cv2.namedWindow("8 Point Court Calibration", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("8 Point Court Calibration", mouse_callback)

print("Click the points IN THIS EXACT ORDER:")
print()

for name in names:
    print(name)

print()
print(f"FIRST: {names[0]}")

while True:

    cv2.imshow("8 Point Court Calibration", display)

    key = cv2.waitKey(20) & 0xFF

    if key == ord("r"):

        points = []
        display = frame.copy()

        print()
        print("Restarted.")
        print(f"FIRST: {names[0]}")

    elif key == ord("s"):

        if len(points) != 8:
            print("You need exactly 8 points.")
            continue

        np.save(
            OUTPUT_POINTS,
            np.array(points, dtype=np.float32)
        )

        cv2.imwrite(OUTPUT_IMAGE, display)

        print()
        print(f"Saved points: {OUTPUT_POINTS}")
        print(f"Saved image: {OUTPUT_IMAGE}")

        break

    elif key == 27:  # ESC
        break


cv2.destroyAllWindows()