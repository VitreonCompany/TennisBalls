import cv2
import numpy as np

VIDEO_PATH = "videos/test_match.mp4"
OUTPUT_PATH = "output/court_points.jpg"

points = []

# Get frame at 60 seconds
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, 60 * 1000)
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read video frame.")

display = frame.copy()


def mouse_callback(event, x, y, flags, param):
    global display

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))

        cv2.circle(display, (x, y), 8, (0, 0, 255), -1)

        cv2.putText(
            display,
            str(len(points)),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        print(f"Point {len(points)}: ({x}, {y})")


cv2.namedWindow("Court Calibration", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Court Calibration", 1200, 700)
cv2.setMouseCallback("Court Calibration", mouse_callback)

print("Click these FOUR points in this order:")
print("1 = near-left singles baseline corner")
print("2 = near-right singles baseline corner")
print("3 = far-right singles baseline corner")
print("4 = far-left singles baseline corner")
print()
print("Press R to reset.")
print("Press ENTER when finished.")

while True:
    cv2.imshow("Court Calibration", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        points = []
        display = frame.copy()
        print("Reset.")

    elif key == 13 and len(points) == 4:
        break

    elif key == 27:
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

print("\nSelected court points:")
for i, point in enumerate(points, start=1):
    print(i, point)

cv2.imwrite(OUTPUT_PATH, display)

np.save("data/court_points.npy", np.array(points, dtype=np.float32))

print("\nSaved to data/court_points.npy")