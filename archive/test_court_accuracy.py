import cv2
import numpy as np

VIDEO_PATH = "videos/test_match.mp4"

H = np.load("data/homography.npy")

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, 60 * 1000)
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read video.")

clicked = None


def mouse_callback(event, x, y, flags, param):
    global clicked

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = (x, y)

        pixel = np.array([[[x, y]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pixel, H)[0][0]

        print(f"\nPixel: ({x}, {y})")
        print(f"Court coordinate: x={result[0]:.2f}m, y={result[1]:.2f}m")


cv2.namedWindow("Accuracy Test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Accuracy Test", 1200, 700)
cv2.setMouseCallback("Accuracy Test", mouse_callback)

print("Click a service-line / singles-sideline intersection.")
print("Press ESC when finished.")

while True:
    cv2.imshow("Accuracy Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()