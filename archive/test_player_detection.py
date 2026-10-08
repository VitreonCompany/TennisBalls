import cv2
from ultralytics import YOLO

video_path = "videos/test_match.mp4"

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(video_path)

# Jump to 60 seconds into the video
cap.set(cv2.CAP_PROP_POS_MSEC, 60 * 1000)

success, frame = cap.read()
cap.release()

if not success:
    print("Could not read frame.")
    exit()

# Detect objects
results = model(frame)

# Draw detections
annotated_frame = results[0].plot()

cv2.imwrite("output/player_test.jpg", annotated_frame)

print("Done!")
print("Open: output/player_test.jpg")

