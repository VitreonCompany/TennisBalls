import cv2
from ultralytics import YOLO

VIDEO_PATH = "videos/test_match.mp4"
OUTPUT_PATH = "output/far_player_test.mp4"

START_TIME = 60
DURATION = 10

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(cv2.CAP_PROP_POS_MSEC, START_TIME * 1000)

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

total_frames = int(DURATION * fps)

detected_frames = 0

for frame_number in range(total_frames):

    success, frame = cap.read()

    if not success:
        break

    # ------------------------------------------------
    # FAR-COURT REGION
    #
    # For this particular camera the far player lives
    # mainly in the upper/left part of the image.
    # ------------------------------------------------

    crop_x1 = 0
    crop_y1 = 120
    crop_x2 = 1000
    crop_y2 = 520

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    # YOLO gets to analyse this smaller region at
    # high resolution
    results = model(
        crop,
        classes=[0],
        conf=0.10,
        imgsz=1280,
        verbose=False
    )[0]

    best_box = None
    best_conf = 0

    for box in results.boxes:

        confidence = float(box.conf[0])

        if confidence > best_conf:
            best_conf = confidence
            best_box = box

    if best_box is not None:

        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()

        # Convert crop coordinates back into
        # original-video coordinates
        x1 += crop_x1
        x2 += crop_x1
        y1 += crop_y1
        y2 += crop_y1

        detected_frames += 1

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"FAR PLAYER {best_conf:.2f}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Show the region we're searching
    cv2.rectangle(
        frame,
        (crop_x1, crop_y1),
        (crop_x2, crop_y2),
        (255, 0, 0),
        2
    )

    writer.write(frame)

    if frame_number % 50 == 0:
        print(f"Processed {frame_number}/{total_frames}")


cap.release()
writer.release()

detection_rate = detected_frames / total_frames * 100

print()
print("Finished!")
print(f"Far player detected: {detected_frames}/{total_frames} frames")
print(f"Detection rate: {detection_rate:.1f}%")
print(f"Video: {OUTPUT_PATH}")