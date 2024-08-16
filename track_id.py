from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8_aug.pt")

# Open the video file
video_path = "Test_Video5.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

track_history = defaultdict(list)
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_count += 1  # Increment frame count

    if success:
        results = model.track(frame, persist=True)

        if results and hasattr(results[0], 'boxes') and results[0].boxes:
            boxes = results[0].boxes

            if hasattr(boxes, 'xywh') and hasattr(boxes, 'id') and boxes.id is not None:
                annotated_frame = results[0].plot()

                track_ids = boxes.id.int().cpu().tolist()
                boxes_xywh = boxes.xywh.cpu().numpy()
                class_names = [results[0].names[i] for i in boxes.cls.int().cpu().tolist()]

                printed_ids = set()  # Keep track of printed IDs in the current frame

                for box, track_id, class_name in zip(boxes_xywh, track_ids, class_names):
                    x, y, w, h = box
                    track_history[track_id].append((x, y))

                    if track_id not in printed_ids:
                        # Print track ID
                        # print(f"Track ID: {track_id}")
                        printed_ids.add(track_id)

            else:
                annotated_frame = frame.copy()
        else:
            annotated_frame = frame.copy()

        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
