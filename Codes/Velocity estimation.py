from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd

# Load the YOLOv8 model
model = YOLO('yolov8_aug.pt')

# Open the video file
video_path = "Dynamic_velocity_test1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = "Output_Dynamic_test15.mp4"
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

track_history = defaultdict(lambda: [])
track_velocities = defaultdict(list)
track_distances = defaultdict(list)
first_frame_id = defaultdict(int)  # Track the first frame for each ID
track_classes = defaultdict(str)  # Store class name for each track ID
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
                # Assuming class names can be accessed like this, modify as needed
                class_names = [results[0].names[i] for i in boxes.cls.int().cpu().tolist()]

                current_frame_ids = set()

                for box, track_id, class_name in zip(boxes_xywh, track_ids, class_names):
                    x, y, w, h = box
                    track = track_history[track_id]
                    current_frame_ids.add(track_id)

                    if track_id not in first_frame_id:
                        first_frame_id[track_id] = frame_count
                        track_classes[track_id] = class_name  # Store class name

                    if track:
                        prev_x, prev_y = track[-1]
                        distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                        cumulative_distance = track_distances[track_id][-1] + distance if track_distances[track_id] else distance
                        track_distances[track_id].append(cumulative_distance)
                        time_elapsed = (frame_count - first_frame_id[track_id] + 1) / frame_rate
                        cumulative_velocity = cumulative_distance / time_elapsed if time_elapsed > 0 else 0
                        track_velocities[track_id].append(cumulative_velocity)
                    else:
                        cumulative_velocity = 0.0
                        track_velocities[track_id].append(cumulative_velocity)
                        track_distances[track_id].append(0.0)

                    track.append((x, y))

                    text = f"Velocity: {cumulative_velocity:.2f} px/s"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    text_width, text_height = text_size[0]
                    cv2.rectangle(annotated_frame, (int(x), int(y - text_height - 6)), (int(x + text_width), int(y)), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, text, (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            else:
                annotated_frame = frame.copy()
        else:
            annotated_frame = frame.copy()

        out.write(annotated_frame)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Preparing data for Excel
max_frame = frame_count
frame_data = {'Frame': list(range(1, max_frame + 1))}

for id_, first_frame in first_frame_id.items():
    class_name = track_classes[id_]
    velocity_col = f'ID {id_} {class_name} Velocity (px/s)'
    distance_col = f'ID {id_} {class_name} Distance (px)'

    velocities = [None] * (first_frame - 1) + track_velocities[id_]
    distances = [None] * (first_frame - 1) + track_distances[id_]

    velocities.extend([None] * (max_frame - len(velocities)))
    distances.extend([None] * (max_frame - len(distances)))

    frame_data[velocity_col] = velocities
    frame_data[distance_col] = distances

df = pd.DataFrame(frame_data)
excel_path = 'Dynamic_test2_velocity_with_id1.xlsx'
df.to_excel(excel_path, index=False)
