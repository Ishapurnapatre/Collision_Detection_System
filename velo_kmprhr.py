2200 * 2865

from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd

class Tracker:
    def __init__(self, video_path, model_path, scale_factor):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = 0
        
        self.scale_factor = scale_factor
        
        self.out = cv2.VideoWriter("Output_Dynamic_test15.mp4",
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   self.frame_rate,
                                   (self.frame_width, self.frame_height))
        
        self.track_history = defaultdict(list)
        self.track_velocities = defaultdict(list)
        self.track_distances = defaultdict(list)
        self.first_frame_id = defaultdict(int)
        self.track_classes = defaultdict(str)

    def track_video(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            self.frame_count += 1

            if success:
                annotated_frame = self._process_frame(frame)
                self.out.write(annotated_frame)
                cv2.imshow("YOLOv8 Tracking", annotated_frame)

                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
            else:
                break

        self._cleanup()

    def _process_frame(self, frame):
        results = self.model.track(frame, persist=True)
        annotated_frame = frame.copy()

        if results and hasattr(results[0], 'boxes') and results[0].boxes:
            boxes = results[0].boxes

            if hasattr(boxes, 'xywh') and hasattr(boxes, 'id') and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                boxes_xywh = boxes.xywh.cpu().numpy()
                class_names = [results[0].names[i] for i in boxes.cls.int().cpu().tolist()]

                for box, track_id, class_name in zip(boxes_xywh, track_ids, class_names):
                    x, y, w, h = box
                    track = self.track_history[track_id]

                    if track_id not in self.first_frame_id:
                        self.first_frame_id[track_id] = self.frame_count
                        self.track_classes[track_id] = class_name

                    if track:
                        prev_x, prev_y = track[-1]
                        distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                        cumulative_distance = self.track_distances[track_id][-1] + distance if self.track_distances[track_id] else distance
                        self.track_distances[track_id].append(cumulative_distance)
                        time_elapsed = (self.frame_count - self.first_frame_id[track_id] + 1) / self.frame_rate
                        cumulative_velocity_px_s = cumulative_distance / time_elapsed if time_elapsed > 0 else 0
                        cumulative_velocity_m_s = cumulative_velocity_px_s * self.scale_factor
                        cumulative_velocity_km_h = cumulative_velocity_m_s * 3.6
                        self.track_velocities[track_id].append(cumulative_velocity_km_h)
                    else:
                        cumulative_velocity_km_h = 0.0
                        self.track_velocities[track_id].append(cumulative_velocity_km_h)
                        self.track_distances[track_id].append(0.0)

                    track.append((x, y))

                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
                    # Draw ID
                    cv2.putText(annotated_frame, f"ID: {track_id}", (int(x - w / 2), int(y - h / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    # Draw velocity
                    text = f"Velocity: {cumulative_velocity_km_h:.2f} km/h"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    text_width, text_height = text_size[0]
                    cv2.rectangle(annotated_frame, (int(x), int(y - text_height - 6)), (int(x + text_width), int(y)), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, text, (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame

    def _cleanup(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

        max_frame = self.frame_count
        frame_data = {'Frame': list(range(1, max_frame + 1))}

        for id_, first_frame in self.first_frame_id.items():
            class_name = self.track_classes[id_]
            velocity_col = f'ID {id_} {class_name} Velocity (km/h)'
            distance_col = f'ID {id_} {class_name} Distance (px)'

            velocities = [None] * (first_frame - 1) + self.track_velocities[id_]
            distances = [None] * (first_frame - 1) + self.track_distances[id_]

            velocities.extend([None] * (max_frame - len(velocities)))
            distances.extend([None] * (max_frame - len(distances)))

            frame_data[velocity_col] = velocities
            frame_data[distance_col] = distances

        df = pd.DataFrame(frame_data)
        df.to_excel('Dynamic_test2_velocity_with_id1.xlsx', index=False)

if __name__ == "__main__":
    scale_factor = 0.02  # Replace this with your actual scale factor (meters per pixel)
    tracker = Tracker("C:/Users/Mech.coep/Desktop/Yolo/Dynamic_velocity_test1.mp4", "C:/Users/Mech.coep/Desktop/Yolo/yolov8_aug.pt", scale_factor)
    tracker.track_video()
