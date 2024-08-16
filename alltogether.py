import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque, Counter, defaultdict

class ObjectTracker:
    def __init__(self, video_path, model_path='yolov8_aug.pt', max_history_len=5, max_direction_len=5, movement_threshold=5, frame_rate=30):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.prev_frame = None
        self.vehicle_histories = defaultdict(lambda: deque(maxlen=max_history_len))
        self.vehicle_directions = defaultdict(lambda: deque(maxlen=max_direction_len))
        self.vehicle_direction_start_time = defaultdict(lambda: cv2.getTickCount())  # Initialize direction start times
        self.colors = np.random.randint(0, 255, (100, 3))  # Random colors for tracking lines
        self.mask = None
        self.max_history_len = max_history_len
        self.max_direction_len = max_direction_len
        self.movement_threshold = movement_threshold
        self.frame_rate = frame_rate  # Assuming 30 frames per second

    def run_tracking(self):
        if not self.cap.isOpened():
            print("Error opening video stream or file")
            return
        
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read the first frame")
            return

        self.prev_frame = frame.copy()
        self.mask = np.zeros_like(frame)  # Initialize mask to match frame dimensions

        while self.cap.isOpened():
            success, frame = self.cap.read()
            
            if not success:
                break
            
            annotated_frame = self.process_frame(frame)
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

    def get_around_center(self, center_x, center_y, distance):
        pixels_around_center = [
            (center_x + distance, center_y),
            (center_x - distance, center_y),
            (center_x, center_y + distance),
            (center_x, center_y - distance),
            (center_x + distance, center_y + distance),
            (center_x + distance, center_y - distance),
            (center_x - distance, center_y + distance),
            (center_x - distance, center_y - distance),
            (center_x + distance // 2, center_y + distance // 2),
            (center_x - distance // 2, center_y - distance // 2)
        ]
        return pixels_around_center

    def average_position(self, positions):
        avg_x = int(sum(x for x, y in positions) / len(positions))
        avg_y = int(sum(y for x, y in positions) / len(positions))
        return avg_x, avg_y

    def calculate_velocity(self, old_pos, new_pos):
        dx = new_pos[0] - old_pos[0]
        dy = new_pos[1] - old_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        velocity = distance * self.frame_rate  # pixels per second
        return velocity

    def process_frame(self, frame, first_frame=False):
        results = self.model.track(frame, persist=True)
        annotated_frame = frame.copy()

        if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes:
            boxes = results[0].boxes

            if hasattr(boxes, 'xywh') and hasattr(boxes, 'id') and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                boxes_xywh = boxes.xywh.cpu().numpy()
                class_names = [results[0].names[i] for i in boxes.cls.int().cpu().tolist()]
                confidences = boxes.conf.cpu().tolist()

                for box, track_id, class_name in zip(boxes_xywh, track_ids, class_names):
                    x, y, w, h = box
                    x1, y1 = int(x - w / 2), int(y - h / 2)
                    x2, y2 = int(x + w / 2), int(y + h / 2)

                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

                    # Safe access to confidence
                    conf = confidences[track_ids.index(track_id)] if track_id < len(confidences) else 0

                    # Initialize vehicle history and direction if not present
                    if track_id not in self.vehicle_histories:
                        self.vehicle_histories[track_id] = deque(maxlen=self.max_history_len)
                        self.vehicle_directions[track_id] = deque(maxlen=self.max_direction_len)
                    
                    if track_id not in self.vehicle_direction_start_time:
                        self.vehicle_direction_start_time[track_id] = cv2.getTickCount()

                    # Add current position to history
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    self.vehicle_histories[track_id].append((center_x, center_y))

                    # Draw center of bounding box
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)

                    # Plot the 10 pixels around the center
                    positions_around_center = self.get_around_center(center_x, center_y, 10)
                    for px, py in positions_around_center:
                        cv2.circle(annotated_frame, (px, py), 3, (255, 255, 0), -1)

                    # Calculate time elapsed, velocity, acceleration, and distance
                    current_time = cv2.getTickCount()
                    time_elapsed = (current_time - self.vehicle_direction_start_time[track_id]) / cv2.getTickFrequency()
                    self.vehicle_direction_start_time[track_id] = current_time
                    
                    velocity = None
                    acceleration = None

                    if len(self.vehicle_histories[track_id]) > 1:
                        prev_center = self.vehicle_histories[track_id][-2]
                        prev_center_x, prev_center_y = prev_center
                        dx = center_x - prev_center_x
                        dy = center_y - prev_center_y
                        distance = np.sqrt(dx**2 + dy**2)
                        velocity = self.calculate_velocity(prev_center, (center_x, center_y))
                        
                        if len(self.vehicle_histories[track_id]) > 2:
                            second_last_center = self.vehicle_histories[track_id][-3]
                            prev_velocity = self.calculate_velocity(second_last_center, prev_center)
                            acceleration = (velocity - prev_velocity) / (1 / self.frame_rate)
                        
                        distance_str = f"Dist: {distance:.2f} px"
                    else:
                        distance_str = "Dist: N/A"

                    # Analyze movement direction
                    if len(self.vehicle_histories[track_id]) > 1:
                        prev_center = self.vehicle_histories[track_id][-2]
                        prev_center_x, prev_center_y = prev_center
                        dx = center_x - prev_center_x
                        dy = center_y - prev_center_y

                        if abs(dx) > self.movement_threshold or abs(dy) > self.movement_threshold:
                            if abs(dx) > abs(dy):  # Mainly horizontal movement
                                direction = "Right" if dx > 0 else "Left"
                            else:  # Mainly vertical movement
                                direction = "Down" if dy > 0 else "Up"
                        else:
                            direction = "Center"

                        # Add direction to history
                        self.vehicle_directions[track_id].append(direction)

                        # Determine and display the most common direction
                        if len(self.vehicle_directions[track_id]) == self.max_direction_len:
                            most_common_direction = Counter(self.vehicle_directions[track_id]).most_common(1)[0][0]
                        else:
                            most_common_direction = direction
                    else:
                        most_common_direction = "N/A"

                    elapsed_time_str = f"Time: {time_elapsed:.2f}s"
                    velocity_str = f"Vel: {velocity:.2f} px/s" if velocity else "Vel: N/A"
                    acceleration_str = f"Acc: {acceleration:.2f} px/s²" if acceleration else "Acc: N/A"
                    confidence_str = f"Conf: {conf:.2f}"

                    # Display ID, class, confidence, time elapsed, distance, velocity, acceleration, and direction
                    cv2.putText(annotated_frame, f"ID: {track_id}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(annotated_frame, f"Class: {class_name}", (int(x1), int(y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(annotated_frame, confidence_str, (int(x1), int(y1 - 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(annotated_frame, elapsed_time_str, (int(x1), int(y1 - 55)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(annotated_frame, distance_str, (int(x1), int(y1 - 70)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(annotated_frame, velocity_str, (int(x1), int(y1 - 85)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(annotated_frame, acceleration_str, (int(x1), int(y1 - 100)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(annotated_frame, f"Dir: {most_common_direction}", (int(x1), int(y1 - 115)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Combine the frame and the mask
        img = cv2.add(annotated_frame, self.mask)

        return img

# Usage example:
video_path = "testvdo (4).mp4"
tracker = ObjectTracker(video_path)
tracker.run_tracking()