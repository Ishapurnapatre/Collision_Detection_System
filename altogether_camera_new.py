import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque, Counter, defaultdict
import os

class ObjectTracker:
    def __init__(self, model_path='yolov8_aug.pt', max_history_len=5, max_direction_len=5, movement_threshold=5, frame_rate=30):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(0)  # Use live webcam instead of a video file
        
        # Increase camera resolution for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Full HD width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Full HD height
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS to 30 for smoothness
        
        self.prev_frame = None
        self.vehicle_histories = defaultdict(lambda: deque(maxlen=max_history_len))
        self.vehicle_directions = defaultdict(lambda: deque(maxlen=max_direction_len))
        self.vehicle_direction_start_time = defaultdict(lambda: cv2.getTickCount())  # Initialize direction start times
        self.colors = np.random.randint(0, 255, (100, 3))  # Random colors for tracking lines
        self.mask = None
        self.max_history_len = max_history_len
        self.max_direction_len = max_direction_len
        self.movement_threshold = movement_threshold
        self.frame_rate = frame_rate  # Assuming 30 FPS
        
        # Define output video path (Desktop) and use avi format
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        self.output_path = os.path.join(desktop_path, "processed_video.avi")

        # Video writer setup for MP4 format
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 'xvid' is the codec for avi format
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(self.output_path, fourcc, 30, (self.frame_width, self.frame_height))

    def run_tracking(self):
        if not self.cap.isOpened():
            print("Error opening webcam")
            return
        
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read from webcam")
            return

        self.prev_frame = frame.copy()
        self.mask = np.zeros_like(frame)  # Initialize mask to match frame dimensions

        while self.cap.isOpened():
            success, frame = self.cap.read()
            
            if not success:
                break
            
            annotated_frame = self.process_frame(frame)

            # Save the processed frame to video
            self.out.write(annotated_frame)

            cv2.imshow("YOLOv8 Live Tracking", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break
        
        self.cap.release()
        self.out.release()  # Release video writer
        cv2.destroyAllWindows()
        print(f"Processed video saved at: {self.output_path}")

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

    def process_frame(self, frame):
        # Reduce blurriness & improve sharpness
        frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Apply Gaussian blur to smooth noise
        frame = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)  # Enhance details
        
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
                    print("w=",w)

                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Safe access to confidence
                    #conf = confidences[track_ids.index(track_id)] if track_id < len(confidences) else 0

                    # Add current position to history
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    self.vehicle_histories[track_id].append((center_x, center_y))

                    # Draw center of bounding box
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)

                    # Calculate velocity
                    velocity = None
                    if len(self.vehicle_histories[track_id]) > 1:
                        prev_center = self.vehicle_histories[track_id][-2]
                        velocity = self.calculate_velocity(prev_center, (center_x, center_y))
                    
                    velocity_str = f"Vel: {velocity:.2f} px/s" if velocity else "Vel: N/A"
                    #confidence_str = f"Conf: {conf:.2f}"

                    # Display tracking info
                    cv2.putText(annotated_frame, f"ID: {track_id}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(annotated_frame, f"Class: {class_name}", (int(x1), int(y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    #cv2.putText(annotated_frame, confidence_str, (int(x1), int(y1 - 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(annotated_frame, velocity_str, (int(x1), int(y1 - 55)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated_frame

    def calculate_velocity(self, old_pos, new_pos):
        dx = new_pos[0] - old_pos[0]
        dy = new_pos[1] - old_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        velocity = distance * self.frame_rate  # pixels per second
        #conversion_factor = 0.01  # Example: 1 pixel = 0.01 meters (Adjust based on real-world measurement)
        #velocity_m_per_sec = velocity_px_per_sec * conversion_factor
        return velocity

# Run live tracking
tracker = ObjectTracker()
tracker.run_tracking()
