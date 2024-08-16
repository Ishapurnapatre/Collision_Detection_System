import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque, Counter

class ObjectTracker:
    def __init__(self, video_path, model_path='yolov8_aug.pt', max_history_len=5, max_direction_len=5, movement_threshold=5, frame_rate=30):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.prev_frame = None
        self.vehicle_histories = {}
        self.vehicle_directions = {}
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
        self.vehicle_histories = {}
        self.vehicle_directions = {}
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

        if first_frame:
            self.vehicle_histories = {}  # Reset histories for the first frame
            self.vehicle_directions = {}  # Reset directions for the first frame
            self.mask = np.zeros_like(frame)  # Reset mask for the first frame

        if results and len(results) > 0 and results[0].boxes:
            vehicle_ids = []
            vehicle_centers = {}

            for idx, box in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = map(int, box.cpu().numpy())

                # Draw bounding box and corners
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(annotated_frame, (x1, y1), 5, (255, 0, 0), -1)  # Top-left
                cv2.circle(annotated_frame, (x2, y1), 5, (255, 0, 0), -1)  # Top-right
                cv2.circle(annotated_frame, (x2, y2), 5, (255, 0, 0), -1)  # Bottom-right
                cv2.circle(annotated_frame, (x1, y2), 5, (255, 0, 0), -1)  # Bottom-left
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                vehicle_ids.append(idx)
                
                # Get positions around the center
                positions_around_center = self.get_around_center(center_x, center_y, 10)
                avg_x, avg_y = self.average_position(positions_around_center)
                vehicle_centers[idx] = (avg_x, avg_y)

                # Draw center of bounding box
                cv2.circle(annotated_frame, (avg_x, avg_y), 5, (0, 0, 255), -1)

                # Plot the 10 pixels around the center
                for px, py in positions_around_center:
                    cv2.circle(annotated_frame, (px, py), 3, (255, 255, 0), -1)

            # Update histories and directions for each vehicle
            for vehicle_id in vehicle_ids:
                new_center = vehicle_centers[vehicle_id]
                if vehicle_id not in self.vehicle_histories:
                    self.vehicle_histories[vehicle_id] = deque(maxlen=self.max_history_len)
                    self.vehicle_directions[vehicle_id] = deque(maxlen=self.max_direction_len)
                
                # Add new centers to history
                self.vehicle_histories[vehicle_id].append(new_center)

                # Initialize variables for direction analysis
                prev_center_x = prev_center_y = new_center_x = new_center_y = None

                # Calculate and analyze movement direction if history is sufficient
                if len(self.vehicle_histories[vehicle_id]) > 1:
                    prev_center = self.vehicle_histories[vehicle_id][-2]
                    prev_center_x, prev_center_y = prev_center
                    new_center_x, new_center_y = new_center
                    cv2.line(self.mask, (prev_center_x, prev_center_y), (new_center_x, new_center_y), self.colors[vehicle_id % 100].tolist(), 2)

                    # Analyze movement direction
                    dx = new_center_x - prev_center_x
                    dy = new_center_y - prev_center_y
                    
                    if abs(dx) > self.movement_threshold or abs(dy) > self.movement_threshold:
                        if abs(dx) > abs(dy):  # Mainly horizontal movement
                            direction = "Right" if dx > 0 else "Left"
                        else:  # Mainly vertical movement
                            direction = "Down" if dy > 0 else "Up"
                    else:
                        direction = "Center"
                    
                    # Add direction to history
                    self.vehicle_directions[vehicle_id].append(direction)

                    # Determine and display the most common direction
                    if len(self.vehicle_directions[vehicle_id]) == self.max_direction_len:
                        most_common_direction = Counter(self.vehicle_directions[vehicle_id]).most_common(1)[0][0]
                        cv2.putText(annotated_frame, f"Vehicle {vehicle_id} Direction: {most_common_direction}", (10, 30 + 30 * vehicle_id), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                # Ensure new_center_x and new_center_y are set before drawing
                if new_center_x is not None and new_center_y is not None:
                    # Draw new center
                    cv2.circle(annotated_frame, (new_center_x, new_center_y), 5, self.colors[vehicle_id % 100].tolist(), -1)

                    # Calculate and display velocity
                    if len(self.vehicle_histories[vehicle_id]) > 1:
                        prev_center = self.vehicle_histories[vehicle_id][-2]
                        velocity = self.calculate_velocity(prev_center, new_center)
                        cv2.putText(annotated_frame, f"Velocity: {velocity:.2f} px/s", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Combine the frame and the mask
        img = cv2.add(annotated_frame, self.mask)
    
        return img

def main():
    video_path = "testvdo (9).mp4"
    
    tracker = ObjectTracker(video_path)
    tracker.run_tracking()

if __name__ == "__main__":
    main()
