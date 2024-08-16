from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Tracker:
    def __init__(self, video_path, model_path, thresholds_path):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.cap = None
        self.frame_rate = 0
        self.frame_width = 0
        self.frame_height = 0
        self.track_history = defaultdict(lambda: [])
        self.track_velocities = defaultdict(list)
        self.track_distances = defaultdict(list)
        self.first_frame_id = defaultdict(int)
        self.track_classes = defaultdict(str)
        self.frame_count = 0
        self.fps = 0  # Initialize FPS to 0

        # Load video and initialize attributes
        self.load_video()
        self.fps = self.get_video_fps()  # Get the original FPS of the video

        # Load thresholds from JSON file
        with open(thresholds_path, 'r') as file:
            self.thresholds = json.load(file)

        # Initialize Matplotlib figure
        self.fig, self.ax = plt.subplots()
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100)

    def load_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print("Error opening video stream or file")
            exit()

        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_video_fps(self):
        if self.cap is None or not self.cap.isOpened():
            print("Video capture is not initialized or opened")
            return None
        return self.cap.get(cv2.CAP_PROP_FPS)

    def process_frame(self, frame):
        annotated_frame = frame.copy()  # Default to returning original frame if no detections

        results = self.model.track(frame, persist=True)

        if results and hasattr(results[0], 'boxes') and results[0].boxes:
            boxes = results[0].boxes

            if hasattr(boxes, 'xywh') and hasattr(boxes, 'id') and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                class_names = [self.model.names[i] for i in boxes.cls.int().cpu().tolist()]

                self.update_tracking_data(boxes, track_ids, class_names)

                annotated_frame = results[0].plot()  # Update annotated frame with detection plot

                for box, track_id, cls in zip(boxes.xywh.cpu().numpy(), track_ids, boxes.cls):
                    x, y, w, h = box
                    x1, y1 = int(x - w / 2), int(y - h / 2)
                    x2, y2 = int(x + w / 2), int(y + h / 2)
                    class_type = self.model.names[int(cls)]
                    area = w * h
                    distance = self.calculate_distance(area, class_type)
                    
                    # Get cumulative velocity
                    cumulative_velocity = self.track_velocities[track_id][-1] if self.track_velocities[track_id] else 0.0

                    text_distance = f"Distance: {distance}"
                    text_velocity = f"Velocity: {cumulative_velocity:.2f} px/s"
                    box_color = (163, 187, 163)  # Color: 73BBA3 in BGR format

                    # Calculate text sizes
                    text_size_distance, _ = cv2.getTextSize(text_distance, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    text_size_velocity, _ = cv2.getTextSize(text_velocity, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                    # Display velocity
                    cv2.rectangle(annotated_frame, (x1, y1 - text_size_velocity[1] - 40), (x1 + text_size_velocity[0] + 10, y1 - 40), box_color, -1)
                    cv2.putText(annotated_frame, text_velocity, (x1 + 5, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                    # Display distance below velocity
                    cv2.rectangle(annotated_frame, (x1, y1 - text_size_distance[1] - 20), (x1 + text_size_distance[0] + 10, y1 - 20), box_color, -1)
                    cv2.putText(annotated_frame, text_distance, (x1 + 5, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return annotated_frame

    def update_tracking_data(self, boxes, track_ids, class_names):
        current_frame_ids = set()

        for box, track_id, class_name in zip(boxes.xywh.cpu().numpy(), track_ids, class_names):
            x, y, w, h = box
            track = self.track_history[track_id]
            current_frame_ids.add(track_id)

            if track_id not in self.first_frame_id:
                self.first_frame_id[track_id] = self.frame_count
                self.track_classes[track_id] = class_name

            if track:
                prev_x, prev_y = track[-1]
                distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                cumulative_distance = self.track_distances[track_id][-1] + distance if self.track_distances[track_id] else distance
                self.track_distances[track_id].append(cumulative_distance)
                time_elapsed = (self.frame_count - self.first_frame_id[track_id] + 1) / self.fps
                cumulative_velocity = cumulative_distance / time_elapsed if time_elapsed > 0 else 0
                self.track_velocities[track_id].append(cumulative_velocity)
            else:
                cumulative_velocity = 0.0
                self.track_velocities[track_id].append(cumulative_velocity)
                self.track_distances[track_id].append(0.0)

            track.append((x, y))

    def calculate_distance(self, area, class_type):
        class_thresholds = self.thresholds.get(class_type, {})
        
        for threshold, distance in sorted(class_thresholds.items(), reverse=True, key=lambda x: int(x[0]) if isinstance(x[0], str) and x[0].isdigit() else float('inf')):
            if threshold == 'default':
                continue  # Skip the 'default' key
            
            if area > int(threshold):
                return distance
        
        return class_thresholds.get('default', "Unknown")

    def update_plot(self, frame):
        self.ax.clear()
        self.ax.set_title('Vehicle Velocities')
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Velocity (px/s)')

        for track_id, velocities in self.track_velocities.items():
            if len(velocities) > 1:
                self.ax.plot(velocities, label=f'Vehicle {track_id}')

        self.ax.legend()

    def run_tracking(self):
        if self.cap is None or not self.cap.isOpened():
            print("Video capture is not initialized or opened. Exiting.")
            return
        
        plt.ion()
        while self.cap.isOpened():
            success, frame = self.cap.read()
            self.frame_count += 1

            if success:
                start_time = cv2.getTickCount()  # Start time for processing
                annotated_frame = self.process_frame(frame)
                end_time = cv2.getTickCount()    # End time for processing
                process_time = (end_time - start_time) / cv2.getTickFrequency()  # Time taken to process frame
                delay = max(int((1 / self.fps - process_time) * 1000), 1)  # Calculate delay to maintain original FPS

                cv2.imshow("YOLOv8 Tracking", annotated_frame)
                
                # Update the plot
                self.update_plot(self.frame_count)
                plt.pause(0.001)

                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()

def main():
    video_path = "test (2).mp4"
    model_path = "yolov8_aug.pt"
    thresholds_path = "thresholds.json"

    tracker = Tracker(video_path, model_path, thresholds_path)
    tracker.run_tracking()

if __name__ == "__main__":
    main()
