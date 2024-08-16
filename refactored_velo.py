# refactored code

from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd

class ObjectTracker:
    def __init__(self, video_path, output_video_path, model_path='yolov8_aug.pt'):
        self.video_path = video_path
        self.output_video_path = output_video_path
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = None
        self.frame_count = 0
        
        self.track_history = defaultdict(list)
        self.track_velocities = defaultdict(list)
        self.track_distances = defaultdict(list)
        self.first_frame_id = defaultdict(int)
        self.track_classes = defaultdict(str)
        
        self.prev_avg_positions = defaultdict(lambda: (0, 0))  # Default previous average positions

    def run_tracking(self):
        if not self.cap.isOpened():
            print("Error opening video stream or file")
            return
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_video_path, fourcc, self.frame_rate, (self.frame_width, self.frame_height))
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            self.frame_count += 1
            
            if not success:
                break
            
            annotated_frame = self.process_frame(frame)
            
            self.out.write(annotated_frame)
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        
        self.save_tracking_data()

    def process_frame(self, frame):
        results = self.model.track(frame, persist=True)
        annotated_frame = frame.copy()
        
        if results and hasattr(results[0], 'boxes') and results[0].boxes:
            boxes = results[0].boxes
            
            if hasattr(boxes, 'xywh') and hasattr(boxes, 'id') and boxes.id is not None:
                annotated_frame = results[0].plot()
                self.update_tracks(boxes, annotated_frame)
        
        return annotated_frame
    
    def update_tracks(self, boxes, annotated_frame):
        track_ids = boxes.id.int().cpu().tolist()
        boxes_xywh = boxes.xywh.cpu().numpy()
        class_names = [self.model.names[i] for i in boxes.cls.int().cpu().tolist()]
        
        for box, track_id, class_name in zip(boxes_xywh, track_ids, class_names):
            x, y, w, h = box
            track = self.track_history[track_id]
            
            if track_id not in self.first_frame_id:
                self.initialize_track(track_id, class_name)
            
            if track:
                self.update_track_metrics(track_id, x, y)
                self.update_average_positions(track_id, x, y)
                
            else:
                self.initialize_track_metrics(track_id)
            
            track.append((x, y))
            self.display_velocity_text(annotated_frame, track_id, self.track_velocities[track_id][-1], x, y)
            
    def initialize_track(self, track_id, class_name):
        self.first_frame_id[track_id] = self.frame_count
        self.track_classes[track_id] = class_name
    
    def initialize_track_metrics(self, track_id):
        self.track_velocities[track_id].append(0.0)
        self.track_distances[track_id].append(0.0)
    
    def update_track_metrics(self, track_id, x, y):
        if track_id in self.track_history:
            prev_x, prev_y = self.track_history[track_id][-1]
            distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
            
            # Update cumulative distance only if movement is significant
            min_movement_threshold = 1.0  # Adjust as needed
            if distance > min_movement_threshold:
                cumulative_distance = self.track_distances[track_id][-1] + distance if self.track_distances[track_id] else distance
            else:
                cumulative_distance = self.track_distances[track_id][-1] if self.track_distances[track_id] else 0.0
            
            time_elapsed = (self.frame_count - self.first_frame_id[track_id] + 1) / self.frame_rate
            cumulative_velocity = cumulative_distance / time_elapsed if time_elapsed > 0 else 0
            
            self.track_velocities[track_id].append(cumulative_velocity)
            self.track_distances[track_id].append(cumulative_distance)
            
            self.track_history[track_id].append((x, y))
        else:
            self.track_history[track_id] = [(x, y)]
            self.track_distances[track_id] = [0.0]
            self.track_velocities[track_id] = [0.0]
            self.first_frame_id[track_id] = self.frame_count
    
    def update_average_positions(self, track_id, x, y):
        prev_avg_x, prev_avg_y = self.prev_avg_positions[track_id]
        avg_x = (prev_avg_x + x) / 2
        avg_y = (prev_avg_y + y) / 2
        
        time_elapsed = (self.frame_count - self.first_frame_id[track_id] + 1) / self.frame_rate
        cumulative_velocity_avg = self.calculate_velocity_avg(prev_avg_x, prev_avg_y, avg_x, avg_y, time_elapsed)
        self.track_velocities[track_id].append(cumulative_velocity_avg)
        
        self.prev_avg_positions[track_id] = (avg_x, avg_y)

    def calculate_velocity_avg(self, prev_avg_x, prev_avg_y, avg_x, avg_y, time_elapsed):
        distance = np.sqrt((avg_x - prev_avg_x) ** 2 + (avg_y - prev_avg_y) ** 2)
        return distance / time_elapsed if time_elapsed > 0 else 0
    
    def display_velocity_text(self, annotated_frame, track_id, velocity, x, y):
        text = f"Velocity: {velocity:.2f} px/s"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_width, text_height = text_size[0]
        cv2.rectangle(annotated_frame, (int(x), int(y - text_height - 6)), (int(x + text_width), int(y)), (0, 0, 0), -1)
        cv2.putText(annotated_frame, text, (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Calculate and display distance
        if track_id in self.track_distances:
            distance = self.track_distances[track_id][-1]  # Get the latest recorded distance
            distance_text = f"Distance: {distance:.2f} px"
            cv2.putText(annotated_frame, distance_text, (int(x), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mark the closest point (assuming (x, y) is the point of interest)
        cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green circle at the tracking point


    def save_tracking_data(self):
        max_frame = self.frame_count
        frame_data = {'Frame': list(range(1, max_frame + 1))}
        
        for id_, first_frame in self.first_frame_id.items():
            class_name = self.track_classes[id_]
            velocity_col = f'ID {id_} {class_name} Velocity (px/s)'
            distance_col = f'ID {id_} {class_name} Distance (px)'
            
            velocities = [None] * (first_frame - 1) + self.track_velocities[id_]
            distances = [None] * (first_frame - 1) + self.track_distances[id_]
            
            velocities.extend([None] * (max_frame - len(velocities)))
            distances.extend([None] * (max_frame - len(distances)))
            
            frame_data[velocity_col] = velocities
            frame_data[distance_col] = distances
        
        df = pd.DataFrame(frame_data)
        excel_path = 'Dynamic_test2_velocity_with_id1.xlsx'
        df.to_excel(excel_path, index=False)

def main():
    video_path = "Dynamic_test2.mp4"
    output_video_path = "Output_Dynamic_test15.mp4"
    
    tracker = ObjectTracker(video_path, output_video_path)
    tracker.run_tracking()

if __name__ == "__main__":
    main()


# main - __init__- run_tracker(loop for video capture and storing) - process_frame(bounding box) - update_tracks(tracks id, dist, velo) - calculate_velocity_avg - display_velocity_text - save_tracking_data()