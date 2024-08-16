import cv2
from ultralytics import YOLO

class ObjectTracker:
    def __init__(self, video_path, model_path='yolov8_aug.pt'):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.vehicle_id = 1  # Initialize vehicle ID
        
    def run_tracking(self):
        if not self.cap.isOpened():
            print("Error opening video stream or file")
            return
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            
            if not success:
                break
            
            annotated_frame = self.process_frame(frame)
            
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

    def calculate_distance(self, area, class_type):
        if class_type == "TW":
            if area > 30000:
                return "<1m"
            elif area > 8000:
                return "<2m"
            elif area > 3000:
                return "<3m"
            elif area > 1000:
                return "<4m"
            elif area > 900:
                return "<5m"
            else:
                return ">=5m"
        elif class_type == "LMV":
            if area > 90000:
                return "<1m"
            elif area > 20000:
                return "<2m"
            elif area > 10000:
                return "<3m"
            elif area > 7000:
                return "<4m"
            elif area > 4000:
                return "<5m"
            else:
                return ">=5m"
        elif class_type == "HMV":
            if area > 250000:
                return "<1m"
            elif area > 70000:
                return "<2m"
            elif area > 30000:
                return "<3m"
            elif area > 20000:
                return "<4m"
            elif area > 10000:
                return "<5m"
            else:
                return ">=5m"
        else:
            return "Unknown"

    def process_frame(self, frame):
        results = self.model.track(frame, persist=True)
        annotated_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2

        if results and len(results) > 0 and results[0].boxes:
            for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                class_type = self.model.names[int(cls)]  # Assuming self.model.names gives class labels
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw blue circles at the corners
                cv2.circle(annotated_frame, (x1, y1), 5, (255, 0, 0), -1)  # Top-left corner
                cv2.circle(annotated_frame, (x2, y1), 5, (255, 0, 0), -1)  # Top-right corner
                cv2.circle(annotated_frame, (x2, y2), 5, (255, 0, 0), -1)  # Bottom-right corner
                cv2.circle(annotated_frame, (x1, y2), 5, (255, 0, 0), -1)  # Bottom-left corner
                
                # Calculate center of bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Draw center point as a red circle
                cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Calculate dimensions and area of the bounding box
                l = x2 - x1
                b = y2 - y1
                area = l * b

                # Calculate approximate distance
                distance = self.calculate_distance(area, class_type)

                # Print coordinates of the bounding box corners, center, and area
                print(f"Class: {class_type}")
                print(f"Bounding box corners: ({x1}, {y1}), ({x2}, {y1}), ({x2}, {y2}), ({x1}, {y2})")
                print(f"Center of bounding box: ({center_x}, {center_y})")
                print(f"Area of bounding box: {area}")
                print(f"Approximate distance: {distance}")
                
                # Display the area and distance on the frame
                cv2.putText(annotated_frame, f"Class: {class_type}", (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Area: {area}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Dist: {distance}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Display alert only when distance is less than 1m
                if distance == "<1m":
                    if center_x > frame_center_x and x1 > frame_center_x:
                        cv2.putText(annotated_frame, f"Alert: Vehicle ID {self.vehicle_id} is moving right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        print(f"Alert: Vehicle ID {self.vehicle_id} is moving right")
                    elif center_x < frame_center_x and x2 < frame_center_x:
                        cv2.putText(annotated_frame, f"Alert: Vehicle ID {self.vehicle_id} is moving left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        print(f"Alert: Vehicle ID {self.vehicle_id} is moving left")
                
                # Increment vehicle ID for next vehicle
                self.vehicle_id += 1
        
        return annotated_frame
    

def main():
    video_path = "Dynamic_test2.mp4"
    
    tracker = ObjectTracker(video_path)
    tracker.run_tracking()

if __name__ == "__main__":
    main()
