import cv2
from ultralytics import YOLO
import numpy as np

class ObjectTracker:
    def __init__(self, video_path, model_path='yolov8_aug.pt'):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        
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

    def process_frame(self, frame):
        results = self.model.track(frame, persist=True)
        annotated_frame = frame.copy()
        
        if results and len(results) > 0 and results[0].boxes:
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                
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
                
                # Calculate d
                d = np.sqrt(center_x**2 + center_y**2)
                
                # Display the value of d on the frame
                cv2.putText(annotated_frame, f'd: {d:.2f}', (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Print coordinates of the bounding box corners and center
                print(f"Bounding box corners: ({x1}, {y1}), ({x2}, {y1}), ({x2}, {y2}), ({x1}, {y2})")
                print(f"Center of bounding box: ({center_x}, {center_y})")
                print(f"Value of d: {d:.2f}")
        
        return annotated_frame
    

def main():
    video_path = "Dynamic_test2.mp4"
    
    tracker = ObjectTracker(video_path)
    tracker.run_tracking()

if __name__ == "__main__":
    main()
