import cv2
from ultralytics import YOLO
import numpy as np

class ObjectTracker:
    def __init__(self, video_path, model_path='yolov8_aug.pt', cm_to_pixels=10):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.prev_frame = None
        self.prev_centers = []
        self.colors = np.random.randint(0, 255, (100, 3))  # Random colors for tracking lines
        self.mask = None
        self.cm_to_pixels = cm_to_pixels  # Conversion factor from cm to pixels

    def run_tracking(self):
        if not self.cap.isOpened():
            print("Error opening video stream or file")
            return
        
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read the first frame")
            return

        self.prev_frame = frame.copy()
        self.prev_centers = []
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

    def process_frame(self, frame, first_frame=False):
        results = self.model.track(frame, persist=True)
        annotated_frame = frame.copy()

        if first_frame:
            self.prev_centers = []
            self.mask = np.zeros_like(frame)  # Reset mask for the first frame

        if results and len(results) > 0 and results[0].boxes:
            new_centers = []
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.cpu().numpy())

                # Draw bounding box and corners
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(annotated_frame, (x1, y1), 5, (255, 0, 0), -1)  # Top-left
                cv2.circle(annotated_frame, (x2, y1), 5, (255, 0, 0), -1)  # Top-right
                cv2.circle(annotated_frame, (x2, y2), 5, (255, 0, 0), -1)  # Bottom-right
                cv2.circle(annotated_frame, (x1, y2), 5, (255, 0, 0), -1)  # Bottom-left
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                new_centers.append((center_x, center_y))
                
                # Draw center of bounding box
                cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Draw 1 cm box around the detected vehicle
                box_size = self.cm_to_pixels
                top_left = (center_x - box_size // 2, center_y - box_size // 2)
                bottom_right = (center_x + box_size // 2, center_y + box_size // 2)
                cv2.rectangle(annotated_frame, top_left, bottom_right, (255, 0, 255), 2)
            
            # Calculate and draw motion lines and analyze direction
            if not first_frame and self.prev_centers:
                for prev_center, new_center, color in zip(self.prev_centers, new_centers, self.colors):
                    prev_center_x, prev_center_y = prev_center
                    new_center_x, new_center_y = new_center
                    cv2.line(self.mask, (prev_center_x, prev_center_y), (new_center_x, new_center_y), color.tolist(), 2)
                    
                    # Determine if the vehicle is moving
                    if self.is_moving(prev_center, new_center):
                        # Analyze direction
                        dx = new_center_x - prev_center_x
                        dy = new_center_y - prev_center_y
                        
                        if abs(dx) > abs(dy):  # Mainly horizontal movement
                            if dx > 0:
                                direction = "Right"
                            else:
                                direction = "Left"
                        else:  # Mainly vertical movement
                            if dy > 0:
                                direction = "Down"
                            else:
                                direction = "Up"
                        
                        # Draw the direction on the frame
                        cv2.putText(annotated_frame, direction, (new_center_x, new_center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color.tolist(), 2)
                    else:
                        cv2.putText(annotated_frame, "Not Moving", (new_center_x, new_center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color.tolist(), 2)
                    
                    # Draw new center
                    cv2.circle(annotated_frame, (new_center_x, new_center_y), 5, color.tolist(), -1)
            
            # Combine the frame and the mask
            img = cv2.add(annotated_frame, self.mask)
            self.prev_centers = new_centers
        
        return img

    def is_moving(self, prev_center, new_center):
        """ Check if the movement is significant or within a small threshold. """
        prev_x, prev_y = prev_center
        new_x, new_y = new_center
        
        # Define the threshold (in pixels)
        threshold = self.cm_to_pixels
        
        # Check if the change in position is within the threshold
        if abs(new_x - prev_x) < threshold and abs(new_y - prev_y) < threshold:
            return False
        return True
    

def main():
    video_path = "test (3).mp4"
    
    tracker = ObjectTracker(video_path)
    tracker.run_tracking()

if __name__ == "__main__":
    main()
