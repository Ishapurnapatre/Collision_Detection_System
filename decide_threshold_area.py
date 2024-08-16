# to calculate area for deciding threshold values

import cv2
from ultralytics import YOLO

class ImageObjectTracker:
    def __init__(self, image_path, model_path='yolov8_aug.pt', width=None, height=None):
        self.image_path = image_path
        self.model = YOLO(model_path)
        self.width = width
        self.height = height
        
    def process_image(self):
        image = cv2.imread(self.image_path)
        if image is None:
            print("Error loading image")
            return
        
        # Resize the image if width and height are specified
        if self.width is not None and self.height is not None:
            image = cv2.resize(image, (self.width, self.height))
        
        annotated_image = self.process_frame(image)
        
        cv2.imshow("YOLOv8 Object Detection", annotated_image)
        cv2.waitKey(0)
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
                
                # Calculate dimensions and area of the bounding box
                l = x2 - x1
                b = y2 - y1
                area = l * b

                # Print coordinates of the bounding box corners and center
                print(f"Bounding box corners: ({x1}, {y1}), ({x2}, {y1}), ({x2}, {y2}), ({x1}, {y2})")
                print(f"Center of bounding box: ({center_x}, {center_y})")
                print(f"Area of bounding box: {area}")
                
                # Display the area on the frame
                cv2.putText(annotated_frame, f"Area: {area}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated_frame
    

def main():
    image_path = "l125.jpg"
    
    # Specify desired width and height
    desired_width = 640
    desired_height = 480
    
    tracker = ImageObjectTracker(image_path, width=desired_width, height=desired_height)
    tracker.process_image()

if __name__ == "__main__":
    main()
