import time
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

class SceneGeometry:
    def __init__(self, video_path, model_path='yolov8_aug.pt'):
        self.midas_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS', pretrained=True)
        self.midas_model.eval()
        self.yolo_model = YOLO(model_path)
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.distances = []
        self.timestamps = []
    
    def get_depth_map(self, frame):
        start_time = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = self.transform(rgb_frame).unsqueeze(0)
        with torch.no_grad():
            depth_map = self.midas_model(input_image)
        depth_map = depth_map.squeeze().cpu().numpy()
        depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        depth_map_normalized = self.normalize_depth_map(depth_map)
        end_time = time.time()
        return depth_map_normalized
    
    def normalize_depth_map(self, depth_map):
        depth_map = np.clip(depth_map, 0, np.percentile(depth_map, 95))
        depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
        depth_map = (depth_map * 255).astype(np.uint8)
        return depth_map
    
    def process_frame(self, frame, depth_map):
        start_time = time.time()
        results = self.yolo_model(frame)
        annotated_frame = frame.copy()
        if results and len(results) > 0:
            for result in results[0].boxes:
                for box in result.xyxy:
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    bbox_depth_map = depth_map[y1:y2, x1:x2]
                    if bbox_depth_map.size > 0:
                        avg_depth = np.mean(bbox_depth_map)
                        self.distances.append(avg_depth)
                        self.timestamps.append(time.time())
                        depth_text = f"Dist: {avg_depth:.2f}"
                        cv2.putText(annotated_frame, depth_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        end_time = time.time()
        return annotated_frame
    
    def process_video(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            start_time = time.time()
            future_depth = self.executor.submit(self.get_depth_map, frame)
            future_detection = self.executor.submit(self.yolo_model, frame)
            depth_map = future_depth.result()
            detection_results = future_detection.result()
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=1.0), cv2.COLORMAP_JET)
            annotated_frame = self.process_frame(frame, depth_map)
            combined_frame = cv2.addWeighted(annotated_frame, 0.7, depth_colormap, 0.3, 0)
            cv2.imshow("Depth Map Overlay", combined_frame)
            end_time = time.time()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def release_resources(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.executor.shutdown()

    def plot_distances(self):
        if not self.timestamps:
            print("No distance data to plot.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.timestamps, self.distances, 'b-', marker='o', linestyle='-', markersize=3)
        plt.xlabel('Time (s)')
        plt.ylabel('Distance (units)')
        plt.title('Distance vs Time')
        plt.grid(True)
        plt.show()

def main():
    video_path = "test (3).mp4"
    scene_geom = SceneGeometry(video_path)
    scene_geom.process_video()
    scene_geom.release_resources()
    scene_geom.plot_distances()

if __name__ == "__main__":
    main()
