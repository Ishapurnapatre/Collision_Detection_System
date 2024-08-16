import json
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def load_data(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data['lmv_x_values'], data['lmv_y_values']

def fit_polynomial_regression(x, y, degree):
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(np.array(x).reshape(-1, 1))
    model = LinearRegression()
    model.fit(x_poly, y)
    return model, poly

def predict_distance(model, poly, area):
    area_poly = poly.transform(np.array([area]).reshape(-1, 1))
    return model.predict(area_poly)[0]

def main():
    # Load the YOLOv8 model
    model = YOLO("yolov8_aug.pt")

    # Load the data from the JSON file
    json_path = "data.json"
    x_values, y_values = load_data(json_path)

    # Fit polynomial regression model
    poly_model, poly_features = fit_polynomial_regression(x_values, y_values, degree=3)

    # Open the video file
    video_path = "testvdo (7).mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Get the FPS of the original video
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    track_history = defaultdict(list)
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        frame_count += 1  # Increment frame count

        if not success:
            break

        results = model.track(frame, persist=True)

        if results and hasattr(results[0], 'boxes') and results[0].boxes:
            annotated_frame = process_frame(results, track_history, poly_model, poly_features)
        else:
            annotated_frame = frame.copy()

        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_frame(results, track_history, model, poly):
    boxes = results[0].boxes

    if hasattr(boxes, 'xywh') and hasattr(boxes, 'id') and boxes.id is not None:
        annotated_frame = results[0].plot()
        track_ids = boxes.id.int().cpu().tolist()
        boxes_xywh = boxes.xywh.cpu().numpy()
        class_names = [results[0].names[i] for i in boxes.cls.int().cpu().tolist()]

        printed_ids = set()  # Keep track of printed IDs in the current frame

        for box, track_id, class_name in zip(boxes_xywh, track_ids, class_names):
            x, y, w, h = box
            area = w * h  # Calculate area of the bounding box
            distance = predict_distance(model, poly, area)
            track_history[track_id].append((x, y))

            if track_id not in printed_ids:
                draw_bounding_box(annotated_frame, x, y, w, h, track_id, distance)
                printed_ids.add(track_id)
    else:
        annotated_frame = results[0].orig_img.copy()

    return annotated_frame

def draw_bounding_box(frame, x, y, w, h, track_id, distance):
    # Draw bounding box and track ID on the frame
    text = f"Dist: {distance:.2f}m"
    cv2.putText(frame, text, (int(x - w / 2), int(y - h / 2) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    print(f"Track ID: {track_id}, Distance: {distance:.2f}m")

if __name__ == "__main__":
    main()
