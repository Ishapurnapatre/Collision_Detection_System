import cv2
import numpy as np
import json
import time
from collections import deque, Counter
from collections import defaultdict
from ultralytics import YOLO
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def predict_distance(file_path):
    
    with open(file_path, 'r') as file:
        data = json.load(file)

    tw_x_values1 = np.array(data['tw_x_values1']).reshape(-1, 1)
    tw_y_values1 = np.array(data['tw_y_values1'])
    tw_x_values2 = np.array(data['tw_x_values2']).reshape(-1, 1)
    tw_y_values2 = np.array(data['tw_y_values2'])

    poly_features1 = PolynomialFeatures(degree=3)
    x_poly1 = poly_features1.fit_transform(tw_x_values1)
    model1 = LinearRegression()
    model1.fit(x_poly1, tw_y_values1)

    poly_features2 = PolynomialFeatures(degree=3)
    x_poly2 = poly_features2.fit_transform(tw_x_values2)
    model2 = LinearRegression()
    model2.fit(x_poly2, tw_y_values2)

    return model1, poly_features1, model2, poly_features2

def predict_velocity(track_id, predicted_y_scalar, current_time, track_history):
    
    predicted_distance_history = defaultdict(deque)
    velocity_history = defaultdict(deque)
    acceleration_history = defaultdict(deque)

    start_time = defaultdict(lambda: time.time())

    velocity = 0.0
    acceleration = 0.0

    e_time = 0

    if len(predicted_distance_history[track_id]) >= 10:
        previous_distances = list(predicted_distance_history[track_id])[-10:]
        previous_times = list(track_history[track_id])[-10:]

        if len(previous_distances) == 10 and len(previous_times) == 10:
            distance_diff = predicted_y_scalar - previous_distances[0]
            time_diff = current_time - previous_times[0]

            if time_diff > 0:
                elapsed_time = current_time - start_time[track_id]
                e_time = elapsed_time
                velocity = abs(distance_diff) / elapsed_time
                acceleration = abs(velocity - velocity[-1]) / elapsed_time

                velocity_history[track_id].append(velocity)
                acceleration_history[track_id].append(acceleration)

            else:
                velocity = 0.0
                acceleration = 0.0

        predicted_distance_history[track_id].popleft()
        track_history[track_id].popleft()

    predicted_distance_history[track_id].append(predicted_y_scalar)
    track_history[track_id].append(current_time)

    return e_time

def track(model, json_path, video_path):

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    if not cap.isOpened():
        print("Error opening video stream or file")
        exit()

    track_history = defaultdict(deque)

    while cap.isOpened():
        success, frame = cap.read()
        frame_count += 1

        if success:
            results = model.track(frame, persist=True)

            if results and hasattr(results[0], 'boxes') and results[0].boxes:
                boxes = results[0].boxes

                if hasattr(boxes, 'xywh') and hasattr(boxes, 'id') and boxes.id is not None:
                    annotated_frame = results[0].plot()

                    track_ids = boxes.id.int().cpu().tolist()
                    boxes_xywh = boxes.xywh.cpu().numpy()
                    class_names = [results[0].names[i] for i in boxes.cls.int().cpu().tolist()]

                    printed_ids = set()

                    for box, track_id, class_name in zip(boxes_xywh, track_ids, class_names):
                        x, y, w, h = box
                        area = w * h
                        track_history[track_id].append((x, y))

                        model1, poly_features1, model2, poly_features2 = predict_distance(json_path)

                        if area > 46200:
                            x_poly1_pred = poly_features1.transform([[area]])
                            predicted_y = model1.predict(x_poly1_pred)
                        else:
                            x_poly2_pred = poly_features2.transform([[area]])
                            predicted_y = model2.predict(x_poly2_pred)

                        predicted_y_scalar = predicted_y[0] if isinstance(predicted_y, np.ndarray) else predicted_y

                        current_time = time.time()

                        elapsed_time = predict_velocity(track_id, predicted_y_scalar, current_time, track_history)

                        cv2.putText(annotated_frame, f"Dist: {predicted_y[0]:.2f}", 
                                    (int(x - w/2), int(y - h/2) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        if track_id not in printed_ids:
                            print(f"Track ID: {track_id}, Area: {area}, Predicted Distance: {predicted_y[0]:.2f}. Time: {elapsed_time:.2f}")
                            printed_ids.add(track_id)

                else:
                    annotated_frame = frame.copy()
            else:
                annotated_frame = frame.copy()

            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    
    model = YOLO("yolov8_aug.pt")
    json_path = ('datasplit.json')
    video_path = "testvdo (5).mp4"
    
    track(model, json_path, video_path)

if __name__ == "__main__":
    main()
