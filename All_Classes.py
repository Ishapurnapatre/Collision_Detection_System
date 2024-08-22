import cv2
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from ultralytics import YOLO

def load_yolo_model():
    model_path = "yolov8_aug.pt"
    model = YOLO(model_path)
    return model

def calculate_distance_models(class_name):
    with open("distance.json", 'r') as file:
        data = json.load(file)

    # Convert class name to lowercase to match JSON keys
    class_name_lower = class_name.lower()

    if class_name_lower in data:
        values = data[class_name_lower]
    else:
        raise KeyError(f"Class name '{class_name}' not found in JSON file.")

    x_values1 = np.array(values['x_values1']).reshape(-1, 1)
    y_values1 = np.array(values['y_values1'])
    x_values2 = np.array(values['x_values2']).reshape(-1, 1)
    y_values2 = np.array(values['y_values2'])

    poly_features1 = PolynomialFeatures(degree=3)
    x_poly1 = poly_features1.fit_transform(x_values1)
    model1 = LinearRegression()
    model1.fit(x_poly1, y_values1)

    poly_features2 = PolynomialFeatures(degree=3)
    x_poly2 = poly_features2.fit_transform(x_values2)
    model2 = LinearRegression()
    model2.fit(x_poly2, y_values2)

    return model1, poly_features1, model2, poly_features2

def calculate_velocity(distance1, distance2, time_elapsed):
    velocity = (abs(distance2 - distance1)) / time_elapsed if time_elapsed > 0 else 0
    print(f"Calculated Velocity: {velocity:.2f} m/s")
    return velocity

def calculate_acceleration(velocity1, velocity2, time_elapsed):
    acceleration = (abs(velocity2 - velocity1)) / time_elapsed if time_elapsed > 0 else 0
    print(f"Calculated Acceleration: {acceleration:.2f} m/s²")
    return acceleration

def generate_random_points(center, radius=1, num_points=10):
    points = center + np.random.uniform(-radius, radius, (num_points, 2))
    print(f"Generated Points Around Center: {points}")
    return points

def determine_direction(prev_points, curr_points):
    dx = curr_points[:, 0] - prev_points[:, 0]
    dy = curr_points[:, 1] - prev_points[:, 1]

    directions = []
    for delta_x, delta_y in zip(dx, dy):
        if abs(delta_x) > abs(delta_y):
            if delta_x > 0:
                directions.append("Right")
            else:
                directions.append("Left")
        else:
            if delta_y > 0:
                directions.append("Down")
            else:
                directions.append("Up")

    direction = "Center" if len(set(directions)) > 1 else max(set(directions), key=directions.count)
    print(f"Determined Direction: {direction}")
    return direction

def track_and_analyze(model, video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    track_history = {}
    velocity_history = {}
    acceleration_history = {}
    previous_times = {}
    previous_centers = {}

    threshold = 15  # Example threshold value

    while cap.isOpened():
        success, frame = cap.read()
        frame_count += 1

        if not success:
            break

        results = model.track(frame, persist=True)
        if results and hasattr(results[0], 'boxes') and results[0].boxes:
            boxes = results[0].boxes

            if hasattr(boxes, 'xywh') and hasattr(boxes, 'id') and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                boxes_xywh = boxes.xywh.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                class_names = [results[0].names[i] for i in boxes.cls.int().cpu().tolist()]

                for box, track_id, confidence, class_name in zip(boxes_xywh, track_ids, confidences, class_names):
                    x, y, w, h = box
                    center = np.array([x, y])
                    area = w * h

                    # Get models specific to the detected class
                    model1, poly_features1, model2, poly_features2 = calculate_distance_models(class_name)

                    if area > 46200:
                        x_poly1_pred = poly_features1.transform([[area]])
                        predicted_y = model1.predict(x_poly1_pred)
                    else:
                        x_poly2_pred = poly_features2.transform([[area]])
                        predicted_y = model2.predict(x_poly2_pred)

                    distance = predicted_y[0] if isinstance(predicted_y, np.ndarray) else predicted_y
                    print(f"Track ID: {track_id}, Class: {class_name}, Confidence: {confidence:.2f}, Distance: {distance:.2f}m")
                
                    if track_id not in track_history:
                        track_history[track_id] = []
                        velocity_history[track_id] = []
                        acceleration_history[track_id] = []
                        previous_times[track_id] = []
                        previous_centers[track_id] = center

                    track_history[track_id].append((x, y, w, h, distance))
                    previous_times[track_id].append(time.time())

                    if len(track_history[track_id]) >= 10 and len(previous_times[track_id]) >= 10:
                        prev_frame_data = track_history[track_id][-10]
                        prev_distance = prev_frame_data[4]
                        time_elapsed = previous_times[track_id][-1] - previous_times[track_id][-10]

                        velocity = calculate_velocity(prev_distance, distance, time_elapsed)
                        velocity_history[track_id].append(velocity)

                        if len(velocity_history[track_id]) >= 2:
                            prev_velocity = velocity_history[track_id][-2]
                            acceleration = calculate_acceleration(prev_velocity, velocity, time_elapsed)
                        else:
                            acceleration = 0
                        acceleration_history[track_id].append(acceleration)

                        prev_points = generate_random_points(previous_centers[track_id])
                        curr_points = generate_random_points(center)
                        direction = determine_direction(prev_points, curr_points)
                        previous_centers[track_id] = center

                        product = distance * velocity
                        color = (0, 0, 255) if product > threshold else (255, 255, 255)  # Red if above threshold, white otherwise

                        text_y = int(y - h/2 - 110)
                        cv2.putText(frame, f"Track ID: {track_id}", (int(x - w/2), text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        text_y += 15
                        cv2.putText(frame, f"Confidence: {confidence:.2f}", (int(x - w/2), text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        text_y += 15
                        cv2.putText(frame, f"Class: {class_name}", (int(x - w/2), text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        text_y += 15
                        cv2.putText(frame, f"Distance: {distance:.2f}m", (int(x - w/2), text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        text_y += 15
                        cv2.putText(frame, f"Elapsed Time: {time_elapsed:.2f}s", (int(x - w/2), text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        text_y += 15
                        cv2.putText(frame, f"Velocity: {velocity:.2f}m/s", (int(x - w/2), text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        text_y += 15
                        cv2.putText(frame, f"Acceleration: {acceleration:.2f}m/s²", (int(x - w/2), text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        text_y += 15
                        cv2.putText(frame, f"Direction: {direction}", (int(x - w/2), text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                        cv2.rectangle(frame, (int(x - w/2), int(y - h/2)),
                                      (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)

        cv2.imshow("YOLO Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    plot_combined_velocity_acceleration(velocity_history, acceleration_history)

def plot_combined_velocity_acceleration(velocity_history, acceleration_history):
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    for track_id, velocities in velocity_history.items():
        plt.plot(velocities, label=f'Vehicle {track_id}')
    plt.xlabel('Frame')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity of All Vehicles')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    for track_id, accelerations in acceleration_history.items():
        plt.plot(accelerations, label=f'Vehicle {track_id}')
    plt.xlabel('Frame')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Acceleration of All Vehicles')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

video_path = "testvdo (6).mp4"
model = load_yolo_model()

track_and_analyze(model, video_path)
