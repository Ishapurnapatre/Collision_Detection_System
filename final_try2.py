import cv2
import numpy as np
import json
import time
import matplotlib.pyplot as plt
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

def track(model, json_path, video_path):

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video stream or file")
        exit()

    track_history = defaultdict(deque)
    # predicted_distance_history = defaultdict(deque)
    # velocity_history = defaultdict(deque)
    # acceleration_history = defaultdict(deque)
    vehicle_directions = defaultdict(deque)
    frame_count = 0
    
    # colors = np.random.randint(0, 255, (100, 3))

    # Dictionaries to store the start time for each track ID
    # start_time = defaultdict(lambda: time.time())

    # Lists for plotting (only for track ID 3)
    velocities_track_3 = []
    accelerations_track_3 = []
    elapsed_times_track_3 = []

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

                    for box, track_id, class_names in zip(boxes_xywh, track_ids, class_names):
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

                        mask = np.zeros_like(frame)
                        prev_frame = frame.copy()

                        if track_id not in track_history:
                            track_history[track_id] = deque(maxlen=max_history_len)
                            vehicle_directions[track_id] = deque(maxlen=max_direction_len)

                        distance = 10
                        max_history_len=5, 
                        max_direction_len=5, 
                        movement_threshold=5
                        positions = [
                        (x + distance, y),
                        (x - distance, y),
                        (x, y + distance),
                        (x, y - distance),
                        (x + distance, y + distance),
                        (x + distance, y - distance),
                        (x - distance, y + distance),
                        (x - distance, y - distance),
                        (x + distance // 2, y + distance // 2),
                        (x - distance // 2, y - distance // 2)
                        ]

                        avg_x = int(sum(x for x, y in positions) / len(positions))
                        avg_y = int(sum(y for x, y in positions) / len(positions))

                        # Analyze movement direction if history is sufficient
                        if len(track_history[track_id]) > 1:
                            prev_center = track_history[track_id][-2]
                            prev_center_x, prev_center_y = prev_center
                            dx = x - prev_center_x
                            dy = y - prev_center_y

                            # Draw line for movement direction
                            # cv2.line(mask, (prev_center_x, prev_center_y), (x, y), colors[track_id % 100].tolist(), 2)

                            # Analyze movement direction
                            if abs(dx) > movement_threshold or abs(dy) > movement_threshold:
                                if abs(dx) > abs(dy):  # Mainly horizontal movement
                                    direction = "Right" if dx > 0 else "Left"
                                else:  # Mainly vertical movement
                                    direction = "Down" if dy > 0 else "Up"
                            else:
                                direction = "Center"
                            
                            # Add direction to history
                            vehicle_directions[track_id].append(direction)

                            # Determine and display the most common direction
                            if len(vehicle_directions[track_id]) == max_direction_len:
                                most_common_direction = Counter(vehicle_directions[track_id]).most_common(1)[0][0]
                            else:
                                most_common_direction = direction  # Default to current direction if not enough history
                        else:
                            most_common_direction = "N/A"

                        # # Initialize velocity and acceleration
                        # velocity = 0.0
                        # acceleration = 0.0

                        # if len(predicted_distance_history[track_id]) >= 10:
                        #     # Calculate velocity and acceleration
                        #     previous_distances = list(predicted_distance_history[track_id])[-10:]
                        #     previous_times = list(track_history[track_id])[-10:]

                        #     elapsed_time = current_time - start_time[track_id]

                        #     # Ensure there are 10 points to calculate
                        #     if len(previous_distances) == 10 and len(previous_times) == 10:
                        #         distance_diff = predicted_y_scalar - previous_distances[0]
                        #         time_diff = current_time - previous_times[0]

                        #         # Calculate velocity and acceleration
                        #         if time_diff > 0:
                        #             velocity = abs(distance_diff) / time_diff
                        #             velocities = [distance_diff / (current_time - t) for t in previous_times[1:]]
                        #             acceleration = abs(velocity - velocities[-1]) / (current_time - previous_times[-1]) if len(velocities) > 1 else 0

                        #             # Update histories
                        #             velocity_history[track_id].append(velocity)
                        #             acceleration_history[track_id].append(acceleration)
                        #         else:
                        #             velocity = 0.0
                        #             acceleration = 0.0

                        #     # Maintain history of distances and times
                        #     predicted_distance_history[track_id].popleft()
                        #     track_history[track_id].popleft()

                        # # Append new data
                        # predicted_distance_history[track_id].append(predicted_y_scalar)
                        # track_history[track_id].append(current_time)

                        # # Collect data only for track ID 3
                        # if track_id == 3:
                        #     if velocity_history[track_id]:
                        #         velocities_track_3.append(velocity_history[track_id][-1])
                        #         elapsed_times_track_3.append(current_time - start_time[track_id])
                        #     if acceleration_history[track_id]:
                        #         accelerations_track_3.append(acceleration_history[track_id][-1])

                        # # Calculate elapsed time from the start of the video
                        # elapsed_time = current_time - start_time[track_id]
                        # elapsed_time_str = f"Time: {elapsed_time:.2f}s"

                        # # Display predicted distance, velocity, acceleration, and elapsed time on the frame
                        # velocity_str = f"Vel: {velocity:.2f}" if velocity else "Vel: N/A"
                        # acceleration_str = f"Acc: {acceleration:.2f}" if acceleration else "Acc: N/A"

                        cv2.putText(annotated_frame, f"Dist: {predicted_y[0]:.2f}", 
                                    (int(x - w/2), int(y - h/2) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        # cv2.putText(annotated_frame, velocity_str, 
                        #         (int(x - w/2), int(y - h/2) - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        # cv2.putText(annotated_frame, acceleration_str, 
                        #             (int(x - w/2), int(y - h/2) - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        # cv2.putText(annotated_frame, elapsed_time_str, 
                        #             (int(x - w/2), int(y - h/2) - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        # if track_id not in printed_ids:
                        #     print(f"Track ID: {track_id}, Area: {area}, Predicted Distance: {predicted_y[0]:.2f}, Velocity: {velocity:.2f}, Acceleration: {acceleration:.2f}, Elapsed Time: {elapsed_time:.2f}s")
                        #     printed_ids.add(track_id)

                        if len(vehicle_directions[track_id]) == max_direction_len:
                            most_common_direction = Counter(vehicle_directions[track_id]).most_common(1)[0][0]
                            cv2.putText(annotated_frame, f"Vehicle {track_id} Direction: {most_common_direction}", (10, 30 + 30 * track_id), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


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

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(elapsed_times_track_3, velocities_track_3, label='Velocity')
    plt.xlabel('Elapsed Time (s)')
    plt.ylabel('Velocity')
    plt.title('Velocity vs Elapsed Time (Track ID 3)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(elapsed_times_track_3, accelerations_track_3, label='Acceleration', color='orange')
    plt.xlabel('Elapsed Time (s)')
    plt.ylabel('Acceleration')
    plt.title('Acceleration vs Elapsed Time (Track ID 3)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    
    model = YOLO("yolov8_aug.pt")
    json_path = ('datasplit.json')
    video_path = "testvdo (5).mp4"
    
    track(model, json_path, video_path)

if __name__ == "__main__":
    main()
