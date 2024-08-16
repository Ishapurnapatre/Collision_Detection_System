import cv2
import numpy as np
import json
from collections import defaultdict, deque
from ultralytics import YOLO
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time

# Load the JSON data for polynomial regression
with open('datasplit.json', 'r') as file:
    data = json.load(file)

tw_x_values1 = np.array(data['tw_x_values1']).reshape(-1, 1)
tw_y_values1 = np.array(data['tw_y_values1'])
tw_x_values2 = np.array(data['tw_x_values2']).reshape(-1, 1)
tw_y_values2 = np.array(data['tw_y_values2'])

# Polynomial Regression for case 1 (area < 46200)
poly_features1 = PolynomialFeatures(degree=3)
x_poly1 = poly_features1.fit_transform(tw_x_values1)
model1 = LinearRegression()
model1.fit(x_poly1, tw_y_values1)

# Polynomial Regression for case 2 (area >= 46200)
poly_features2 = PolynomialFeatures(degree=3)
x_poly2 = poly_features2.fit_transform(tw_x_values2)
model2 = LinearRegression()
model2.fit(x_poly2, tw_y_values2)

# Load the YOLOv8 model
model = YOLO("yolov8_aug.pt")

# Open the video file
video_path = "testvdo (4).mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Get frame rate and dimensions of the input video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
dt = 1 / fps  # Time difference between frames

track_history = defaultdict(deque)
predicted_distance_history = defaultdict(deque)
velocity_history = defaultdict(deque)
acceleration_history = defaultdict(deque)
frame_count = 0

# Dictionaries to store the start time for each track ID
start_time = defaultdict(lambda: time.time())

# Lists for plotting (only for track ID 3)
velocities_track_3 = []
accelerations_track_3 = []
elapsed_times_track_3 = []

while cap.isOpened():
    success, frame = cap.read()
    frame_count += 1  # Increment frame count

    if success:
        results = model.track(frame, persist=True)

        if results and hasattr(results[0], 'boxes') and results[0].boxes:
            boxes = results[0].boxes

            if hasattr(boxes, 'xywh') and hasattr(boxes, 'id') and boxes.id is not None:
                annotated_frame = results[0].plot()

                track_ids = boxes.id.int().cpu().tolist()
                boxes_xywh = boxes.xywh.cpu().numpy()
                class_names = [results[0].names[i] for i in boxes.cls.int().cpu().tolist()]

                printed_ids = set()  # Keep track of printed IDs in the current frame

                for box, track_id, class_name in zip(boxes_xywh, track_ids, class_names):
                    x, y, w, h = box
                    area = w * h
                    current_position = (x, y)

                    # Predict distance based on area
                    if area > 46200:
                        x_poly1_pred = poly_features1.transform([[area]])
                        predicted_y = model1.predict(x_poly1_pred)
                    else:
                        x_poly2_pred = poly_features2.transform([[area]])
                        predicted_y = model2.predict(x_poly2_pred)

                    predicted_y_scalar = predicted_y[0] if isinstance(predicted_y, np.ndarray) else predicted_y

                    # Store the current timestamp and predicted distance
                    current_time = time.time()

                    # Initialize velocity and acceleration
                    velocity = 0.0
                    acceleration = 0.0

                    if len(predicted_distance_history[track_id]) >= 10:
                        # Calculate velocity and acceleration
                        previous_distances = list(predicted_distance_history[track_id])[-10:]
                        previous_times = list(track_history[track_id])[-10:]

                        # Ensure there are 10 points to calculate
                        if len(previous_distances) == 10 and len(previous_times) == 10:
                            distance_diff = predicted_y_scalar - previous_distances[0]
                            time_diff = current_time - previous_times[0]

                            # Calculate velocity and acceleration
                            if time_diff > 0:
                                velocity = abs(distance_diff) / time_diff
                                velocities = [distance_diff / (current_time - t) for t in previous_times[1:]]
                                acceleration = abs(velocity - velocities[-1]) / (current_time - previous_times[-1]) if len(velocities) > 1 else 0

                                # Update histories
                                velocity_history[track_id].append(velocity)
                                acceleration_history[track_id].append(acceleration)
                            else:
                                velocity = 0.0
                                acceleration = 0.0

                        # Maintain history of distances and times
                        predicted_distance_history[track_id].popleft()
                        track_history[track_id].popleft()

                    # Append new data
                    predicted_distance_history[track_id].append(predicted_y_scalar)
                    track_history[track_id].append(current_time)

                    # Collect data only for track ID 3
                    if track_id == 3:
                        if velocity_history[track_id]:
                            velocities_track_3.append(velocity_history[track_id][-1])
                            elapsed_times_track_3.append(current_time - start_time[track_id])
                        if acceleration_history[track_id]:
                            accelerations_track_3.append(acceleration_history[track_id][-1])

                    # Calculate elapsed time from the start of the video
                    elapsed_time = current_time - start_time[track_id]
                    elapsed_time_str = f"Time: {elapsed_time:.2f}s"

                    # Display predicted distance, velocity, acceleration, and elapsed time on the frame
                    velocity_str = f"Vel: {velocity:.2f}" if velocity else "Vel: N/A"
                    acceleration_str = f"Acc: {acceleration:.2f}" if acceleration else "Acc: N/A"
                    cv2.putText(annotated_frame, f"Dist: {predicted_y_scalar:.2f}", 
                                (int(x - w/2), int(y - h/2) - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(annotated_frame, velocity_str, 
                                (int(x - w/2), int(y - h/2) - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(annotated_frame, acceleration_str, 
                                (int(x - w/2), int(y - h/2) - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(annotated_frame, elapsed_time_str, 
                                (int(x - w/2), int(y - h/2) - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    if track_id not in printed_ids:
                        # Print track ID, area, distance, velocity, acceleration, and elapsed time
                        print(f"Track ID: {track_id}, Area: {area}, Predicted Distance: {predicted_y_scalar:.2f}, Velocity: {velocity:.2f}, Acceleration: {acceleration:.2f}, Elapsed Time: {elapsed_time:.2f}s")
                        printed_ids.add(track_id)

            else:
                annotated_frame = frame.copy()
        else:
            annotated_frame = frame.copy()

        # Write the annotated frame to the output video
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

# Plotting the results for track ID 3
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
