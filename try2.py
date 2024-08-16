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
def load_polynomial_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def train_polynomial_model(x_values, y_values, degree=3):
    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(np.array(x_values).reshape(-1, 1))
    model = LinearRegression().fit(x_poly, np.array(y_values))
    return model, poly_features

# Predict distance based on the area and the appropriate polynomial model
def predict_distance(area, model1, poly_features1, model2, poly_features2, threshold=46200):
    if area > threshold:
        x_poly_pred = poly_features1.transform([[area]])
        predicted_y = model1.predict(x_poly_pred)
    else:
        x_poly_pred = poly_features2.transform([[area]])
        predicted_y = model2.predict(x_poly_pred)
    return predicted_y[0] if isinstance(predicted_y, np.ndarray) else predicted_y

# Initialize the YOLOv8 model and video capture
def initialize_yolo_and_video(model_path, video_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video stream or file")
    return model, cap

# Initialize video writer with input video properties
def initialize_video_writer(cap, output_path, fourcc_str='XVID'):
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    return out, 1 / fps  # Return writer and time difference between frames

# Calculate velocity and acceleration based on history
def calculate_velocity_acceleration(predicted_distances, track_times, current_distance, current_time):
    if len(predicted_distances) >= 10:
        distance_diff = current_distance - predicted_distances[0]
        time_diff = current_time - track_times[0]
        if time_diff > 0:
            velocity = abs(distance_diff) / time_diff
            velocities = [distance_diff / (current_time - t) for t in track_times[1:]]
            acceleration = abs(velocity - velocities[-1]) / (current_time - track_times[-1]) if len(velocities) > 1 else 0
            return velocity, acceleration
    return 0.0, 0.0

# Main tracking and video processing loop
def process_video(model, cap, out, model1, poly_features1, model2, poly_features2):
    track_history = defaultdict(deque)
    predicted_distance_history = defaultdict(deque)
    velocity_history = defaultdict(deque)
    acceleration_history = defaultdict(deque)
    start_time = defaultdict(lambda: time.time())

    velocities_track_3 = []
    accelerations_track_3 = []
    elapsed_times_track_3 = []

    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        frame_count += 1

        if not success:
            break

        results = model.track(frame, persist=True)
        annotated_frame = process_frame(results, frame, track_history, predicted_distance_history,
                                        velocity_history, acceleration_history, start_time,
                                        model1, poly_features1, model2, poly_features2,
                                        velocities_track_3, accelerations_track_3, elapsed_times_track_3)

        # Write and display the annotated frame
        out.write(annotated_frame)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Plotting results for track ID 3
    plot_results(velocities_track_3, accelerations_track_3, elapsed_times_track_3)

# Process each frame: track objects, calculate distance, velocity, and acceleration
def process_frame(results, frame, track_history, predicted_distance_history, velocity_history, 
                  acceleration_history, start_time, model1, poly_features1, model2, poly_features2,
                  velocities_track_3, accelerations_track_3, elapsed_times_track_3):
    
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
                current_position = (x, y)

                predicted_y_scalar = predict_distance(area, model1, poly_features1, model2, poly_features2)

                current_time = time.time()
                velocity, acceleration = calculate_velocity_acceleration(
                    predicted_distance_history[track_id], track_history[track_id],
                    predicted_y_scalar, current_time)

                # Update histories
                update_history(predicted_distance_history, track_history, velocity_history, 
                               acceleration_history, track_id, predicted_y_scalar, 
                               current_time, velocity, acceleration)

                # Collect data only for track ID 3
                if track_id == 3:
                    collect_data_for_plot(track_id, velocity_history, acceleration_history, 
                                          velocities_track_3, accelerations_track_3, elapsed_times_track_3, start_time)

                # Display and print information
                display_and_print_info(annotated_frame, x, y, w, h, track_id, area, 
                                       predicted_y_scalar, velocity, acceleration, current_time, 
                                       start_time, printed_ids)
        else:
            annotated_frame = frame.copy()
    else:
        annotated_frame = frame.copy()

    return annotated_frame

# Update the history of distances, times, velocities, and accelerations
def update_history(predicted_distance_history, track_history, velocity_history, acceleration_history,
                   track_id, predicted_y_scalar, current_time, velocity, acceleration):
    if len(predicted_distance_history[track_id]) >= 10:
        predicted_distance_history[track_id].popleft()
        track_history[track_id].popleft()

    predicted_distance_history[track_id].append(predicted_y_scalar)
    track_history[track_id].append(current_time)

    if velocity != 0.0:
        velocity_history[track_id].append(velocity)
    if acceleration != 0.0:
        acceleration_history[track_id].append(acceleration)

# Collect data for plotting for track ID 3
def collect_data_for_plot(track_id, velocity_history, acceleration_history,
                          velocities_track_3, accelerations_track_3, elapsed_times_track_3, start_time):
    if velocity_history[track_id]:
        velocities_track_3.append(velocity_history[track_id][-1])
        elapsed_times_track_3.append(time.time() - start_time[track_id])
    if acceleration_history[track_id]:
        accelerations_track_3.append(acceleration_history[track_id][-1])

# Display predicted distance, velocity, acceleration, and elapsed time on the frame
def display_and_print_info(annotated_frame, x, y, w, h, track_id, area, 
                           predicted_y_scalar, velocity, acceleration, current_time, 
                           start_time, printed_ids):
    
    elapsed_time = current_time - start_time[track_id]
    elapsed_time_str = f"Time: {elapsed_time:.2f}s"
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
        print(f"Track ID: {track_id}, Area: {area}, Predicted Distance: {predicted_y_scalar:.2f}, "
              f"Velocity: {velocity:.2f}, Acceleration: {acceleration:.2f}, Elapsed Time: {elapsed_time:.2f}s")
        printed_ids.add(track_id)

# Plot results for track ID 3
def plot_results(velocities, accelerations, elapsed_times):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(elapsed_times, velocities, label='Velocity')
    plt.xlabel('Elapsed Time (s)')
    plt.ylabel('Velocity')
    plt.title('Velocity vs Elapsed Time (Track ID 3)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(elapsed_times, accelerations, label='Acceleration', color='orange')
    plt.xlabel('Elapsed Time (s)')
    plt.ylabel('Acceleration')
    plt.title('Acceleration vs Elapsed Time (Track ID 3)')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    data = load_polynomial_data('datasplit.json')
    
    model1, poly_features1 = train_polynomial_model(data['tw_x_values1'], data['tw_y_values1'])
    model2, poly_features2 = train_polynomial_model(data['tw_x_values2'], data['tw_y_values2'])
    
    model, cap = initialize_yolo_and_video("yolov8_aug.pt", "testvdo (4).mp4")
    out, dt = initialize_video_writer(cap, "output_video.avi")
    
    process_video(model, cap, out, model1, poly_features1, model2, poly_features2)
