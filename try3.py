import cv2
import numpy as np
import json
from collections import defaultdict, deque
from ultralytics import YOLO
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time
import streamlit as st

def load_polynomial_regression_models(json_file):
    with open(json_file, 'r') as file:
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

    return poly_features1, model1, poly_features2, model2

def initialize_streamlit_state():
    if 'run' not in st.session_state:
        st.session_state.run = False
    if 'pause' not in st.session_state:
        st.session_state.pause = False
    if 'current_pos' not in st.session_state:
        st.session_state.current_pos = 0
    if 'annotated_frame' not in st.session_state:
        st.session_state.annotated_frame = None

def initialize_streamlit_ui():
    st.title("YOLOv8 Object Tracking")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('Start'):
            st.session_state.run = True
            st.session_state.pause = False
    with col2:
        if st.button('Pause'):
            st.session_state.pause = True
            st.session_state.run = False
    with col3:
        if st.button('Stop'):
            st.session_state.run = False
            st.session_state.pause = False
            st.session_state.current_pos = 0
            st.session_state.annotated_frame = None

def initialize_tracking_data():
    return defaultdict(deque), defaultdict(deque), defaultdict(deque), defaultdict(deque), defaultdict(lambda: time.time())

def update_track_histories(track_id, predicted_y_scalar, current_time, predicted_distance_history, track_history):
    if len(predicted_distance_history[track_id]) >= 10:
        previous_distances = list(predicted_distance_history[track_id])[-10:]
        previous_times = list(track_history[track_id])[-10:]

        if len(previous_distances) == 10 and len(previous_times) == 10:
            distance_diff = predicted_y_scalar - previous_distances[0]
            time_diff = current_time - previous_times[0]

            if time_diff > 0:
                velocity = abs(distance_diff) / time_diff
                velocities = [distance_diff / (current_time - t) for t in previous_times[1:]]
                acceleration = abs(velocity - velocities[-1]) / (current_time - previous_times[-1]) if len(velocities) > 1 else 0
            else:
                velocity, acceleration = 0.0, 0.0

            predicted_distance_history[track_id].popleft()
            track_history[track_id].popleft()
        else:
            velocity, acceleration = 0.0, 0.0
    else:
        velocity, acceleration = 0.0, 0.0

    predicted_distance_history[track_id].append(predicted_y_scalar)
    track_history[track_id].append(current_time)
    
    return velocity, acceleration

def process_frame(frame, model, poly_features1, model1, poly_features2, model2, track_history, predicted_distance_history, velocity_history, acceleration_history, velocities_track_3, accelerations_track_3, elapsed_times_track_3, start_time):
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
                current_time = time.time()

                # Predict distance based on area
                if area >= 46200:
                    x_poly = poly_features1.transform([[area]])
                    predicted_y = model1.predict(x_poly)
                else:
                    x_poly = poly_features2.transform([[area]])
                    predicted_y = model2.predict(x_poly)

                predicted_y_scalar = predicted_y[0] if isinstance(predicted_y, np.ndarray) else predicted_y

                # Update track histories
                velocity, acceleration = update_track_histories(track_id, predicted_y_scalar, current_time, predicted_distance_history, track_history)
                
                # Collect data for track ID 3
                if track_id == 3:
                    if velocity_history[track_id]:
                        velocities_track_3.append(velocity_history[track_id][-1])
                        elapsed_times_track_3.append(current_time - start_time[track_id])
                    if acceleration_history[track_id]:
                        accelerations_track_3.append(acceleration_history[track_id][-1])

                # Display information on the frame
                elapsed_time = current_time - start_time[track_id]
                cv2.putText(annotated_frame, f"Dist: {predicted_y_scalar:.2f}", (int(x - w/2), int(y - h/2) - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Vel: {velocity:.2f}", (int(x - w/2), int(y - h/2) - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Acc: {acceleration:.2f}", (int(x - w/2), int(y - h/2) - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Time: {elapsed_time:.2f}s", (int(x - w/2), int(y - h/2) - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if track_id not in printed_ids:
                    print(f"Track ID: {track_id}")
                    print(f"Area: {area:.2f}")
                    print(f"Predicted Distance: {predicted_y_scalar:.2f}")
                    print(f"Velocity: {velocity:.2f}")
                    print(f"Acceleration: {acceleration:.2f}")
                    print(f"Elapsed Time: {elapsed_time:.2f}s")
                    print("---------------")
                    printed_ids.add(track_id)

            return annotated_frame
    return frame

def plot_graphs(velocities_track_3, accelerations_track_3, elapsed_times_track_3):
    if velocities_track_3 and accelerations_track_3 and elapsed_times_track_3:
        # Plot Velocity vs Elapsed Time
        fig, ax = plt.subplots()
        ax.plot(elapsed_times_track_3, velocities_track_3, label='Velocity', color='blue', marker='o')
        ax.set_xlabel('Elapsed Time (s)')
        ax.set_ylabel('Velocity')
        ax.set_title('Velocity vs Elapsed Time')
        ax.legend()
        st.pyplot(fig)

        # Plot Acceleration vs Elapsed Time
        fig, ax = plt.subplots()
        ax.plot(elapsed_times_track_3, accelerations_track_3, label='Acceleration', color='red', marker='o')
        ax.set_xlabel('Elapsed Time (s)')
        ax.set_ylabel('Acceleration')
        ax.set_title('Acceleration vs Elapsed Time')
        ax.legend()
        st.pyplot(fig)

def main():
    poly_features1, model1, poly_features2, model2 = load_polynomial_regression_models('datasplit.json')
    model = YOLO("yolov8_aug.pt")

    initialize_streamlit_state()
    initialize_streamlit_ui()

    video_path = "testvdo (4).mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error opening video stream or file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1 / fps
    track_history, predicted_distance_history, velocity_history, acceleration_history, start_time = initialize_tracking_data()

    velocities_track_3 = []
    accelerations_track_3 = []
    elapsed_times_track_3 = []

    stframe = st.empty()

    while cap.isOpened():
        if st.session_state.run and not st.session_state.pause:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_pos)
            success, frame = cap.read()

            if success:
                annotated_frame = process_frame(frame, model, poly_features1, model1, poly_features2, model2, track_history, predicted_distance_history, velocity_history, acceleration_history, velocities_track_3, accelerations_track_3, elapsed_times_track_3, start_time)
                st.session_state.annotated_frame = annotated_frame
                stframe.image(annotated_frame, channels="BGR")
                plot_graphs(velocities_track_3, accelerations_track_3, elapsed_times_track_3)
                st.session_state.current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            else:
                break
        elif st.session_state.pause:
            if st.session_state.annotated_frame is not None:
                stframe.image(st.session_state.annotated_frame, channels="BGR")
            time.sleep(dt)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
