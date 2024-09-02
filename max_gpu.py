import cv2
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from ultralytics import YOLO
from threading import Thread, Event
from queue import Queue, Full, Empty
import GPUtil

# Initialize queues
frame_queue = Queue(maxsize=10)
result_queue = Queue(maxsize=10)
display_frame_queue = Queue(maxsize=10)  # Queue for frames to be displayed

# Flag to indicate video end
video_end_event = Event()

# Initialize GPU metrics storage
gpu_times = []
gpu_utilizations = []
gpu_memory_frees = []
gpu_memory_useds = []
gpu_temperatures = []

def print_gpu_metrics_averages():
    if gpu_times:
        avg_utilization = np.mean(gpu_utilizations)
        avg_memory_free = np.mean(gpu_memory_frees)
        avg_memory_used = np.mean(gpu_memory_useds)
        avg_temperature = np.mean(gpu_temperatures)

        print(f"Average GPU Utilization: {avg_utilization:.2f}%")
        print(f"Average GPU Memory Free: {avg_memory_free:.2f} MB")
        print(f"Average GPU Memory Used: {avg_memory_used:.2f} MB")
        print(f"Average GPU Temperature: {avg_temperature:.2f} °C")
    else:
        print("No GPU metrics to calculate averages.")

def get_gpu_utilization():
    gpus = GPUtil.getGPUs()
    if len(gpus) > 0:
        gpu = gpus[0]
        utilization = gpu.load * 100
        memory_free = gpu.memoryFree
        memory_used = gpu.memoryUsed
        temperature = gpu.temperature
        return utilization, memory_free, memory_used, temperature
    return None, None, None, None

def load_yolo_model():
    model_path = "yolov8_aug.pt"
    model = YOLO(model_path)
    return model

def calculate_distance_models(class_name):
    with open("distance.json", 'r') as file:
        data = json.load(file)

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

def determine_direction(prev_center, curr_center):
    dx = curr_center[0] - prev_center[0]
    dy = curr_center[1] - prev_center[1]

    if abs(dx) > abs(dy):
        if dx > 0:
            direction = "Right"
        else:
            direction = "Left"
    else:
        if dy > 0:
            direction = "Down"
        else:
            direction = "Up"

    print(f"Determined Direction: {direction}")
    return direction

def frame_capture_thread(video_path):
    cap = cv2.VideoCapture(video_path)
    while not video_end_event.is_set():
        if not cap.isOpened():
            print("Video capture is not opened. Reinitializing.")
            cap.open(video_path)

        start_time = time.time()
        success, frame = cap.read()
        if not success:
            print("End of video reached.")
            video_end_event.set()
            break

        if not frame_queue.full():
            try:
                frame_queue.put(frame, timeout=1)
                display_frame_queue.put(frame, timeout=1)  # Also put frame in display queue
            except Full:
                pass
        
        # Synchronize frame rate (adjust as needed)
        elapsed_time = time.time() - start_time
        frame_time = 1 / 30  # Assuming video at 30 FPS
        if elapsed_time < frame_time:
            time.sleep(frame_time - elapsed_time)

    cap.release()
    frame_queue.put(None)  # Signal the end of the video processing
    display_frame_queue.put(None)  # Signal the end of the display processing
    print("Frame capture thread ending.")

def process_frame_thread(model):
    while not video_end_event.is_set() or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:
                break

            start_time = time.time()
            results = model.track(frame, persist=True)
            inference_time = (time.time() - start_time) * 1000  # in milliseconds

            if results and hasattr(results[0], 'boxes') and results[0].boxes:
                boxes = results[0].boxes

                if hasattr(boxes, 'xywh') and hasattr(boxes, 'id') and boxes.id is not None:
                    track_ids = boxes.id.int().cpu().tolist()
                    boxes_xywh = boxes.xywh.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    class_names = [results[0].names[i] for i in boxes.cls.int().cpu().tolist()]

                    result_queue.put((boxes_xywh, track_ids, confidences, class_names, inference_time))
        except Empty:
            continue

    result_queue.put(None)  # Signal the end of the result processing
    print("Frame processing thread ending.")

def result_display_thread():
    track_history = {}
    velocity_history = {}
    acceleration_history = {}
    previous_times = {}
    previous_centers = {}
    frame_count = 0

    while not video_end_event.is_set() or not display_frame_queue.empty():
        try:
            frame = display_frame_queue.get(timeout=1)
            if frame is None:
                break

            while not result_queue.empty():
                result = result_queue.get(timeout=1)
                if result is None:
                    break

                boxes_xywh, track_ids, confidences, class_names, inference_time = result

                for box, track_id, confidence, class_name in zip(boxes_xywh, track_ids, confidences, class_names):
                    x, y, w, h = box
                    center = np.array([x, y])
                    area = w * h

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

                    if len(track_history[track_id]) >= 10 and len(previous_times[track_id]) >= 2:
                        time_elapsed = previous_times[track_id][-1] - previous_times[track_id][-2]
                        prev_distance = track_history[track_id][-2][4]
                        velocity = calculate_velocity(prev_distance, distance, time_elapsed)
                        velocity_history[track_id].append(velocity)

                        if len(velocity_history[track_id]) >= 2:
                            prev_velocity = velocity_history[track_id][-2]
                            acceleration = calculate_acceleration(prev_velocity, velocity, time_elapsed)
                            acceleration_history[track_id].append(acceleration)

                    prev_center = previous_centers[track_id]
                    curr_center = center
                    direction = determine_direction(prev_center, curr_center)

                    if track_id not in previous_centers:
                        previous_centers[track_id] = curr_center

                    label = f"ID: {track_id}, Class: {class_name}, Conf: {confidence:.2f}, Dist: {distance:.2f}m, Vel: {velocity_history.get(track_id, [-1])[-1] if velocity_history.get(track_id) else 'N/A'}, Acc: {acceleration_history.get(track_id, [-1])[-1] if acceleration_history.get(track_id) else 'N/A'}, Dir: {direction}"
                    cv2.putText(frame, label, (int(x - w/2), int(y - h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('YOLO Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video_end_event.set()
                break

            frame_count += 1
        except Empty:
            continue

    cv2.destroyAllWindows()
    print_gpu_metrics_averages()
    print("Result display thread ending.")

def monitor_gpu_metrics():
    while not video_end_event.is_set():
        utilization, memory_free, memory_used, temperature = get_gpu_utilization()
        if utilization is not None:
            gpu_times.append(time.time())
            gpu_utilizations.append(utilization)
            gpu_memory_frees.append(memory_free)
            gpu_memory_useds.append(memory_used)
            gpu_temperatures.append(temperature)
        time.sleep(1)  # Monitor every second

def main():
    video_path = "testvdo (6).mp4"

    model = load_yolo_model()

    # Start threads
    capture_thread = Thread(target=frame_capture_thread, args=(video_path,))
    process_thread = Thread(target=process_frame_thread, args=(model,))
    display_thread = Thread(target=result_display_thread)
    gpu_monitor_thread = Thread(target=monitor_gpu_metrics)

    capture_thread.start()
    process_thread.start()
    display_thread.start()
    gpu_monitor_thread.start()

    capture_thread.join()
    process_thread.join()
    display_thread.join()
    gpu_monitor_thread.join()

if __name__ == "__main__":
    main()
