import cv2
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from pynvml import *
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, nvmlDeviceGetTemperature, nvmlDeviceGetClockInfo, NVML_CLOCK_GRAPHICS, NVML_TEMPERATURE_GPU
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from ultralytics import YOLO

def print_gpu_metrics_averages(gpu_utilizations, gpu_memory_frees, gpu_memory_useds, gpu_temperatures, gpu_clock_speeds):
    if gpu_utilizations:
        avg_utilization = np.mean(gpu_utilizations)
        avg_memory_free = np.mean(gpu_memory_frees)
        avg_memory_used = np.mean(gpu_memory_useds)
        avg_temperature = np.mean(gpu_temperatures)
        avg_clock_speed = np.mean(gpu_clock_speeds)

        print(f"Average GPU Utilization: {avg_utilization:.2f}%")
        print(f"Average GPU Memory Free: {avg_memory_free:.2f} MB")
        print(f"Average GPU Memory Used: {avg_memory_used:.2f} MB")
        print(f"Average GPU Temperature: {avg_temperature:.2f} °C")
        print(f"Average GPU Clock Speed: {avg_clock_speed:.2f} MHz")
    else:
        print("No GPU metrics to calculate averages.")


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

def get_gpu_utilization():
    try:
        nvmlInit()
        gpu_count = nvmlDeviceGetCount()
        if gpu_count == 0:
            print("No GPUs detected.")
            return 0, 0, 0, 0, 0  # No utilization data

        gpu = nvmlDeviceGetHandleByIndex(0)  # Assuming only one GPU
        utilization = nvmlDeviceGetUtilizationRates(gpu).gpu
        memory_info = nvmlDeviceGetMemoryInfo(gpu)
        memory_free = memory_info.free / 1024**2  # Convert bytes to MB
        memory_used = memory_info.used / 1024**2  # Convert bytes to MB
        temperature = nvmlDeviceGetTemperature(gpu, NVML_TEMPERATURE_GPU)
        clock_speed = nvmlDeviceGetClockInfo(gpu, NVML_CLOCK_GRAPHICS)  # in MHz

        return utilization, memory_free, memory_used, temperature, clock_speed
    except Exception as e:
        print(f"Error monitoring GPU: {e}")
        return 0, 0, 0, 0, 0
    finally:
        nvmlShutdown()

def track_and_analyze(model, video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    track_history = {}
    velocity_history = {}
    acceleration_history = {}
    previous_times = {}
    previous_centers = {}
    
    # Lists to store GPU metrics
    gpu_times = []
    gpu_utilizations = []
    gpu_memory_frees = []
    gpu_memory_useds = []
    gpu_temperatures = []
    gpu_clock_speeds = []
    
    threshold = 15  # Example threshold value

    while cap.isOpened():
        success, frame = cap.read()
        frame_count += 1

        if not success:
            print("End of video or error reading frame.")
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
                                      (int(x + w/2), int(y + h/2)), color, 2)

        # Monitor GPU utilization every 20 frames
        if frame_count % 20 == 0:
            print("Monitoring GPU...")
            utilization, memory_free, memory_used, temperature, clock_speed = get_gpu_utilization()
            gpu_times.append(frame_count)
            gpu_utilizations.append(utilization)
            gpu_memory_frees.append(memory_free)
            gpu_memory_useds.append(memory_used)
            gpu_temperatures.append(temperature)
            gpu_clock_speeds.append(clock_speed)
            print(f"GPU Utilization: {utilization:.2f}%")
            print(f"GPU Memory Free: {memory_free}MB")
            print(f"GPU Memory Used: {memory_used}MB")
            print(f"GPU Temperature: {temperature}°C")
            print(f"GPU Clock Speed: {clock_speed}MHz")

        cv2.imshow("YOLOv8 Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Plot GPU utilization
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 2, 1)
    plt.plot(gpu_times, gpu_utilizations, label='GPU Utilization (%)', color='b')
    plt.xlabel('Frame Number')
    plt.ylabel('Utilization (%)')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(gpu_times, gpu_memory_frees, label='GPU Memory Free (MB)', color='g')
    plt.xlabel('Frame Number')
    plt.ylabel('Memory Free (MB)')
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(gpu_times, gpu_memory_useds, label='GPU Memory Used (MB)', color='r')
    plt.xlabel('Frame Number')
    plt.ylabel('Memory Used (MB)')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(gpu_times, gpu_temperatures, label='GPU Temperature (°C)', color='m')
    plt.xlabel('Frame Number')
    plt.ylabel('Temperature (°C)')
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(gpu_times, gpu_clock_speeds, label='GPU Clock Speed (MHz)', color='c')
    plt.xlabel('Frame Number')
    plt.ylabel('Clock Speed (MHz)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print average GPU metrics
    print_gpu_metrics_averages(gpu_utilizations, gpu_memory_frees, gpu_memory_useds, gpu_temperatures, gpu_clock_speeds)

if __name__ == "__main__":
    model = load_yolo_model()
    video_path = 'testvdo (2).mp4'
    track_and_analyze(model, video_path)
