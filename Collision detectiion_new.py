import cv2  # OpenCV for video processing
import numpy as np  # NumPy for numerical operations
import time  # Time module for tracking time intervals
import json  # JSON for reading and handling configuration data
import matplotlib.pyplot as plt  # Matplotlib for plotting graphs
from sklearn.preprocessing import PolynomialFeatures  # Polynomial transformation for regression
from sklearn.linear_model import LinearRegression  # Linear regression model
from ultralytics import YOLO  # YOLO model for object detection

def load_yolo_model(model_path="yolov8_aug.pt"):
    """Load the YOLO model from the specified path."""
    return YOLO(model_path)  # Load YOLOv8 model

def load_distance_models(json_path="datasplit.json"):
    """Load precomputed polynomial regression models for distance estimation."""
    with open(json_path, 'r') as file:
        data = json.load(file)  # Load JSON data

    models = {}  # Dictionary to store regression models
    for i in [1, 2]:  # Loop for two different models
        x_values = np.array(data[f'tw_x_values{i}']).reshape(-1, 1)  # Reshape X values
        y_values = np.array(data[f'tw_y_values{i}'])  # Get corresponding Y values
        poly = PolynomialFeatures(degree=3)  # Create polynomial features
        x_poly = poly.fit_transform(x_values)  # Transform X values
        model = LinearRegression().fit(x_poly, y_values)  # Train regression model
        models[i] = (model, poly)  # Store model and transformation
    
    return models  # Return trained models

def calculate_velocity(distance1, distance2, time_elapsed):
    """Calculate velocity given two distances and time elapsed."""
    return (abs(distance2 - distance1) / time_elapsed) if time_elapsed > 0 else 0

def calculate_acceleration(velocity1, velocity2, time_elapsed):
    """Calculate acceleration given two velocities and time elapsed."""
    return (abs(velocity2 - velocity1) / time_elapsed) if time_elapsed > 0 else 0

def track_and_analyze(model, models, camera_index=0, output_path="processed_video.avi", fps=30):
    """Track objects, estimate distances, and analyze their movement."""
    cap = cv2.VideoCapture(camera_index)  # Capture video from camera
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))  # Get frame dimensions
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))  # Initialize video writer
    
    track_data = {}  # Dictionary to store tracking data
    threshold = 3  # Time to collision threshold (in seconds)
    
    while cap.isOpened():  # Process video until closed
        success, frame = cap.read()  # Read frame from video
        if not success:
            break

        results = model.track(frame, persist=True)  # Perform object tracking
        if results and hasattr(results[0], 'boxes'):
            boxes = results[0].boxes  # Extract bounding boxes
            if hasattr(boxes, 'xywh') and hasattr(boxes, 'id'):
                for box, track_id in zip(boxes.xywh.cpu().numpy(), boxes.id.int().cpu().tolist()):
                    x, y, w, h = box  # Extract bounding box coordinates
                    area = w * h  # Compute object area
                    print("w=", w, "h=", h)
                    print("Area=", area)
                    model_key = 1 if area > 46200 else 2  # Choose model based on area
                    model, poly = models[model_key]  # Get respective model
                    distance = model.predict(poly.transform([[area]]))[0]  # Predict distance



                    # if track_id not in track_data:  # Initialize tracking data if new ID
                    #     track_data[track_id] = {'history': [], 'velocity': [], 'acceleration': [], 'last_time': time.time()} #last time stores the timestamp when object is first detected

                    # track = track_data[track_id]  # Get tracking data for ID
                    # time_now = time.time()  # Current time
                    # time_elapsed = time_now - track['last_time']  # Time difference current-last
                    # track['history'].append(distance)  # Store distance history
                    
                    # Initialize tracking data for a new track_id
                    if track_id not in track_data:
                        track_data[track_id] = {
                        'history': [],        # Store distance history
                        'velocity': [],       # Store velocity history
                        'acceleration': [],   # Store acceleration history
                        'timestamps': [time.time()]  # Store initial timestamp
                         }
                    # Get tracking data for this track_id
                    track = track_data[track_id]
                    # Get current time
                    time_now = time.time()
                    # Retrieve previous timestamps
                    timestamps = track['timestamps']
                    # Calculate time elapsed since the last frame
                    if len(timestamps) >= 1:
                        time_elapsed = time_now - timestamps[-1]  # Time since the last frame
                    else:
                        time_elapsed = 0  # No previous frame, so no time difference
                    # Store distance history
                    track['history'].append(distance)
                    # Maintain a rolling window of the last 5 timestamps
                    timestamps.append(time_now)
                    if len(timestamps) > 5:
                        timestamps.pop(0)  # Remove the oldest timestamp

                        
                    if len(track['history']) > 1:  # Ensure enough data for velocity
                        velocity = calculate_velocity(track['history'][-2], distance, time_elapsed)
                        track['velocity'].append(velocity)  # Store velocity
                        
                        if len(track['velocity']) > 1:  # Ensure enough data for acceleration
                            acceleration = calculate_acceleration(track['velocity'][-2], velocity, time_elapsed)
                        else:
                            acceleration = 0
                        track['acceleration'].append(acceleration)  # Store acceleration
                        
                        time_to_collide = distance / velocity if velocity > 0 else float('inf')  # Estimate collision time
                        color = (0, 0, 255) if time_to_collide < threshold else (255, 255, 255)  # Set text color
                        
                        text_y = int(y - h/2 - 20)  # Position text label
                        cv2.putText(frame, f"Track ID: {track_id}", (int(x - w/2), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        text_y += 15
                        cv2.putText(frame, f"Distance: {distance:.2f}m", (int(x - w/2), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        text_y += 15
                        cv2.putText(frame, f"Velocity: {velocity:.2f}m/s", (int(x - w/2), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        text_y += 15
                        cv2.putText(frame, f"Acceleration: {acceleration:.2f}m/s²", (int(x - w/2), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        cv2.rectangle(frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)  # Draw bounding box
                        track['last_time'] = time_now  # Update last time

        out.write(frame)  # Save processed frame
        cv2.imshow("Tracking Analysis", frame)  # Show video
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit if 'q' is pressed
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    plot_velocity_acceleration(track_data)  # Plot analysis results

def plot_velocity_acceleration(track_data):
    """Plot velocity and acceleration for all tracked objects."""
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    for track_id, data in track_data.items():
        plt.plot(data['velocity'], label=f'Object {track_id}')
    plt.xlabel('Frames')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity Analysis')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 1, 2)
    for track_id, data in track_data.items():
        plt.plot(data['acceleration'], label=f'Object {track_id}')
    plt.xlabel('Frames')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Acceleration Analysis')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to load models and run tracking."""
    model = load_yolo_model()
    distance_models = load_distance_models()
    track_and_analyze(model, distance_models)

if __name__ == "__main__":
    main()
