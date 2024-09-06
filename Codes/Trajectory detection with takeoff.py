from collections import defaultdict, Counter
import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np

def main():
    model = YOLO('yolov8_aug.pt')
    video_path = "Dynamic_test2.mp4"
    cap = cv2.VideoCapture(video_path)
    detection_data = []

    if not cap.isOpened():
        print("Error opening video file.")
        return

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video.")
        cap.release()
        return

    frame_height, frame_width = frame.shape[:2]
    frame_center_x, frame_center_y = frame_width // 2, frame_height // 2

    cap.release()  # Reset for processing
    cap = cv2.VideoCapture(video_path)

    track_history = defaultdict(lambda: {
        "points": [],
        "counter": 0,
        "velocities": [],
        "distances": [],
        "approaches": [],
        "directions": [],
        "last_common_approach": "N/A",
        "last_common_direction": "N/A",
        "last_position": (0, 0),
        "last_area": 0,
        "frame_count_since_last_area_update": 0
    })

    frame_counter = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_counter += 1

        results = model.track(frame, persist=True)

        if results and results[0].boxes is not None and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            annotated_frame = results[0].plot()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                center_x, center_y = x + w / 2, y + h / 2

                adjusted_x = x - frame_center_x
                adjusted_y = frame_center_y - y  # Adjust for the inverted y-axis in images

                area = w * h

                track_info = track_history[track_id]

                # Distance and velocity calculation
                points = track_info["points"]
                if points:
                    prev_x, prev_y = points[-1]
                    distance = np.sqrt((adjusted_x - prev_x) ** 2 + (adjusted_y - prev_y) ** 2)
                    track_info["distances"].append(distance)
                    
                    if len(track_info["distances"]) > 1:
                        cumulative_distance = sum(track_info["distances"])
                        time_elapsed = frame_counter / frame_rate
                        velocity = cumulative_distance / time_elapsed
                    else:
                        velocity = 0.0  # Initial velocity is 0 if there's not enough history
                    track_info["velocities"].append(velocity)
                    
                    # Display velocity
                    cv2.putText(annotated_frame, f"Velocity: {velocity:.2f} px/s", (int(x), int(y) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                track_info["points"].append((adjusted_x, adjusted_y))

                # Area comparison logic
                if track_info["frame_count_since_last_area_update"] >= 3:
                    if area > track_info["last_area"]:
                        approach = "Approaching"
                    else:
                        approach = "Moving Away"
                    track_info["last_area"] = area
                    track_info["frame_count_since_last_area_update"] = 0
                else:
                    approach = track_info["approaches"][-1] if track_info["approaches"] else "N/A"
                
                track_info["frame_count_since_last_area_update"] += 1
                track_info["approaches"].append(approach)

                # Direction calculation logic
                if len(track_info["points"]) > 3:
                    current_adjusted_x = adjusted_x
                    previous_adjusted_x = track_info["points"][-4][0]
                    direction = "Right" if current_adjusted_x > previous_adjusted_x else "Left"
                else:
                    direction = track_info["last_common_direction"]
                
                track_info["directions"].append(direction)
                track_info["points"].append((adjusted_x, adjusted_y))
                track_info["counter"] += 1

                # Common approach and direction
                if track_info["counter"] % 30 == 0:
                    common_approach = Counter(track_info["approaches"]).most_common(1)[0][0]
                    common_direction = Counter(track_info["directions"]).most_common(1)[0][0]
                    track_info["last_common_approach"] = common_approach
                    track_info["last_common_direction"] = common_direction
                    track_info["approaches"].clear()
                    track_info["directions"].clear()
                else:
                    common_approach = track_info["last_common_approach"]
                    common_direction = track_info["last_common_direction"]

                # Displaying the calculated velocity, distance, and area on the annotated frame
                #cv2.putText(annotated_frame, f"Velocity: {velocity:.2f} px/s", (int(x), int(y - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                #cv2.putText(annotated_frame, f"Distance: {sum(track_info['distances']):.2f} px", (int(x), int(y - 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                #cv2.putText(annotated_frame, f"Area: {area:.2f} px^2", (int(x), int(y - 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(annotated_frame, f"{common_approach}, {common_direction}", (int(x), int(y - 80)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("No detections or track IDs in this frame.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
