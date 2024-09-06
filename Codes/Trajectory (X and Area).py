from collections import defaultdict, Counter
import cv2
from ultralytics import YOLO
import pandas as pd  # Adjust this import based on your actual YOLO installation

def main():
    model = YOLO("C:/Users/Mech.coep/Desktop/Rahalkar Project Data/Situational Intellegence Model\yolo\yolov8_aug.pt")  # Make sure the model path is correct
    video_path = "C:/Users/Mech.coep/Desktop/Rahalkar Project Data/Situational Intellegence Model/yolo/Test videos/Dynamic_velocity_test1.mp4"
    cap = cv2.VideoCapture(video_path)
    detection_data = []

    if not cap.isOpened():
        print("Error opening video file.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Failed to read video.")
        cap.release()
        return

    frame_height, frame_width = frame.shape[:2]
    frame_center_x, frame_center_y = frame_width // 2, frame_height // 2

    cap.release()  # Reset for processing
    cap = cv2.VideoCapture(video_path)

    track_history = defaultdict(lambda: {"points": [], "counter": 0, "approaches": [], "directions": [], "last_common_approach": "N/A", "last_common_direction": "N/A", "last_area": 0})
    frame_counter = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_counter +=1      
        # Draw a small circle at the center of the frame
        cv2.circle(frame, (frame_center_x, frame_center_y), 5, (0, 255, 0), -1)

        results = model.track(frame, persist=True)

        if results[0].boxes is not None and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            annotated_frame = results[0].plot()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                center_x = x + (w / 2)
                center_y = y + (h / 2)

                adjusted_x = x - frame_center_x
                adjusted_y = frame_center_y - y  # Adjust for the inverted y-axis in images

                area = w * h  # Calculate the area of the bounding box
                area = area.item()
                track_info = track_history[track_id]
                # Assuming approach and direction are calculated as before and stored in track_info

                # Convert approach to numerical value for Excel output
                # "Approach" is 1 for "Approaching", 0 for "Moving Away"
                numerical_approach = 1 if track_info["last_common_approach"] == "Approaching" else 0 if track_info["last_common_approach"] == "Moving Away" else "N/A"

                # Convert direction to numerical value for Excel output
                # "Direction" is 1 for "Right", 0 for "Left"
                numerical_direction = 1 if track_info["last_common_direction"] == "Right" else 0 if track_info["last_common_direction"] == "Left" else "N/A"

                detection_data.append({
                    "Frame": frame_counter,
                    "Track ID": track_id,
                    "X": float(adjusted_x.item()),
                    "Y": float(adjusted_y.item()),
                    "Area": float(area),
                    "Approach": numerical_approach,  # Use numerical value for approach
                    "Direction": numerical_direction  # Use numerical value for direction
                })


                # Draw a circle at the center of the bounding box
                cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 0, 255), -1)

                # Display coordinates of the bounding box center
                coordinates_text = f"({adjusted_x:.2f}, {adjusted_y:.2f})"
                cv2.putText(annotated_frame, coordinates_text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Display the area of the bounding box
                area_text = f"Area: {area:.2f}"
                cv2.putText(annotated_frame, area_text, (int(x), int(y) - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

               # Inside the while loop, after calculating the area
                track_info = track_history[track_id]
                track_info["points"].append((adjusted_x, adjusted_y))
                track_info["counter"] += 1

                # Initialize a frame counter for area comparison if it doesn't exist
                if "frame_count_since_last_area_update" not in track_info:
                    track_info["frame_count_since_last_area_update"] = 0

                # Increment the frame counter for each track
                track_info["frame_count_since_last_area_update"] += 1

                # Check if it's time to compare the area (every 3 frames)
                if track_info["frame_count_since_last_area_update"] >= 3:
                    if area > track_info["last_area"]:
                        approach = "Approaching"
                    else:
                        approach = "Moving Away"
                    
                    # Update the last area and reset the frame counter
                    track_info["last_area"] = area
                    track_info["frame_count_since_last_area_update"] = 0
                else:
                    # If not comparing yet, keep the last approach state
                    if track_info.get("approaches"):  # Checks if the list is not empty
                        approach = track_info["approaches"][-1]
                    else:
                        approach = "N/A"  # Default state if no approaches have been recorded yet  # Use the most recent approach if available

                

                if len(track_info["points"]) > 3:
                    current_adjusted_x = adjusted_x
                    previous_adjusted_x = track_info["points"][-4][0]
                    direction = "Right" if current_adjusted_x > previous_adjusted_x else "Left"
                else:
                    direction = track_info["last_common_direction"]

                track_info["approaches"].append(approach)
                track_info["directions"].append(direction)

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

                # Display the common approach and direction on the annotated frame
                info_text = f"{common_approach}, {common_direction}"
                cv2.putText(annotated_frame, info_text, (int(x), int(y - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Display the frame with all annotations
            frame_number_text = f"Frame: {frame_counter}"
            cv2.putText(annotated_frame, frame_number_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# The rest of your display code continues here...
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop and exit by pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("No detections or track IDs in this frame.")
            continue

    # Release the video capture object and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    df = pd.DataFrame(detection_data)
    # Write the DataFrame to an Excel file
    df.to_excel("detection_data2.xlsx", index=False, engine='xlsxwriter')

if __name__ == "__main__":
    main()
