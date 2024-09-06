from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd

def main():
    # Load the YOLOv8 model
    model = YOLO('yolov8_aug.pt')

    # Open the video file
    video_path = "Dynamic_velocity_test1.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Initially, get a frame to calculate the center of the video
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video.")
        cap.release()
        return

    frame_height, frame_width = frame.shape[:2]
    frame_center_x, frame_center_y = frame_width // 2, frame_height // 2
    cap.release()  # Close the video to reset it

    # Re-open the video for processing
    cap = cv2.VideoCapture(video_path)

    # Store the track history along with additional details
    track_history = defaultdict(lambda: {"points": [], "approaching": None, "moving_direction": None, "counter": 0, "last_direction": None, "last_approach": None})

    # Initialize a list to store the bounding box center points for each detection
    bbox_data_list = []

    # Loop through the video frames
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            frame_count += 1
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            if results[0].boxes is not None and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    center_x = x + w / 2
                    # Adjust coordinates to be relative to the center of the frame
                    adjusted_x = center_x - frame_center_x
                    # Invert Y to match the traditional Cartesian coordinate system
                    adjusted_y = frame_center_y - y
                    cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 0, 255), -1)

                    track_info = track_history[track_id]
                    track_info["points"].append((adjusted_x, adjusted_y))
                    track_info["counter"] += 1

                    if track_info["counter"] % 10 == 0:
                        # Determine direction and approach based on changes in position
                        initial_adjusted_x, initial_adjusted_y = track_info["points"][0]
                        delta_x = adjusted_x - initial_adjusted_x
                        delta_y = adjusted_y - initial_adjusted_y

                        direction = "Right" if delta_x > 0 else "Left"
                        approach = "Approaching" if delta_y < 0 else "Moving Away"  # Adjusted for inverted Y

                        track_info["last_direction"] = direction
                        track_info["last_approach"] = approach

                        bbox_data_list.append({
                            'Track ID': track_id,
                            'Frame': frame_count,
                            'Adjusted Center X': adjusted_x,
                            'Adjusted Center Y': adjusted_y,
                            'Direction': direction,
                            'Approach': approach
                        })
                    else:
                        # Use the last calculated direction and approach
                        direction = track_info["last_direction"] if track_info["last_direction"] else "N/A"
                        approach = track_info["last_approach"] if track_info["last_approach"] else "N/A"

                    # Display approach and direction information on the frame
                    info_text = f"{approach}, {direction}"
                    cv2.putText(annotated_frame, info_text, (int(x / 2), int(y -h/ 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                # Display the annotated frame
                cv2.imshow("YOLOv8 Tracking", annotated_frame)

            else:
                print("No detections or track IDs in this frame.")
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Create a DataFrame from the list after processing all frames
    #bbox_data = pd.DataFrame(bbox_data_list)

    # Save the DataFrame to an Excel file
    #bbox_data.to_excel('adjusted_bounding_box_centers1.xlsx', index=False)

if __name__ == "__main__":
    main()
