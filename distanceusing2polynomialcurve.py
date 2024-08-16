import cv2
import numpy as np
import json
from collections import defaultdict
from ultralytics import YOLO
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load the JSON data for polynomial regression
with open('datasplit.json', 'r') as file:
    data = json.load(file)

# tw_x_values1 = np.array(data['tw_x_values1']).reshape(-1, 1)
# tw_y_values1 = np.array(data['tw_y_values1'])
# tw_x_values2 = np.array(data['tw_x_values2']).reshape(-1, 1)
# tw_y_values2 = np.array(data['tw_y_values2'])

tw_x_values1 = np.array(data['lmv_x_values1']).reshape(-1, 1)
tw_y_values1 = np.array(data['lmv_y_values1'])
tw_x_values2 = np.array(data['lmv_x_values2']).reshape(-1, 1)
tw_y_values2 = np.array(data['lmv_y_values2'])

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
video_path = "testvdo (3).mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

track_history = defaultdict(list)
frame_count = 0

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
                    track_history[track_id].append((x, y))

                    # Predict distance based on area
                    if area > 46200:
                        x_poly1_pred = poly_features1.transform([[area]])
                        predicted_y = model1.predict(x_poly1_pred)
                    else:
                        x_poly2_pred = poly_features2.transform([[area]])
                        predicted_y = model2.predict(x_poly2_pred)

                    
                    # Display predicted distance on the frame
                    cv2.putText(annotated_frame, f"Dist: {predicted_y[0]:.2f}", 
                                (int(x - w/2), int(y - h/2) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    if track_id not in printed_ids:
                        # Print track ID
                        print(f"Track ID: {track_id}, Area: {area}, Predicted Distance: {predicted_y[0]:.2f}")
                        printed_ids.add(track_id)

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
