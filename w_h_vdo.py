# to find Height and width of a frame

import cv2

video_path = "test (2).mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    ret, frame = cap.read()

    if ret:
        height, width, channels = frame.shape
        print(f'Width: {width}')
        print(f'Height: {height}')
    else:
        print("Error: Could not read frame from video.")

    cap.release()