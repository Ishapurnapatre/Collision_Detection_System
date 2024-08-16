import cv2

def get_video_fps(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video file opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return None
    
    # Get frames per second (FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Release the video capture object and close the video file
    cap.release()
    
    return fps

# Example usage:
video_path = 'testvdo (2).mp4'  # Replace with your video file path
fps = get_video_fps(video_path)

if fps is not None:
    print(f"FPS of the video: {fps}")
