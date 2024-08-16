import cv2
import numpy as np

def detect_edges(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges

def resize_image(image, scale_percent):
    # Calculate the new dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    new_dimensions = (width, height)
    
    # Resize the image
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    
    return resized_image

# Load the image
image_path = '600.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Detect edges
edges = detect_edges(image)

# Resize the edges image
scale_percent = 50  # Scale percentage (e.g., 50 means 50% of the original size)
resized_edges = resize_image(edges, scale_percent)

# Display the result
cv2.imshow('Edges', resized_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
