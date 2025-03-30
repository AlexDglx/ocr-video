import cv2
import numpy as np

# Load the Canny edge-detected image
image = cv2.imread('canny.jpg', cv2.IMREAD_GRAYSCALE)

# Define the rectangle where artifacts will be removed
top_left = (50, 50)  # Example coordinates for the top-left corner
bottom_right = (300, 200)  # Example coordinates for the bottom-right corner

# Create a mask for the rectangle
mask = np.zeros_like(image)
cv2.rectangle(mask, top_left, bottom_right, 0, -1)  # White rectangle

# Combine the mask and the original image
cleaned_image = cv2.bitwise_and(image, cv2.bitwise_not(mask))

# Optional: Clean the rectangle area further
# Example: Apply morphological operations within the rectangle
roi = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cleaned_roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
cleaned_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = cleaned_roi

# Save or display the cleaned image
cv2.imshow('Cleaned Image', cleaned_image)
cv2.waitKey(0)
cv2.destroyAllWindows()