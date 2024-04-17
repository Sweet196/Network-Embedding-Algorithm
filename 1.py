import cv2
import numpy as np

def harris_corner_detection(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Harris corner detection
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    
    # Normalize Harris response for display
    cv2.normalize(corners, corners, 0, 255, cv2.NORM_MINMAX)
    
    return corners

def find_circles(image, corners, box):
    circles = []
    for corner in corners:
        x, y = corner[0], corner[1]
        # Check if the corner is inside the box
        if box[0][0] <= x <= box[1][0] and box[0][1] <= y <= box[1][1]:
            circles.append(np.array([x, y, 1]))  # Convert to homogeneous coordinates
    return circles

# Load the image
image = cv2.imread("image.jpg")

# Apply Harris corner detection
corners = harris_corner_detection(image)

# Define the coordinates of the two boxes (format: [(x1, y1), (x2, y2)])
box1 = [(100, 100), (200, 200)]  # Example coordinates for box 1
box2 = [(300, 300), (400, 400)]  # Example coordinates for box 2

# Find circles around detected corners in the two boxes
circles_box1 = find_circles(image, corners, box1)
circles_box2 = find_circles(image, corners, box2)

# Output the homogeneous coordinates of circles in the two boxes
print("Homogeneous coordinates of circles in box 1:")
for circle in circles_box1:
    print(circle)

print("\nHomogeneous coordinates of circles in box 2:")
for circle in circles_box2:
    print(circle)

# Display the Harris response image
cv2.imshow("Harris Response", corners)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
