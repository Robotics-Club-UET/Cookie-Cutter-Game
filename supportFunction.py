import cv2
from cvzone import HandTrackingModule, overlayPNG
import numpy as np

# Define colors for visualization
color_true = (0, 255, 0)
color_false = (255, 255, 255)


def resize_image_with_padding(image, new_width, new_height):
    # Calculate the center of the original image
    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2

    # Calculate the padding needed around the original image
    delta_x = (new_width - image.shape[1]) // 2
    delta_y = (new_height - image.shape[0]) // 2

    # Create a new image with the desired size and keep the center of the original image
    result = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    result[delta_y:delta_y+image.shape[0],
           delta_x:delta_x+image.shape[1]] = image

    return result


def get_pixels_in_rectangle(image):
    def is_square(contour):
        # Approximate the shape of the contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # If the contour has close to 4 vertices (a square)
        return len(approx) == 4

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the Canny edge detection method to find edges
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the image
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours that resemble squares
    square_contours = [contour for contour in contours if is_square(contour)]
    smallest_square_contour = max(square_contours, key=cv2.contourArea)

    # Find the bounding rectangle around the contour
    x, y, w, h = cv2.boundingRect(smallest_square_contour)

    edge_pixels_in_rectangle = []

    error_range = 21  # Error range

    # Iterate over pixels on the top and bottom edges of the rectangle
    for i in range(x, x + w):
        for error_value in range(error_range):
            edge_pixels_in_rectangle.append([y + error_value, i])  # Top edge
            edge_pixels_in_rectangle.append(
                [y + h - 1 - error_value, i])  # Bottom edge

    # Iterate over pixels on the left and right edges of the rectangle (excluding 2 corners already processed)
    for i in range(y + 1, y + h - 3):
        for error_value in range(error_range):
            edge_pixels_in_rectangle.append([i, x + error_value])  # Left edge
            edge_pixels_in_rectangle.append(
                [i, x + w - 1 - error_value])  # Right edge

    return np.unique(np.array(edge_pixels_in_rectangle), axis=0).tolist()
