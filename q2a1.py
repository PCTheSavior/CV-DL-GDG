import cv2
import numpy as np


# Function to capture an image from webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        print("Error: Failed to capture image.")
        return None


# Part a: Convert to grayscale
def grayscale_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


# Part b.i: Thresholding to black and white (2 colors)
def threshold_binary(image, threshold_value=127):
    _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary


# Part b.ii: Thresholding to 16 levels of gray
def threshold_16_gray(image):
    # Divide pixel intensity range into 16 bins (0-255 mapped to 8 regions)
    levels = image // 16 * 16
    return levels


# Part c: Sobel filter and Canny edge detector
def edge_detection(image):
    # Sobel filter
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)

    # Canny edge detector
    canny_edges = cv2.Canny(image, 50, 150)

    return sobel_combined, canny_edges


# Part d: Gaussian filtering to remove noise
def gaussian_filter(image, kernel_size=5):
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred


# Part e: Sharpen the image using a kernel
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


# Part f: Convert RGB to BGR
def convert_rgb_to_bgr(image):
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return bgr_image


# Main function to execute all tasks
def main():
    # Capture an image
    image = capture_image()
    if image is None:
        return

    # Task a: Grayscale
    gray = grayscale_image(image)

    # Task b.i: Binary thresholding
    binary = threshold_binary(gray)

    # Task b.ii: 16-level grayscale
    gray_16 = threshold_16_gray(gray)

    # Task c: Sobel and Canny edge detection
    sobel, canny = edge_detection(gray)

    # Task d: Gaussian blur
    blurred = gaussian_filter(gray)

    # Task e: Sharpen the blurred image
    sharpened = sharpen_image(blurred)

    # Task f: Convert RGB to BGR
    bgr_image = convert_rgb_to_bgr(image)

    # Display the results in a 2x4 grid
    cv2.imshow("Original", image)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Binary Threshold", binary)
    cv2.imshow("16-Level Grayscale", gray_16)
    cv2.imshow("Sobel Edge Detection", sobel / sobel.max() * 255)  # Normalize for display
    cv2.imshow("Canny Edge Detection", canny)
    cv2.imshow("Gaussian Blur", blurred)
    cv2.imshow("Sharpened", sharpened)

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
