"""Image processing utilities for computer vision tasks."""

import cv2
import numpy as np


def preprocess_image(
    image: np.ndarray, img_height: int = 66, img_width: int = 200
) -> np.ndarray:
    """
    Preprocess image for neural network input.

    Applies cropping, color space conversion, Gaussian blur, and normalization.

    Args:
        image: Input image (numpy array)
        img_height: Target height for resized image
        img_width: Target width for resized image

    Returns:
        Preprocessed image normalized to [0, 1] range
    """
    # Crop top and bottom portions (remove sky and hood)
    image = image[60:135, :, :]

    # Convert RGB to YUV color space (better for neural networks)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # Apply Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # Resize to target dimensions
    image = cv2.resize(image, (img_width, img_height))

    # Normalize to [0, 1] range
    image = image / 255.0

    return image


def canny_edge_detection(
    image: np.ndarray,
    blur_kernel: tuple = (5, 5),
    threshold_low: int = 50,
    threshold_high: int = 150,
) -> np.ndarray:
    """
    Detect edges using Canny edge detection algorithm.

    Args:
        image: Input image (BGR or grayscale)
        blur_kernel: Gaussian blur kernel size
        threshold_low: Lower threshold for Canny edge detection
        threshold_high: Upper threshold for Canny edge detection

    Returns:
        Canny edge detection result
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold_low, threshold_high)

    return edges


def region_of_interest(image: np.ndarray, polygon: np.ndarray = None) -> np.ndarray:
    """
    Apply region of interest (ROI) mask to image.

    Extracts a triangular region of interest from the image.
    Default triangle focuses on the road ahead.

    Args:
        image: Input image
        polygon: Custom polygon vertices. If None, uses default road-focused triangle.
                Shape should be (N, 2) where N is number of vertices.

    Returns:
        Masked image with ROI extracted
    """
    height, width = image.shape[:2]

    if polygon is None:
        # Default triangle polygon for road detection
        polygon = np.array(
            [[(200, height), (width - 200, height), (width // 2, int(height * 0.4))]]
        )

    # Create mask
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon.astype(np.int32), 255)

    # Apply bitwise AND to extract ROI
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image
