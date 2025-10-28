"""Lane detection utilities for extracting road lines from images."""

import cv2
import numpy as np


def calculate_line_parameters(x1: int, y1: int, x2: int, y2: int) -> tuple:
    """
    Calculate slope and intercept for a line defined by two points.

    Args:
        x1, y1: First point coordinates
        x2, y2: Second point coordinates

    Returns:
        Tuple of (slope, intercept)
    """
    coefficients = np.polyfit((x1, x2), (y1, y2), 1)
    slope = coefficients[0]
    intercept = coefficients[1]
    return slope, intercept


def make_coordinates(image: np.ndarray, line_parameters: tuple) -> np.ndarray:
    """
    Convert line parameters (slope, intercept) to image coordinates.

    Creates a line that spans from the bottom of the image to 3/5 of its height.

    Args:
        image: Reference image (used for dimensions)
        line_parameters: Tuple of (slope, intercept)

    Returns:
        Array of [x1, y1, x2, y2] coordinates
    """
    slope, intercept = line_parameters

    y1 = image.shape[0]  # Bottom of image
    y2 = int(y1 * (3 / 5))  # 3/5 of height

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """
    Average lines by separating left and right lane lines.

    Classifies lines based on slope (negative = left lane, positive = right lane),
    then averages parameters within each group.

    Args:
        image: Reference image (used for dimensions)
        lines: Array of detected lines in format [[x1, y1, x2, y2], ...]

    Returns:
        Array of averaged lines for left and right lanes
    """
    left_fit = []
    right_fit = []

    if lines is None:
        return np.array([])

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        slope, intercept = calculate_line_parameters(x1, y1, x2, y2)

        # Classify as left (negative slope) or right (positive slope)
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    # Calculate averages
    averaged_lines = []

    if left_fit:
        left_fit_avg = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_avg)
        averaged_lines.append(left_line)

    if right_fit:
        right_fit_avg = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_avg)
        averaged_lines.append(right_line)

    return np.array(averaged_lines)


def display_lines(
    image: np.ndarray,
    lines: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 10,
) -> np.ndarray:
    """
    Draw detected lines on a blank image.

    Args:
        image: Reference image for dimensions
        lines: Array of lines in format [[x1, y1, x2, y2], ...]
        color: Line color in BGR format (default: green)
        thickness: Line thickness in pixels

    Returns:
        Image with lines drawn
    """
    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)

    return line_image


def detect_lanes(
    image: np.ndarray,
    rho: float = 2,
    theta: float = np.pi / 180,
    threshold: int = 100,
    min_line_length: int = 40,
    max_line_gap: int = 5,
) -> np.ndarray:
    """
    Complete lane detection pipeline using Hough line transform.

    Args:
        image: Input image (should be edge-detected, typically from Canny)
        rho: Distance resolution of the accumulator in pixels
        theta: Angle resolution of the accumulator in radians
        threshold: Accumulator threshold parameter
        min_line_length: Minimum line length threshold
        max_line_gap: Maximum gap allowed between line points

    Returns:
        Array of detected lines
    """
    lines = cv2.HoughLinesP(
        image,
        rho=rho,
        theta=theta,
        threshold=threshold,
        lines=np.array([]),
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    return lines
