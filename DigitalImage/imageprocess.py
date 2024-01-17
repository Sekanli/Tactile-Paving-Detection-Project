import cv2
import numpy as np
from sklearn.cluster import DBSCAN



def convert_rgb_to_hsv(rgb_values):
    color = np.uint8([[rgb_values]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    return hsv_color[0][0]

def adjust_for_brightness(avg_brightness, base_values):
    adjusted_values = base_values.copy()
    
    if avg_brightness < 150:  # Considered low brightness
        adjusted_values['base_proportion'] *= 0.6
        adjusted_values['adjustment_value'] -= 20
    elif avg_brightness > 180:  # Considered high brightness
        adjusted_values['base_proportion'] *= 1.2
        adjusted_values['adjustment_value'] -= 10
    # No adjustment for mid-range brightness levels
    
    return adjusted_values
def adjust_for_contrast(contrast, adjusted_values):
    if contrast < 15:  # Low contrast
        adjusted_values['base_proportion'] *= 0.5
        adjusted_values['adjustment_value'] -= 10
    elif contrast > 30:  # High contrast
        adjusted_values['base_proportion'] *= 1.1
        adjusted_values['adjustment_value'] -= 30
    # No adjustment for mid-range contrast levels
    
    return adjusted_values


def calculate_contrast(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(grayscale)

def calculate_average_brightness(hsv):
    return np.mean(hsv[:, :, 2])

def dynamic_hsv_adjustment(avg_brightness, base_hsv_range):
    # Adjust the hue range based on brightness
    hue_adjustment = int(10 * (1 - avg_brightness / 255))  # Example adjustment
    lower_hsv = base_hsv_range[0] - [hue_adjustment, 0, 0]
    upper_hsv = base_hsv_range[1] + [hue_adjustment, 0, 0]
    
    # Ensure HSV values are within valid range
    lower_hsv = np.clip(lower_hsv, 0, [180, 255, 255])
    upper_hsv = np.clip(upper_hsv, 0, [180, 255, 255])
    
    return (lower_hsv, upper_hsv)

def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))  # Adjusted parameters
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return enhanced_frame




def apply_bilateral_filter(frame, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(frame, d, sigmaColor, sigmaSpace)


def isolate_color_range(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    avg_brightness = calculate_average_brightness(hsv)
    contrast = calculate_contrast(frame)
    
    # Adjust base values based on brightness and contrast
    adjusted_values = adjust_for_brightness(avg_brightness, base_values)
    adjusted_values = adjust_for_contrast(contrast, adjusted_values)
    
    final_mask = np.zeros(hsv.shape[:2], dtype="uint8")
    for base_range in base_ranges:
        # Adjust the range based on brightness and other factors
        adjusted_range = dynamic_hsv_adjustment(avg_brightness, base_range)

        # Create a mask for the current adjusted range
        mask = cv2.inRange(hsv, adjusted_range[0], adjusted_range[1])

        # Combine the current mask with the final mask
        final_mask = cv2.bitwise_or(final_mask, mask)
    
    # Apply the final mask to the frame to isolate the colors
    result = cv2.bitwise_and(frame, frame, mask=final_mask)
    return result




# Example RGB colors
full_yellow_rgb = [255, 255, 0]
shadowed_yellow_rgb = [164, 145, 159]
additional_color_rgb = [255, 128, 0]
x1_rgb = [114,92,108]
# Convert the RGB colors to HSV
full_yellow_hsv = convert_rgb_to_hsv(full_yellow_rgb)
shadowed_yellow_hsv = convert_rgb_to_hsv(shadowed_yellow_rgb)
additional_color_hsv = convert_rgb_to_hsv(additional_color_rgb)
x1_rgb = convert_rgb_to_hsv(x1_rgb)
# Base values
base_values = {
    'base_proportion': 0.3,
    'adjustment_value':5,
}
# Define your base ranges
base_ranges = [
    # Broadened Yellow Range
    (np.array([25, 50, 50]), np.array([35, 255, 255])),
    
    # Broadened Orange Range
    (np.array([10, 50, 50]), np.array([30, 255, 255])),
    
    # Your additional custom ranges
    (np.array([full_yellow_hsv[0] - 10, 50, 50]), np.array([full_yellow_hsv[0] + 10, 255, 255])),
    (np.array([shadowed_yellow_hsv[0] - 10, 50, 50]), np.array([shadowed_yellow_hsv[0] + 10, 255, 255])),
    (np.array([additional_color_hsv[0] - 10, 50, 50]), np.array([additional_color_hsv[0] + 10, 255, 255])),
    (np.array([x1_rgb[0] - 10, 50, 50]), np.array([x1_rgb[0] + 10, 255, 255])),
]

def detect_and_draw_lines(edge_image):
    # Use Hough Line Transform to detect lines in the edge-detected image
    # Step 1: Detect lines using the Hough Transform
    lines = cv2.HoughLinesP(edge_image, 1, np.pi / 180, threshold=120, minLineLength=70, maxLineGap=7)

    # Step 2: Create a mask where the detected lines are drawn
    line_mask = np.zeros(edge_image.shape, dtype=np.uint8)  # Create a mask the same size as edges
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness = 3)  # Draw the line on the mask

    # Step 3: Keep only the edges that correspond to the straight lines
    filtered_edges = cv2.bitwise_and(edge_image, line_mask)
    return filtered_edges

def apply_gaussian_blur(frame, kernel_size=(5, 5), sigmaX=2):
    # Apply Gaussian Blur to the frame
    blurred_frame = cv2.GaussianBlur(frame, kernel_size, sigmaX)
    return blurred_frame

def remove_noise(edges):
    # Define a kernel size for morphological operations
    kernel = np.ones((5,5), np.uint8)
    
    # Apply opening to remove small dots/noise
    opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    opening = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    opening = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    opening = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return opening


def line_meets_criteria(line):
    # Implement any criteria here to filter lines, e.g., orientation, length
    x1, y1, x2, y2 = line
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return length > 50  # Example criterion: line length

def angle_between_lines(line1, line2):
    # Calculate the angle between two lines
    _, _, x1, y1 = line1[0]
    _, _, x2, y2 = line2[0]
    angle1 = np.arctan2(y1, x1)
    angle2 = np.arctan2(y2, x2)
    return np.abs(angle1 - angle2)

def is_parallel(line1, line2, angle_threshold=10):
    """ Check if two lines are parallel within a certain angle threshold. """
    angle1 = np.arctan2(line1[3] - line1[1], line1[2] - line1[0]) * 180 / np.pi
    angle2 = np.arctan2(line2[3] - line2[1], line2[2] - line2[0]) * 180 / np.pi
    return abs(angle1 - angle2) < angle_threshold

def filter_for_road_borders(lines, angle_threshold=10, min_length=200):
    """ Filter lines based on their orientation, parallelism, and length. """
    filtered_lines = []
    for line in lines:
        if (line[0][2] - line[0][0])**2 + (line[0][3] - line[0][1])**2 >= min_length**2:
            filtered_lines.append(line)
    
    # Further filter for parallel lines
    parallel_lines = []
    for i in range(len(filtered_lines)):
        for j in range(i + 1, len(filtered_lines)):
            if is_parallel(filtered_lines[i][0], filtered_lines[j][0], angle_threshold):
                parallel_lines.append(filtered_lines[i])
                break  # Break to avoid adding the same line multiple times

    return parallel_lines

def detect_road_borders_and_create_mask(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=130, maxLineGap=10)
    if lines is None:
        return edges, np.zeros(edges.shape, dtype=np.uint8)

    # Filter lines for potential road borders
    road_lines = filter_for_road_borders(lines)

    if len(road_lines) < 5:
        return edges, np.zeros(edges.shape, dtype=np.uint8)

    # Prepare data for clustering (using midpoints of lines)
    midpoints = np.array([[(line[0][0] + line[0][2]) / 2, (line[0][1] + line[0][3]) / 2] for line in road_lines])

    # DBSCAN clustering
    clustering = DBSCAN(eps=50, min_samples=2).fit(midpoints)
    labels = clustering.labels_

    road_mask = np.zeros(edges.shape, dtype=np.uint8)
    unique_labels = set(labels)
    for k in unique_labels:
        if k == -1:  # Noise in DBSCAN
            continue

        class_member_mask = (labels == k)
        
        # Use lines in this cluster to create the mask
        for line, included in zip(road_lines, class_member_mask):
            if included:
                x1, y1, x2, y2 = line[0]
                cv2.line(road_mask, (x1, y1), (x2, y2), 255, thickness=3)

    # Fill in the area enclosed by the road borders
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(road_mask, contours, -1, (255), thickness=cv2.FILLED)

    # Optional: Dilate the mask
    kernel = np.ones((5, 5), np.uint8)
    road_mask = cv2.dilate(road_mask, kernel, iterations=2)

    road_edges = cv2.bitwise_and(edges, road_mask)
    return road_edges, road_mask
def is_cluster_too_regular(cluster_points):
    # Implement a check for regularity in the cluster
    if len(cluster_points) < 2:
        return False
    distances = np.linalg.norm(np.diff(cluster_points, axis=0), axis=1)
    if np.std(distances) < 10:  # Threshold for regularity
        return True
    return False



def slope(line):
    x1, y1, x2, y2 = line[0]
    return (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')

def cluster_lines(lines):
    left_lines = []
    right_lines = []
    for line in lines:
        m = slope(line)
        # Assuming that for left lines slope will be positive and for right lines, it will be negative
        if m > 0:
            left_lines.append(line)
        elif m < 0:
            right_lines.append(line)
    return left_lines, right_lines

def select_representative_line(lines):
    if not lines:
        return None
    # Simple approach: average the start and end points of the lines
    avg_line = np.mean(np.array([line[0] for line in lines]), axis=0).astype(int)
    return avg_line


def apply_edge_detection(frame):
    # Adjust these parameters as needed
    blur_kernel_size = (5, 5)
    # Lower these values to increase sensitivity
    canny_lower_threshold = 18 # Try reducing these values
    canny_upper_threshold = 18  # Try reducing these values
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, blur_kernel_size, 0)
    edges = cv2.Canny(blur, canny_lower_threshold, canny_upper_threshold)
    
    # Dilation to make edges thicker and more connected
    kernel = np.ones((3,3), np.uint8)
    altered_edges = cv2.dilate(edges, kernel, iterations=1)
    #altered_edges = cv2.erode(altered_edges,kernel,iterations=1)
    
    
    return altered_edges


def line_to_line_distance(line1, line2):
    # Extract points from the lines
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # Calculate midpoints of the lines
    mid_x1, mid_y1 = (x1 + x2) / 2, (y1 + y2) / 2
    mid_x2, mid_y2 = (x3 + x4) / 2, (y3 + y4) / 2
    
    # Calculate the distance between midpoints
    distance = np.sqrt((mid_x2 - mid_x1) ** 2 + (mid_y2 - mid_y1) ** 2)
    return distance





def clear_noise(edges, min_contour_area=10):
   

    # Morphological opening to remove small noise points
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    # Filter out small contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_contour_area:
            cv2.drawContours(edges, [cnt], -1, (0), thickness=cv2.FILLED)

    # Apply a slight blur to reduce noise
    edges = cv2.GaussianBlur(edges, (3, 3), 0)

    # Can be added if further edge refinement is needed
    # edges = cv2.Canny(edges, lower_threshold, upper_threshold, L2gradient=True)

    return edges