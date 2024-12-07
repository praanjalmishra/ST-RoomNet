import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def calculate_angle(line):
    x1, y1, x2, y2 = line[0]
    return math.atan2(y2 - y1, x2 - x1)

def calculate_distance(line1, line2):
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    dist1 = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
    dist2 = math.sqrt((x4 - x2) ** 2 + (y4 - y2) ** 2)
    return min(dist1, dist2)

# Function to merge two lines
def merge_lines(line1, line2):
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    merged_line = [[min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4)]]
    return np.array(merged_line)

def calculate_length(line):
    x1, y1, x2, y2 = line[0]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def merge_nearby_lines(lines, distance_threshold=40, angle_threshold=10, length_threshold=100):
    merged_lines = []

    for line in lines:
        if calculate_length(line) < length_threshold:
            continue

        merged = False
        for i, existing_line in enumerate(merged_lines):
            angle1 = calculate_angle(line)
            angle2 = calculate_angle(existing_line)
            distance = calculate_distance(line, existing_line)

            angle_diff = abs(angle1 - angle2)
            if angle_diff > math.pi:  # Adjust for angles wrapping around
                angle_diff = 2 * math.pi - angle_diff

            if angle_diff < math.radians(angle_threshold) and distance < distance_threshold:
                merged_lines[i] = merge_lines(existing_line, line)
                merged = True
                break

        if not merged:
            merged_lines.append(line)

    return np.array(merged_lines)

edges = cv2.imread('enhanced_edges_plot.png', cv2.IMREAD_GRAYSCALE)
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=20, maxLineGap=10)

merged_lines = merge_nearby_lines(lines)

# Visualize merged lines
line_image = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
for line in merged_lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw in red

# Overlay with original edge-detected image
overlay = cv2.addWeighted(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.7, line_image, 1, 0)

# Display results
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title('Merged Lines')
plt.axis('off')
plt.show()

# Print merged line segments
merged_line_segments = [(line[0][0], line[0][1], line[0][2], line[0][3]) for line in merged_lines]
print("Merged Line Segments:", merged_line_segments)

