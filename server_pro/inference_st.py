import os
import cv2
import numpy as np
import tensorflow as tf
import math
from tensorflow.keras.applications.convnext import ConvNeXtTiny, preprocess_input
from tensorflow.keras.models import Model
from ST_RoomNet.spatial_transformer import ProjectiveTransformer

# Constants
MODEL_WEIGHTS = '/home/scai/mr/ST-RoomNet/Weight_ST_RroomNet_ConvNext.h5'
REF_IMAGE_PATH = '/home/scai/mr/server_ml/ST_RoomNet/ref_img2.png'

# Function to load and preprocess images
def load_and_preprocess_image(image_path, target_size=(400, 400)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return preprocess_input(img[np.newaxis, ...])

# Function to initialize RoomNet model
def load_roomnet_model():
    ref_img = tf.io.read_file(REF_IMAGE_PATH)
    ref_img = tf.io.decode_png(ref_img)
    ref_img = tf.cast(ref_img, tf.float32) / 51.0
    ref_img = ref_img[tf.newaxis, ...]
    
    base_model = ConvNeXtTiny(include_top=False, weights="imagenet", input_shape=(400, 400, 3), pooling='avg')
    theta = tf.keras.layers.Dense(8)(base_model.output)
    stl = ProjectiveTransformer((400, 400)).transform(ref_img, theta)
    model = Model(base_model.input, stl)
    model.load_weights(MODEL_WEIGHTS)
    return model

# Function to perform RoomNet inference
def roomnet_inference(model, image):
    out = model.predict(image)
    out = np.rint(out[0, :, :, 0])  # Round to nearest integer for segmentation
    return out

# Function to postprocess the segmented image
def postprocess_segmentation(segmented_image):
    blurred_image = cv2.GaussianBlur(segmented_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    enhanced_edges = cv2.dilate(edges, kernel, iterations=1)
    return enhanced_edges

# Function to calculate line angle
def calculate_angle(line):
    x1, y1, x2, y2 = line[0]
    return math.atan2(y2 - y1, x2 - x1)

# Function to calculate distance between two lines
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

# Function to calculate line length
def calculate_length(line):
    x1, y1, x2, y2 = line[0]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to merge nearby lines
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
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff

            if angle_diff < math.radians(angle_threshold) and distance < distance_threshold:
                merged_lines[i] = merge_lines(existing_line, line)
                merged = True
                break

        if not merged:
            merged_lines.append(line)

    return np.array(merged_lines)

# Main function to process an image and return line endpoints
def roomnet_process_image(image_path):
    """
    Process an input image to extract RoomNet line endpoints.
    :param image_path: Path to the input image.
    :return: List of dictionaries containing start and end points of lines.
    """
    # Load and preprocess the input image
    img = load_and_preprocess_image(image_path)
    
    # Load RoomNet model (global constants for weights and ref image path are used)
    model = load_roomnet_model()
    
    # Perform RoomNet inference
    segmented_output = roomnet_inference(model, img)

    print("############Segmented output roomnet", segmented_output.shape)

    segmented_output = cv2.resize(segmented_output, (640, 480), interpolation=cv2.INTER_NEAREST)
    print(f"Resized RoomNet Output Shape: {segmented_output.shape}")
    
    # Postprocess segmented image
    segmented_image = (segmented_output * (255 / segmented_output.max())).astype(np.uint8)  # Normalize for visualization
    enhanced_edges = postprocess_segmentation(segmented_image)
    
    # Detect and merge lines
    lines = cv2.HoughLinesP(enhanced_edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=20, maxLineGap=10)
    if lines is not None:
        merged_lines = merge_nearby_lines(lines)
        # Extract endpoints as dictionaries
        line_endpoints = [{"start": [line[0][0], line[0][1]], "end": [line[0][2], line[0][3]]} for line in merged_lines]
    else:
        line_endpoints = []

    return line_endpoints




if __name__ == "__main__":
    # Input paths
    IMAGE_PATH = '/home/scai/mr/surface_relabel/train/0a5c601a476b916a5a2b09513c301e52b2a92afc.jpg'


    # Process the image
    endpoints = roomnet_process_image(IMAGE_PATH)
    
    # Print the results
    print("Merged Line Endpoints:", endpoints)

