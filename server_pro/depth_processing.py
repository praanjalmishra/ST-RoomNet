import cv2
import numpy as np
from .inference_depth import load_depth_model, infer_depth

# Camera Intrinsic Parameters
CAMERA_INTRINSICS = {
    'fx': 506.91,  # Focal length in x-direction
    'fy': 506.72,  # Focal length in y-direction
    'cx': 320.42,  # Principal point x-coordinate
    'cy': 243.13   # Principal point y-coordinate
}

# Function to unproject 2D edge points into 3D (Camera Frame)
def unproject_2d_to_3d(edge_points, depth_map, intrinsics):
    """
    Unprojects 2D edge points into 3D points in the camera's local frame.
    :param edge_points: List of 2D edge points [(u1, v1), (u2, v2), ...].
    :param depth_map: 2D numpy array containing depth values.
    :param intrinsics: Dictionary containing camera intrinsics: 'fx', 'fy', 'cx', 'cy'.
    :return: List of 3D points [(X1, Y1, Z1), (X2, Y2, Z2), ...].
    """
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    points_3d = []
    for u, v in edge_points:
        if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
            depth = depth_map[int(v), int(u)]  # Depth value at pixel (u, v)
            if depth > 0:  # Ensure valid depth
                X = (u - cx) * depth / fx
                Y = (v - cy) * depth / fy
                Z = depth
                points_3d.append((X, Y, Z))
            else:
                print(f"Invalid depth at ({u}, {v}): {depth}")
                points_3d.append(None)  # Invalid depth
        else:
            print(f"Out-of-bounds point: ({u}, {v})")
            points_3d.append(None)  # Out of bounds

    return points_3d


# Function to transform 3D points from Camera Frame to World Frame
def transform_to_world_frame(points_3d, rotation, translation):
    """
    Transforms 3D points from the camera's frame to the world frame using extrinsic parameters.
    :param points_3d: List of 3D points in the camera's local frame.
    :param rotation: 3x3 rotation matrix or 4-element quaternion.
    :param translation: 3x1 translation vector [Tx, Ty, Tz].
    :return: List of 3D points in the world frame.
    """
    # Convert quaternion to rotation matrix if necessary
    if len(rotation) == 4:  # Quaternion [x, y, z, w]
        rotation = quaternion_to_matrix(rotation)

    points_world = []
    for point in points_3d:
        if point:
            point_cam = np.array(point).reshape(3, 1)  # Column vector
            point_world = np.dot(rotation, point_cam) + np.array(translation).reshape(3, 1)
            points_world.append(tuple(point_world.flatten()))
        else:
            points_world.append(None)  # Invalid point

    return points_world

# Function to convert quaternion to rotation matrix
def quaternion_to_matrix(q):
    """
    Converts a quaternion [x, y, z, w] to a 3x3 rotation matrix.
    :param q: Quaternion as [x, y, z, w].
    :return: 3x3 rotation matrix.
    """
    x, y, z, w = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

# Main function to perform depth inference and unprojection with extrinsics
def process_image_with_depth_and_extrinsics(image_path, edge_points, encoder='vitl', input_size=518, extrinsics=None):
    """
    Infers depth for an image, unprojects 2D edge points into 3D points, and transforms them to the world frame.
    :param image_path: Path to the input image.
    :param edge_points: List of 2D edge points [(u1, v1), (u2, v2), ...].
    :param encoder: Depth model encoder ('vits', 'vitb', 'vitl', 'vitg').
    :param input_size: Input size for the depth model.
    :param extrinsics: Dictionary with 'rotation' (matrix or quaternion) and 'translation' vector.
    :return: List of 3D points in the world frame.
    """
    # Load the depth model
    model = load_depth_model(encoder)

    # Load the image
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Infer the depth map
    depth_map = infer_depth(raw_image, model, input_size, grayscale=True)

    # Normalize depth map (assume depth map is scaled to [0, 255])
    depth_map = depth_map.astype(np.float32) / 255.0  # Scale to 0–1

    # Unproject 2D points into 3D (Camera Frame)
    points_3d_camera = unproject_2d_to_3d(edge_points, depth_map, CAMERA_INTRINSICS)

    # Transform 3D points to World Frame
    if extrinsics:
        points_3d_world = transform_to_world_frame(
            points_3d_camera,
            rotation=extrinsics['rotation'],
            translation=extrinsics['translation']
        )
        return points_3d_world
    else:
        return points_3d_camera  # Return Camera Frame if no extrinsics are provided




# from depth_processing import process_image_with_depth_and_extrinsics

# # Path to the image
# image_path = "/path/to/image.jpg"

# # Example 2D edge points from ST RoomNet
# edge_points = [
#     (100, 150),  # Example 2D point (u, v)
#     (200, 250),
#     (300, 350)
# ]

# # Extrinsics for the camera
# extrinsics = {
#     'rotation': [0.0, -0.707, 0.0, 0.707],  # Example quaternion (x, y, z, w)
#     'translation': [1.0, 0.5, 2.0]          # Example translation (Tx, Ty, Tz)
# }

# # Process the image to get 3D points in the world frame
# encoder = 'vitl'  # Depth model encoder
# points_3d_world = process_image_with_depth_and_extrinsics(
#     image_path,
#     edge_points,
#     encoder=encoder,
#     input_size=518,
#     extrinsics=extrinsics
# )

# # Print the resulting 3D points
# print("Transformed 3D Points (World Frame):")
# for i, point in enumerate(points_3d_world):
#     if point:
#         print(f"2D Point {edge_points[i]} -> 3D Point {point}")
#     else:
#         print(f"2D Point {edge_points[i]} -> No valid depth")
