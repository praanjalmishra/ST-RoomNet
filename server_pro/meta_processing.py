import cv2
import numpy as np
from meta_read import parse_metadata_csv

# Dummy placeholders for RoomNet and depth estimation models
roomnet_model = RoomNetModel(pretrained=True)  # Replace with actual model
depth_model = DepthModel(pretrained=True)      # Replace with actual model

def process_image(metadata):
    """
    Processes a single image to extract room edges, depth map, and 3D room coordinates.
    :param metadata: Dictionary containing image path, intrinsics, and extrinsics.
    :return: 3D coordinates of the room.
    """
    # Load image
    image = cv2.imread(metadata["image_path"])
    if image is None:
        print(f"Failed to load image: {metadata['image_path']}")
        return None

    # Extract room edges using RoomNet
    room_edges = roomnet_model.predict(image)  # Output: Binary edge map

    # Estimate depth using the depth model
    depth_map = depth_model.predict(image)  # Output: Depth map

    # Reconstruct 3D room coordinates
    room_3d_coords = reconstruct_3d_coordinates(
        room_edges, depth_map, metadata["intrinsics"], metadata["extrinsics"]
    )

    return room_3d_coords


def reconstruct_3d_coordinates(edges, depth, intrinsics, extrinsics):
    """
    Reconstructs 3D coordinates of room edges from depth map and camera parameters.
    :param edges: Binary edge map of the room.
    :param depth: Depth map corresponding to the image.
    :param intrinsics: Dictionary of camera intrinsic parameters.
    :param extrinsics: Dictionary of camera extrinsic parameters.
    :return: List of 3D room coordinates.
    """
    height, width = depth.shape
    fx, fy = intrinsics["focal_length"]
    cx, cy = intrinsics["principal_point"]

    room_3d_coords = []
    for v in range(height):
        for u in range(width):
            if edges[v, u]:  # Process edge pixels only
                depth_value = depth[v, u]
                if depth_value == 0:  # Ignore invalid depth values
                    continue

                # Convert to camera coordinates
                x = (u - cx) * depth_value / fx
                y = (v - cy) * depth_value / fy
                z = depth_value

                # Transform to world coordinates using extrinsics
                world_coords = apply_extrinsics([x, y, z], extrinsics)
                room_3d_coords.append(world_coords)

    return np.array(room_3d_coords)

def apply_extrinsics(coord, extrinsics):
    """
    Applies extrinsic transformation to convert camera space to world space.
    :param coord: 3D point in camera space.
    :param extrinsics: Dictionary containing rotation and position.
    :return: 3D point in world space.
    """
    rotation = extrinsics["rotation"]
    position = extrinsics["position"]

    # Convert rotation quaternion to matrix (placeholder, replace with actual implementation)
    rotation_matrix = quaternion_to_matrix(rotation)

    # Transform coordinate
    coord = np.dot(rotation_matrix, coord) + position
    return coord

def quaternion_to_matrix(quaternion):
    """
    Converts a quaternion to a 3x3 rotation matrix.
    :param quaternion: Quaternion as (x, y, z, w).
    :return: 3x3 rotation matrix.
    """
    x, y, z, w = quaternion
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])


def main():
    metadata_file = "metadata.csv"
    metadata_list = parse_metadata_csv(metadata_file)

    results = {}
    for metadata in metadata_list:
        print(f"Processing {metadata['image_path']}...")
        room_3d_coords = process_image(metadata)
        if room_3d_coords is not None:
            results[metadata["image_path"]] = room_3d_coords

    # Save or send results
    for image_path, coords in results.items():
        output_path = image_path.replace("received_data", "processed_data").replace(".jpg", "_3d.json")
        np.savetxt(output_path, coords, delimiter=",")
        print(f"Saved 3D coordinates to {output_path}")

if __name__ == "__main__":
    main()
