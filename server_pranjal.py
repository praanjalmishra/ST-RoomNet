import asyncio
import websockets
import cv2
import numpy as np
import torch
from roomnet import RoomNetModel  # Import RoomNet (assume pre-trained)
from depth_model import DepthModel  # Import a monocular depth estimation model

# Initialize models
roomnet_model = RoomNetModel(pretrained=True)
depth_model = DepthModel(pretrained=True)

async def handle_connection(websocket, path):
    print("Client connected")
    try:
        while True:
            # Receive data from client
            message = await websocket.recv()
            metadata_end_index = message.find(b'||')
            if metadata_end_index == -1:
                print("Invalid message format")
                continue

            metadata = message[:metadata_end_index].decode("utf-8")
            image_data = message[metadata_end_index + 2:]

            # Parse metadata
            camera_params = parse_metadata(metadata)  # Implement this function
            intrinsics = camera_params["intrinsics"]
            extrinsics = camera_params["extrinsics"]

            # Decode image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is not None:
                # Step 1: RoomNet processing for room edges
                room_edges = roomnet_model.predict(image)

                # Step 2: Monocular depth estimation
                depth_map = depth_model.predict(image)

                # Step 3: Reconstruct 3D room coordinates
                room_3d_coords = reconstruct_3d_coordinates(
                    room_edges, depth_map, intrinsics, extrinsics
                )

                # Step 4: Send back 3D room coordinates
                response = {
                    "room_3d_coords": room_3d_coords.tolist(),  # Convert to serializable format
                }
                await websocket.send(json.dumps(response))
                print("Sent 3D room coordinates to client.")
            else:
                print("Failed to decode image.")
    except websockets.ConnectionClosed:
        print("Client disconnected")
    finally:
        pass


# 3D Reconstruction
def reconstruct_3d_coordinates(edges, depth, intrinsics, extrinsics):
    """
    Combine room edges and depth map with camera parameters to generate 3D coordinates.
    """
    # Placeholder implementation:
    # Convert 2D edge coordinates + depth into 3D using camera intrinsics/extrinsics
    # For each edge pixel (u, v):
    #     depth_value = depth[v, u]
    #     Convert (u, v, depth_value) to (x, y, z) in world space
    height, width = depth.shape
    fx, fy = intrinsics["focal_length"]
    cx, cy = intrinsics["principal_point"]

    room_3d_coords = []
    for v in range(height):
        for u in range(width):
            if edges[v, u]:  # If pixel is part of the room edge
                depth_value = depth[v, u]
                x = (u - cx) * depth_value / fx
                y = (v - cy) * depth_value / fy
                z = depth_value
                # Transform to world coordinates using extrinsics
                world_coords = apply_extrinsics([x, y, z], extrinsics)
                room_3d_coords.append(world_coords)

    return np.array(room_3d_coords)


def apply_extrinsics(coord, extrinsics):
    """
    Apply extrinsic transformation to convert camera space to world space.
    """
    rotation = extrinsics["rotation"]
    position = extrinsics["position"]
    coord = np.dot(rotation, coord) + position
    return coord
