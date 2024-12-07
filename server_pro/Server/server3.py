import asyncio
import websockets
import cv2
import numpy as np
import json
import os
from pathlib import Path  # Import Path for path normalization
import threading

# Global flag for exiting the server
exit_flag = False

# Paths for saving received data
BASE_DIR = "received_data"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.json")

# Ensure directories exist
os.makedirs(IMAGE_DIR, exist_ok=True)

# Global metadata storage
metadata_storage = {"images": []}


def save_metadata():
    """Save the metadata to a JSON file."""
    with open(METADATA_PATH, "w") as metadata_file:
        json.dump(metadata_storage, metadata_file, indent=4)
        print(f"Metadata saved at {METADATA_PATH}")


def keyboard_listener():
    """Listen for 'Ctrl + E' to exit the server."""
    global exit_flag
    print("Press 'Ctrl + E' to exit the server.")
    while not exit_flag:
        key = input()
        if key.lower() == '\x05':  # Ctrl + E to exit
            exit_flag = True
            print("Exiting server...")


async def handle_connection(websocket, path):
    """Handle incoming WebSocket connections."""
    global exit_flag
    global metadata_storage
    image_count = 1  # Counter for naming images

    print("Client connected")

    try:
        while not exit_flag:
            message = await websocket.recv()

            if message:
                # Separate metadata and image data
                metadata_end_index = message.find(b'||')
                if metadata_end_index == -1:
                    print("Invalid message format, missing metadata delimiter")
                    continue

                raw_metadata = message[:metadata_end_index].decode("utf-8")
                image_data = message[metadata_end_index + 2:]

                print(f"Raw Metadata: {raw_metadata}")
                print(f"Image data size: {len(image_data)} bytes")

                # Save image
                image_filename = f"image_{image_count}.jpg"
                image_path = os.path.join(IMAGE_DIR, image_filename)
                with open(image_path, "wb") as image_file:
                    image_file.write(image_data)
                print(f"Image saved at {image_path}")

                # Normalize path to use forward slashes
                normalized_image_path = Path(image_path).as_posix()

                # Parse metadata
                try:
                    metadata_json = json.loads(raw_metadata)
                    metadata_json["path"] = normalized_image_path  # Use normalized path
                    metadata_storage["images"].append(metadata_json)

                    # Save updated metadata
                    save_metadata()
                except json.JSONDecodeError as e:
                    print(f"Failed to parse metadata: {e}")

                # Display the received image
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is not None:
                    cv2.imshow("Received Image", image)
                    cv2.waitKey(1)
                else:
                    print("Failed to decode image.")

                # Send acknowledgment
                response = f"Server processed image and metadata for {image_filename}"
                await websocket.send(response)
                print(f"Sent back to client: {response}")

                image_count += 1
    except websockets.ConnectionClosed:
        print("Client disconnected")
    finally:
        cv2.destroyAllWindows()


async def main():
    """Main coroutine to start the WebSocket server."""
    global exit_flag

    start_server = websockets.serve(handle_connection, "0.0.0.0", 8765)

    print("WebSocket server running on ws://localhost:8765")
    print("Press 'Ctrl + E' to exit the server.")

    # Start the WebSocket server
    server = await start_server

    while not exit_flag:
        await asyncio.sleep(1)

    server.close()
    await server.wait_closed()
    print("Server shutdown.")


# Start keyboard listener in a separate thread
keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
keyboard_thread.start()

# Run the server
try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("Server stopped by keyboard interrupt.")
finally:
    print("Exiting program.")
