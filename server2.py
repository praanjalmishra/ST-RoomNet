import asyncio
import websockets
import cv2
import numpy as np
import threading


exit_flag = False


def keyboard_listener():
    global exit_flag
    print("Press 'Ctrl + E' to exit the server.")
    while not exit_flag:
        key = input()  
        if key.lower() == '\x05':  # Ctrl + E to exit
            exit_flag = True
            print("Exiting server...")

# WebSocket
async def handle_connection(websocket, path):
    global exit_flag
    print("Client connected")
    try:
        while not exit_flag:
            message = await websocket.recv()

            if message:
                metadata_end_index = message.find(b'||')
                if metadata_end_index == -1:
                    print("Invalid message format, missing metadata delimiter")
                    continue

                metadata = message[:metadata_end_index].decode("utf-8")
                image_data = message[metadata_end_index + 2:]

                print(f"Metadata: {metadata}")
                print(f"Image data size: {len(image_data)} bytes")

                # Save raw image data for debugging
                with open("debug_image.jpg", "wb") as f:
                    f.write(image_data)

                # Convert byte stream to OpenCV image
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is not None:
                    cv2.imshow("Received Image", image)
                    cv2.waitKey(1)
                else:
                    print("Failed to decode image. Check debug_image.jpg for details.")

                response = f"Server processed image and metadata: {metadata}"
                await websocket.send(response)
                print(f"Sent back to client: {response}")
    except websockets.ConnectionClosed:
        print("Client disconnected")
    finally:
        cv2.destroyAllWindows()


async def main():
    global exit_flag
    start_server = websockets.serve(handle_connection, "0.0.0.0", 8765)

    print("WebSocket server running on ws://localhost:8765")
    print("Press 'Ctrl + E' to exit the server.")

    # start WebSocket server
    server = await start_server

    
    while not exit_flag:
        await asyncio.sleep(1)

 
    server.close()
    await server.wait_closed()
    print("Server shutdown.")


keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
keyboard_thread.start()


try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("Server stopped by keyboard interrupt.")
finally:
    print("Exiting program.")
