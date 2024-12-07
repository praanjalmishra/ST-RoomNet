import asyncio
import websockets
import os
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Folder containing the JSON file
JSON_FOLDER = "/home/scai/mr/server_ml/server_pro/processed_data/"
os.makedirs(JSON_FOLDER, exist_ok=True)  # Ensure the folder exists

# Global variable to track WebSocket clients
connected_clients = set()

class JSONFileHandler(FileSystemEventHandler):
    """Handles changes in JSON files."""
    def __init__(self, loop):
        self.loop = loop

    def on_modified(self, event):
        if event.src_path.endswith(".json"):
            print(f"Detected change in JSON file: {event.src_path}")
            # Schedule sending updates to clients
            self.loop.call_soon_threadsafe(asyncio.create_task, send_json_update(event.src_path))

async def send_json_update(file_path):
    """Send updated JSON data to all connected clients."""
    try:
        # Load the updated JSON file
        with open(file_path, "r") as file:
            data = json.load(file)
        filename = os.path.basename(file_path)

        # Prepare the message
        message = json.dumps({"filename": filename, "content": data})

        # Send to all connected clients
        if connected_clients:
            print(f"Sending updated JSON to {len(connected_clients)} clients.")
            await asyncio.gather(*(client.send(message) for client in connected_clients))
        else:
            print("No connected clients to send the update.")
    except Exception as e:
        print(f"Failed to send JSON update: {str(e)}")

async def handle_connection(websocket, path):
    """Handle incoming WebSocket connections."""
    print("Client connected")
    connected_clients.add(websocket)
    try:
        # Keep the connection alive
        await websocket.wait_closed()
    finally:
        connected_clients.remove(websocket)
        print("Client disconnected")

async def main():
    """Main coroutine to start the WebSocket server and monitor JSON files."""
    # Start WebSocket server
    server = await websockets.serve(handle_connection, "0.0.0.0", 8765)
    print("WebSocket server running on ws://localhost:8765")

    # Start watchdog observer
    loop = asyncio.get_event_loop()
    event_handler = JSONFileHandler(loop)
    observer = Observer()
    observer.schedule(event_handler, JSON_FOLDER, recursive=False)
    observer.start()
    print(f"Watching for changes in folder: {JSON_FOLDER}")

    try:
        await server.wait_closed()
    finally:
        observer.stop()
        observer.join()
        print("Server and observer stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by keyboard interrupt.")
