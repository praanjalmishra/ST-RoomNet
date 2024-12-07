import asyncio
import websockets
import os
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Folder containing JSON files
JSON_FOLDER = "/home/scai/mr/server_ml/server_pro/processed_data"
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

async def send_existing_json_files(websocket):
    """Send all existing JSON files in the folder to a connected client."""
    try:
        # List all JSON files in the folder
        json_files = [f for f in os.listdir(JSON_FOLDER) if f.endswith(".json")]

        if not json_files:
            message = "No JSON files available."
            print(message)
            await websocket.send(json.dumps({"status": "error", "message": message}))
            return

        for json_file in json_files:
            # Construct the full path
            file_path = os.path.join(JSON_FOLDER, json_file)
            
            # Load the JSON file
            with open(file_path, "r") as file:
                data = json.load(file)

            # Send JSON data to the client
            message = json.dumps({"filename": json_file, "content": data})
            await websocket.send(message)
            print(f"Sent existing JSON file: {json_file}")
    except Exception as e:
        error_message = f"Failed to send existing JSON files: {str(e)}"
        print(error_message)
        await websocket.send(json.dumps({"status": "error", "message": error_message}))

async def handle_connection(websocket, path):
    """Handle incoming WebSocket connections."""
    print("Client connected")
    connected_clients.add(websocket)
    try:
        # Send existing JSON files to the client
        await send_existing_json_files(websocket)

        # Keep the connection alive and send real-time updates
        await websocket.wait_closed()
    finally:
        connected_clients.remove(websocket)
        print("Client disconnected")

async def main():
    """Main coroutine to start the WebSocket server and monitor JSON files."""
    # Start WebSocket server
    server = await websockets.serve(handle_connection, "0.0.0.0", 8765)
    print(f"WebSocket server running on ws://localhost:8765")
    print(f"Watching for changes in folder: {JSON_FOLDER}")

    # Start watchdog observer
    loop = asyncio.get_event_loop()
    event_handler = JSONFileHandler(loop)
    observer = Observer()
    observer.schedule(event_handler, JSON_FOLDER, recursive=False)
    observer.start()

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
