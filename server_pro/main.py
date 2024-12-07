import os
import json
import time
import cv2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .inference_st import roomnet_process_image
from .inference_depth import load_depth_model, infer_depth
from .depth_processing import unproject_2d_to_3d, transform_to_world_frame

# Resolve paths relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECEIVED_DIR = os.path.join(SCRIPT_DIR, "received_data")
IMAGES_DIR = os.path.join(RECEIVED_DIR, "images")
METADATA_PATH = os.path.join(RECEIVED_DIR, "metadata.json")
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "processed_data")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DIR, "coordinates.json")

# Ensure processed_data directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load the Depth Model
depth_model = load_depth_model(encoder="vitb")  # Load the depth model globally


class DataHandler(FileSystemEventHandler):
    """
    Handles existing files and monitors new files in the 'received_data' directory.
    """

    def __init__(self, metadata_path, processed_data_path):
        self.metadata_path = os.path.abspath(metadata_path)
        self.processed_data_path = processed_data_path
        self.processed_results = []  # Store processed results

    def initialize_existing_files(self):
        """
        Processes all existing files in metadata.json.
        """
        print("Initializing existing files...")
        self.process_metadata()

    def on_modified(self, event):
        """
        Triggered when metadata.json is modified or new files are added.
        """
        if os.path.abspath(event.src_path) == self.metadata_path:
            print(f"Detected change in {self.metadata_path}. Processing...")
            self.process_metadata()

    def process_metadata(self):
        """
        Reads metadata.json, processes new images, and writes each result sequentially.
        """
        try:
            with open(self.metadata_path, "r") as file:
                metadata_entries = json.load(file)["images"]

            for entry in metadata_entries:
                image_path = os.path.join(IMAGES_DIR, os.path.basename(entry["path"]))
                if any(res["image_path"] == image_path for res in self.processed_results):
                    # Skip already processed entries
                    continue

                print(f"Processing {image_path}...")
                try:
                    # Process the image entry
                    result = self.process_image_entry(entry, image_path)
                    print(f"Result: {result}")
                    self.processed_results.append(result)

                    # Save the result immediately to the JSON file
                    self.append_result_to_json(result, self.processed_data_path)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
        except Exception as e:
            print(f"Error reading or processing metadata: {e}")


    def append_result_to_json(self, result, output_path):
        """
        Append a single result to the JSON file.
        :param result: Processed result to append.
        :param output_path: Path to the JSON file.
        """
        try:
            # Load existing data or initialize a new list
            if os.path.exists(output_path):
                with open(output_path, "r") as file:
                    data = json.load(file)
            else:
                data = []

            # Append the new result
            data.append(result)

            # Write updated data back to the file
            with open(output_path, "w") as file:
                json.dump(data, file, indent=4)
            print(f"Appended result to {output_path}.")
        except Exception as e:
            print(f"Error appending result to {output_path}: {e}")


    def process_image_entry(self, entry, image_path):
        """
        Process a single image entry: RoomNet + depth inference + unprojection.
        :param entry: Metadata entry for a single image.
        :param image_path: Path to the corresponding image file.
        :return: Processed result dictionary with image path and 3D line segments.
        """
        intrinsics = entry["intrinsics"]
        extrinsics = entry["extrinsics"]

        # Perform RoomNet inference to get line segments
        edge_segments = roomnet_process_image(image_path)
        if not edge_segments:
            print(f"No lines detected in {image_path}. Skipping.")
            return {
                "image_path": image_path,
                "3d_line_segments": [],
            }

        print(f"RoomNet Line Segments: {edge_segments}")

        # Format intrinsics and extrinsics
        intrinsics_formatted = {
            "fx": intrinsics["focalLength"]["fx"],
            "fy": intrinsics["focalLength"]["fy"],
            "cx": intrinsics["principalPoint"]["cx"],
            "cy": intrinsics["principalPoint"]["cy"],
        }
        extrinsics_formatted = {
            "rotation": [
                extrinsics["rotation"]["x"],
                extrinsics["rotation"]["y"],
                extrinsics["rotation"]["z"],
                extrinsics["rotation"]["w"],
            ],
            "translation": [
                extrinsics["position"]["x"],
                extrinsics["position"]["y"],
                extrinsics["position"]["z"],
            ],
        }

        # Perform depth inference
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Infer the depth map
        depth_map = infer_depth(raw_image, depth_model, input_size=518, grayscale=True)

        # Convert to single-channel if necessary
        if len(depth_map.shape) == 3 and depth_map.shape[2] == 3:
            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
            print("Converted depth map to single-channel.")

        # Normalize the depth map
        depth_map = depth_map.astype(float) / 255.0  # Normalize to range [0, 1]

        print(f"Depth Map Shape (After Processing): {depth_map.shape}")


        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 10))
        # plt.title("Depth Map")
        # plt.imshow(depth_map, cmap="viridis")
        # plt.colorbar(label="Depth (normalized)")
        # plt.axis("off")
        # plt.show()

        print(f"Depth Map Shape: {depth_map.shape}")

        # Initialize list to store 3D line segments
        line_segments_3d = []

        for segment in edge_segments:
            x1, y1 = segment["start"]
            x2, y2 = segment["end"]

            print(f"Processing line segment: Start=({x1}, {y1}), End=({x2}, {y2})")

            try:
                # Unproject endpoints of the line segment to 3D (camera frame)
                point1_camera = unproject_2d_to_3d([(x1, y1)], depth_map, intrinsics_formatted)
                #print(f"Point1 (Camera Frame): {point1_camera}, Type: {type(point1_camera)}")

                point2_camera = unproject_2d_to_3d([(x2, y2)], depth_map, intrinsics_formatted)
                #print(f"Point2 (Camera Frame): {point2_camera}, Type: {type(point2_camera)}")

                # Validate unprojected points
                if not point1_camera or not point2_camera:
                    print(f"Skipping invalid line segment due to invalid unprojection: {segment}")
                    continue

                point1_camera = point1_camera[0]
                point2_camera = point2_camera[0]
                #print(f"Point1 (Camera Frame Extracted): {point1_camera}")
                #print(f"Point2 (Camera Frame Extracted): {point2_camera}")

                # Transform endpoints to world frame
                point1_world = transform_to_world_frame(
                    [point1_camera],
                    rotation=extrinsics_formatted["rotation"],
                    translation=extrinsics_formatted["translation"],
                )
                #print(f"Point1 (World Frame): {point1_world}, Type: {type(point1_world)}")

                point2_world = transform_to_world_frame(
                    [point2_camera],
                    rotation=extrinsics_formatted["rotation"],
                    translation=extrinsics_formatted["translation"],
                )
                #print(f"Point2 (World Frame): {point2_world}, Type: {type(point2_world)}")

                # Validate transformed points
                if not isinstance(point1_world, list) or not isinstance(point2_world, list):
                    print(f"Skipping invalid transformation for line segment: {segment}")
                    continue

                point1_world = point1_world[0]
                point2_world = point2_world[0]

                #print(f"Final Point1 (World Frame): {point1_world}")
                #print(f"Final Point2 (World Frame): {point2_world}")

                # Store the 3D line segment
                line_segments_3d.append({"start": point1_world, "end": point2_world})

            except Exception as e:
                print(f"Error processing line segment: {segment}, Error: {e}")
                continue

        return {
            "image_path": image_path,
            "3d_line_segments": line_segments_3d,
        }


    # def save_results(self, results, output_path):
    #     """
    #     Save processed results to a JSON file.
    #     :param results: List of processed results.
    #     :param output_path: Path to the output JSON file.
    #     """
    #      # Debugging print statements
    #     print(f"Debug: Saving results to {output_path}")
    #     print(f"Debug: Results data: {results}")

    #     with open(output_path, "w") as file:
    #         json.dump(results, file, indent=4)
    #         print("Debug: JSON file successfully written.")


def main():
    """
    Main function to start the watchdog observer.
    """
    print("Starting observer...")

    # Clear and initialize the processed JSON file
    with open(PROCESSED_DATA_PATH, "w") as file:
        json.dump([], file, indent=4)
    print(f"Initialized {PROCESSED_DATA_PATH} for new run.")
    event_handler = DataHandler(METADATA_PATH, PROCESSED_DATA_PATH)

    # Process existing files on startup
    event_handler.initialize_existing_files()

    # Start monitoring for new files
    observer = Observer()
    observer.schedule(event_handler, RECEIVED_DIR, recursive=False)
    observer.start()

    try:
        print(f"Monitoring {RECEIVED_DIR} for changes...")
        while True:
            time.sleep(1)  # Keep the observer running
    except KeyboardInterrupt:
        print("Stopping observer...")
        observer.stop()

    observer.join()


if __name__ == "__main__":
    main()
