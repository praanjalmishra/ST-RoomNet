import os
import csv

# Mock CSV file content
CSV_CONTENT = [
    ["received_data\\image_1415250.jpg", "Width: 640, Height: 480, FOV: 74.32295, FocalLength: (506.91, 506.72), PrincipalPoint: (320.42, 243.13)&&Position: (3.27, 0.46, -0.73), Rotation: (0.00400, -0.68306, -0.00134, 0.73035)"],
    ["received_data\\image_1333171.jpg", "Width: 1280, Height: 720, FOV: 90.0, FocalLength: (800.0, 800.0), PrincipalPoint: (640.0, 360.0)&&Position: (0.0, 0.0, 0.0), Rotation: (0.0, 0.0, 0.0, 1.0)"],
]

# Function to parse metadata row
def parse_metadata_row(row):
    """
    Parses a single metadata row to extract camera intrinsics and extrinsics.
    :param row: CSV row containing metadata.
    :return: Dictionary with image path, intrinsics, and extrinsics.
    """
    image_path, params = row
    intrinsics, extrinsics = params.split("&&")

    # Parse intrinsics
    width, height, fov, focal_length, principal_point = intrinsics.split(", ")
    intrinsics_dict = {
        "fx": float(focal_length.split(": ")[1].strip("()").split(",")[0]),
        "fy": float(focal_length.split(": ")[1].strip("()").split(",")[1]),
        "cx": float(principal_point.split(": ")[1].strip("()").split(",")[0]),
        "cy": float(principal_point.split(": ")[1].strip("()").split(",")[1]),
    }

    # Parse extrinsics
    position, rotation = extrinsics.split(", Rotation: ")
    extrinsics_dict = {
        "position": list(map(float, position.split(": ")[1].strip("()").split(", "))),
        "rotation": list(map(float, rotation.strip("()").split(", "))),
    }

    return {
        "image_path": image_path,
        "intrinsics": intrinsics_dict,
        "extrinsics": extrinsics_dict,
    }


# Simulate parsing CSV rows
def test_parsing():
    for row in CSV_CONTENT:
        print("Testing row:", row[0])
        metadata = parse_metadata_row(row)
        print("Parsed metadata:")
        print(metadata)
        print()

# Test path comparison
def test_path_comparison():
    RECEIVED_IMAGES_DIR = "/home/scai/mr/server_ml/server_pro/received_data/images/"
    event_src_path = os.path.join(RECEIVED_IMAGES_DIR, "image_1333171.jpg")
    print(f"Event Source Path: {event_src_path}")

    for row in CSV_CONTENT:
        metadata_path = os.path.normpath(os.path.join(RECEIVED_IMAGES_DIR, os.path.basename(row[0])))
        print(f"Comparing Metadata Path: {metadata_path} with Event Path: {event_src_path}")
        print(f"Match: {metadata_path == event_src_path}")
        print()

if __name__ == "__main__":
    print("=== Testing Metadata Parsing ===")
    test_parsing()
    print("\n=== Testing Path Comparison ===")
    test_path_comparison()
