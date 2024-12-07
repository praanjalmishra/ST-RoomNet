import csv

def parse_metadata_csv(metadata_file):
    """
    Parses the metadata CSV file to extract image paths and camera parameters.
    :param metadata_file: Path to the metadata CSV file.
    :return: List of dictionaries with metadata.
    """
    metadata_list = []
    with open(metadata_file, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            image_path, metadata = row
            intrinsic_str, extrinsic_str = metadata.split("&&")
            
            # Parse intrinsics
            intrinsics = {}
            for item in intrinsic_str.split(", "):
                key, value = item.split(": ")
                key = key.strip()
                if key in ["Width", "Height"]:
                    intrinsics[key.lower()] = int(value)
                elif key == "FOV":
                    intrinsics["fov"] = float(value)
                elif key == "FocalLength":
                    intrinsics["focal_length"] = tuple(map(float, value.strip("()").split(", ")))
                elif key == "PrincipalPoint":
                    intrinsics["principal_point"] = tuple(map(float, value.strip("()").split(", ")))
            
            # Parse extrinsics
            extrinsics = {}
            for item in extrinsic_str.split(", "):
                key, value = item.split(": ")
                key = key.strip()
                if key == "Position":
                    extrinsics["position"] = tuple(map(float, value.strip("()").split(", ")))
                elif key == "Rotation":
                    extrinsics["rotation"] = tuple(map(float, value.strip("()").split(", ")))

            metadata_list.append({
                "image_path": image_path,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics
            })

    return metadata_list
