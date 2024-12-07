import os
import cv2
import numpy as np
import torch
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
import matplotlib

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.backends.mps.is_available() else 'cpu'
CHECKPOINT_DIR = '/home/scai/mr/server_ml/Depth_Anything_V2/checkpoints'
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}
CMAP = matplotlib.colormaps.get_cmap('Spectral_r')


# Function to load the DepthAnythingV2 model
def load_depth_model(encoder='vitb'):
    """
    Load the DepthAnythingV2 model with the specified encoder.
    :param encoder: Encoder type ('vits', 'vitb', 'vitl', or 'vitg').
    :return: Loaded model in evaluation mode.
    """
    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"Invalid encoder: {encoder}. Choose from {list(MODEL_CONFIGS.keys())}.")
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'depth_anything_v2_{encoder}.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}.")
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    return model.to(DEVICE).eval()


# Function to process a single image and return the depth map
def infer_depth(raw_image, model, input_size=518, grayscale=False):
    """
    Perform depth estimation on a single image.
    :param raw_image: Input image as a numpy array.
    :param model: Preloaded depth model.
    :param input_size: Input size for the depth model.
    :param grayscale: Whether to output the depth map in grayscale.
    :return: Depth map as a numpy array.
    """
    if raw_image is None:
        raise ValueError("Invalid input image. Ensure the image is loaded correctly.")
    
    # Infer depth
    depth_map = model.infer_image(raw_image, input_size)

    # Normalize depth to 0–255
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
    depth_map = depth_map.astype(np.uint8)

    # Apply colormap or grayscale
    if grayscale:
        return np.repeat(depth_map[..., np.newaxis], 3, axis=-1)
    else:
        return (CMAP(depth_map)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)


# Function to handle multiple images
def process_images(image_paths, model, input_size=518, grayscale=False):
    """
    Process a list of images for depth estimation.
    :param image_paths: List of paths to the images.
    :param model: Preloaded depth model.
    :param input_size: Input size for the depth model.
    :param grayscale: Whether to output the depth map in grayscale.
    :return: List of depth maps (one per image).
    """
    depth_maps = []
    for idx, image_path in enumerate(image_paths):
        print(f"Processing {idx + 1}/{len(image_paths)}: {image_path}")

        # Load image
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            print(f"Warning: Could not read image {image_path}. Skipping...")
            continue

        # Infer depth
        depth_map = infer_depth(raw_image, model, input_size, grayscale)
        depth_maps.append(depth_map)

    return depth_maps





#from inference_depth import load_depth_model, infer_depth, process_images

# # Load the model
# model = load_depth_model(encoder='vitb')  # Choose encoder: 'vits', 'vitb', 'vitl', 'vitg'

# # Single image processing
# image_path = '/home/scai/mr/server_ml/Depth_Anything_V2/depth_ml/1.png'
# raw_image = cv2.imread(image_path)
# depth_map = infer_depth(raw_image, model, input_size=518, grayscale=False)

# # # Multiple image processing
# # image_paths = ['/path/to/image1.jpg', '/path/to/image2.jpg']
# # depth_maps = process_images(image_paths, model, input_size=518, grayscale=True)

# # Visualize or save depth maps as needed
# cv2.imshow("Depth Map", depth_map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()