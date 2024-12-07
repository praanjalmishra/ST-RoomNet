from flask import Flask, request, jsonify
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.convnext import ConvNeXtTiny, preprocess_input
from ST_RoomNet.spatial_transformer import ProjectiveTransformer
from tensorflow.keras.models import Model
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
import torch
import base64
import os
import matplotlib

# Initialize Flask app
app = Flask(__name__)
print("Initializing Flask app and registering endpoints...")

# Load the RoomNet model and weights
base_model = ConvNeXtTiny(include_top=False, weights="imagenet", input_shape=(400, 400, 3), pooling='avg')
theta = tf.keras.layers.Dense(8)(base_model.output)
ref_img = tf.io.read_file('/home/scai/mr/server_ml/ST_RoomNet/ref_img2.png')
ref_img = tf.io.decode_png(ref_img)
ref_img = tf.cast(ref_img, tf.float32) / 51.0
ref_img = ref_img[tf.newaxis, ...]
stl = ProjectiveTransformer((400, 400)).transform(ref_img, theta)
roomnet_model = Model(base_model.input, stl)
roomnet_model.load_weights('/home/scai/mr/ST-RoomNet/Weight_ST_RroomNet_ConvNext.h5')

print("Initializing DepthAnythingV2 model...")
depth_anything = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
DEVICE = 'cpu'
try:
    print("Loading checkpoint...")
    state_dict = torch.load(
        '/home/scai/mr/server_ml/Depth_Anything_V2/checkpoints/depth_anything_v2_vitb.pth',
        map_location=DEVICE,
        weights_only=True
    )
    depth_anything.load_state_dict(state_dict)
    print("Checkpoint loaded successfully.")
except Exception as e:
    print(f"Failed to load checkpoint: {e}")
    raise

try:
    print("Setting model to evaluation mode...")
    depth_anything = depth_anything.to(DEVICE).eval()
    print("Model is ready.")
except Exception as e:
    print(f"Failed to set the model to evaluation mode: {e}")
    raise

# RoomNet Inference
def roomnet_inference(image_bytes):
    try:
        # Decode and preprocess the input image
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode the image. Ensure the input is a valid image file.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize input to 400x400 (model-compatible size)
        img_resized = cv2.resize(img, (400, 400))
        print(f"Resized image shape: {img_resized.shape}")
        img_preprocessed = preprocess_input(img_resized[tf.newaxis, ...])  # Preprocess for RoomNet

        # Perform inference
        out = roomnet_model.predict(img_preprocessed)
        out = np.rint(out[0, :, :, 0])  # Extract single-channel output

        # Resize the output back to 480x640 to match the depth map
        out_resized = cv2.resize(out, (640, 480), interpolation=cv2.INTER_NEAREST)
        print(f"Resized output shape of output.png: {out_resized.shape}")  # Should be (480, 640)

        # Save the resized output using OpenCV
        out_resized_uint8 = (out_resized * (255 / out_resized.max())).astype(np.uint8)  # Scale to 8-bit
        cv2.imwrite('output.png', out_resized_uint8)
        print("Saved output.png using OpenCV.")

        # Post-process the output for edge detection
        segmented_image = cv2.imread('output.png', cv2.IMREAD_GRAYSCALE)
        blurred_image = cv2.GaussianBlur(segmented_image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        enhanced_edges = cv2.dilate(edges, kernel, iterations=1)
        overlay = cv2.addWeighted(segmented_image, 0.8, enhanced_edges, 1, 0)

        return overlay
    except Exception as e:
        print(f"Error during RoomNet inference: {e}")
        raise



# def roomnet_inference(image_bytes):
#     try:
#         img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
#         if img is None:
#             raise ValueError("Failed to decode the image. Ensure the input is a valid image file.")
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (400, 400))
#         img = preprocess_input(img[tf.newaxis, ...])

#         out = roomnet_model.predict(img)
#         out = np.rint(out[0, :, :, 0])

#         plt.figure('seg')
#         plt.imshow(out, vmin=1, vmax=5)
#         plt.axis('off')
#         plt.savefig('output.png', bbox_inches='tight', pad_inches=0)
#         plt.close()

#         segmented_image = cv2.imread('output.png', cv2.IMREAD_GRAYSCALE)
#         blurred_image = cv2.GaussianBlur(segmented_image, (5, 5), 0)
#         edges = cv2.Canny(blurred_image, 50, 150)
#         kernel = np.ones((3, 3), np.uint8)
#         enhanced_edges = cv2.dilate(edges, kernel, iterations=1)
#         overlay = cv2.addWeighted(segmented_image, 0.8, enhanced_edges, 1, 0)

#         return overlay
#     except Exception as e:
#         print(f"Error during RoomNet inference: {e}")
#         raise

# Depth Estimation Inference
def depth_inference(image_bytes):
    try:
        # Decode the input image from bytes
        raw_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if raw_image is None:
            raise ValueError("Failed to decode the image. Ensure the input is a valid image file.")

        # Debugging: Log the image shape
        print(f"Decoded image shape: {raw_image.shape}") # (480, 640, 3)

        # Perform Depth Estimation inference
        depth = depth_anything.infer_image(raw_image, input_size=raw_image.shape[0])

        # Normalize the depth map to [0, 255]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        # Convert depth map to RGB for visualization if needed
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')  # Use Spectral_r for the colormap
        depth_colored = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)

        # Encode the depth map as PNG for returning
        _, buffer = cv2.imencode('.png', depth_colored)
        return buffer
    except Exception as e:
        print(f"Error during Depth Estimation inference: {e}")
        raise




# RoomNet Endpoint
@app.route('/process-roomnet', methods=['POST'])
def process_roomnet():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        file = request.files['image']
        img_bytes = file.read()

        result_overlay = roomnet_inference(img_bytes)
        _, buffer = cv2.imencode('.png', result_overlay)
        result_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"result": result_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Depth Estimation Endpoint
@app.route('/process-depth', methods=['POST'])
def process_depth():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        file = request.files['image']
        img_bytes = file.read()

        # Debugging: Check the size of the received data
        print(f"Received image size: {len(img_bytes)} bytes")

        result_depth = depth_inference(img_bytes)
        print(f"Depth result size: {len(result_depth)} bytes")  ## 190888 bytes
        result_base64 = base64.b64encode(result_depth).decode('utf-8')

        return jsonify({"result": result_base64})
    except Exception as e:
        print(f"Error in /process-depth: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
