import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def extract_dominant_colors(image_path, n_colors=5):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image for faster processing
    resized_image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
    reshaped_image = resized_image.reshape(-1, 3)  # Reshape to (num_pixels, 3)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=0)
    kmeans.fit(reshaped_image)
    colors = kmeans.cluster_centers_.astype(int)

    # Sort colors by frequency
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_colors = colors[np.argsort(-counts)]

    return sorted_colors

def visualize_colors(colors):
    # Create a color palette for visualization
    palette = np.zeros((50, 300, 3), dtype=int)
    step = 300 // len(colors)
    for i, color in enumerate(colors):
        palette[:, i * step:(i + 1) * step, :] = color

    plt.figure(figsize=(10, 2))
    plt.imshow(palette)
    plt.axis('off')
    plt.title("Dominant Colors")
    plt.show()

# Analyze the images
image_paths = ['ceiling.png', 'floor.png', 'r_wall.png', 'l_wall.png', 'f_wall.png']
for image_path in image_paths:
    print(f"Processing {image_path}...")
    dominant_colors = extract_dominant_colors(image_path, n_colors=5)
    print(f"Dominant colors for {image_path}:")
    print(dominant_colors)
    visualize_colors(dominant_colors)




######### 


# ceiling.png:
# [[255 255 255]]



# floor.png:
# [[204 204 204]]


# r_wall.png:
# [[153 153 153]]


# l_wall.png:
# [[102 102 102]]



# f_wall.png:
# [[51 51 51]]
