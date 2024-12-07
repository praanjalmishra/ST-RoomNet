import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# File name containing the JSON data
json_file = 'coordinates.json'


# Read JSON data from the file
with open(json_file, 'r') as file:
    data = json.load(file)

# Create a single 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot all line segments
for item in data:
    image_path = item["image_path"]
    segments = item["3d_line_segments"]

    for segment in segments:
        start = segment["start"]
        end = segment["end"]

        # Extract x, y, z coordinates
        x = [start[0], end[0]]
        y = [start[1], end[1]]
        z = [start[2], end[2]]

        # Plot the line segment
        ax.plot(x, y, z, marker='o', label=f"{image_path.split('/')[-1]}" if len(data) == 1 else "")

# Label the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set the title
ax.set_title("3D Line Segments from All Images")

# Show the plot
plt.show()