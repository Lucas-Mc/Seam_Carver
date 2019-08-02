# Lucas McCullum
# Create a Seam Carving Algorithm

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import the image to be analyzed
img_file = Image.open('Penalty_Kick.jpg')
img = img_file.load()

# Get the image width and height in pixels
[xs, ys] = img_file.size
max_intensity = 100
hues = {}

# Initialize an empty array to store the rgb values
r_vals = [[0] * ys for i in range(xs)]
g_vals = [[0] * ys for i in range(xs)]
b_vals = [[0] * ys for i in range(xs)]

# Examine each pixel in the image file
for x in range(xs):
  for y in range(ys):
    # Get the RGB values of each pixel
    [r, g, b] = img[x, y]

    # Normalize the pixel color values
    r /= 255.0
    g /= 255.0
    b /= 255.0

    # Store these new normalized values into the initialized arrays
    r_vals[x][y] = r
    g_vals[x][y] = g
    b_vals[x][y] = b

print(r_vals)
