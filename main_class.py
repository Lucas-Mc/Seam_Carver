# Lucas McCullum
# Create a Seam Carving Algorithm

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class MainClass():

  def __init__(self):
    pass

  def calc_grad_corner(self, val_c, val_x, val_y):
    """
    val_c
    val_x
    val_y
    """
    grad_x = val_c - val_x
    grad_y = val_c - val_y
    return grad_x, grad_y

  def calc_grad_edge_x(self, val_c, val_x1, val_x2, val_y):
    """
    val_c
    val_x1
    val_x2
    val_y
    """
    grad_x = val_x2 - 2*val_c + val_x1
    grad_y = val_c - val_y
    return grad_x, grad_y

  def calc_grad_edge_y(self, val_c, val_x, val_y1, val_y2):
    """
    val_c
    val_x
    val_y1
    val_y2
    """
    grad_x = val_c - val_x
    grad_y = val_y2 - 2*val_c + val_y1
    return grad_x, grad_y

  def calc_grad_center(self, val_c, val_x1, val_x2, val_y1, val_y2):
    """
    val_c
    val_x1
    val_x2
    val_y1
    val_y2
    """
    grad_x = val_x2 - 2*val_c + val_x1
    grad_y = val_y2 - 2*val_c + val_y1
    return grad_x, grad_y

  def import_RGB_vals(self, file_name):
    # Import the image to be analyzed
    img_file = Image.open(file_name)
    img = img_file.load()

    # Get the image width and height in pixels
    [xs, ys] = img_file.size

    return img, xs, ys

  def seperate_RGB(self, img, xs, ys):
    # Initialize an empty array to store the RGB values
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

    return r_vals, g_vals, b_vals

  def calc_img_grads(self, p_vals, xs, ys):
    # Initialize an empty array to store the RGB gradient values
    p_grads_x = [[0] * ys for i in range(xs)]
    p_grads_y = [[0] * ys for i in range(xs)]

    # Determine the image gradients for each RGB value
    for x in range(xs):
      for y in range(ys):
        # Refactor this shit
        if ((x == 0) and (y == 0)):
          p_grads_x[x][y], p_grads_y[x][y] = self.calc_grad_corner(p_vals[x][y], p_vals[x+1][y], p_vals[x][y+1])

        elif ((x == 0) and (y == ys-1)):
          p_grads_x[x][y], p_grads_y[x][y] = self.calc_grad_corner(p_vals[x][y], p_vals[x+1][y], p_vals[x][y-1])

        elif ((x == xs-1) and (y == 0)):
          p_grads_x[x][y], p_grads_y[x][y] = self.calc_grad_corner(p_vals[x][y], p_vals[x-1][y], p_vals[x][y+1])

        elif ((x == xs-1) and (y == ys-1)):
          p_grads_x[x][y], p_grads_y[x][y] = self.calc_grad_corner(p_vals[x][y], p_vals[x-1][y], p_vals[x][y-1])

        elif ((x == 0) and (y != 0) and (y != ys-1)):
          p_grads_x[x][y], p_grads_y[x][y] = self.calc_grad_edge_y(p_vals[x][y], p_vals[x+1][y], p_vals[x][y-1], p_vals[x][y+1])
        
        elif ((x == xs-1) and (y != 0) and (y != ys-1)):
          p_grads_x[x][y], p_grads_y[x][y] = self.calc_grad_edge_y(p_vals[x][y], p_vals[x-1][y], p_vals[x][y-1], p_vals[x][y+1])

        elif ((y == 0) and (x != 0) and (x != xs-1)):
          p_grads_x[x][y], p_grads_y[x][y] = self.calc_grad_edge_x(p_vals[x][y], p_vals[x-1][y], p_vals[x+1][y], p_vals[x][y+1])
        
        elif ((y == ys-1) and (x != 0) and (x != xs-1)):
          p_grads_x[x][y], p_grads_y[x][y] = self.calc_grad_edge_x(p_vals[x][y], p_vals[x-1][y], p_vals[x+1][y], p_vals[x][y-1])
        
        else:
          p_grads_x[x][y], p_grads_y[x][y] = self.calc_grad_center(p_vals[x][y], p_vals[x-1][y], p_vals[x+1][y], p_vals[x][y-1], p_vals[x][y+1])

    return p_grads_x, p_grads_y

  def calc_grad_magnitude(self, r_vals, g_vals, b_vals, xs, ys):
    # Normalize all of these values to remove potential sign errors
    # Initialize an empty array to store the RGB gradient values
    norm_grads = [[0] * ys for i in range(xs)]

    # Determine the image gradients for each RGB value
    for x in range(xs):
      for y in range(ys):
        norm_grads[x][y] = r_vals[x][y]**2 + g_vals[x][y]**2 + b_vals[x][y]**2

    return norm_grads

  def calc_energy(self, norm_grads_x, norm_grads_y, xs, ys):
    # Determine the energy of the normalized gradients
    # Initialize an empty array to store the RGB gradient values
    energy_mat = [[0] * ys for i in range(xs)]

    # Determine the energy from the image gradients for each RGB value
    for x in range(xs):
      for y in range(ys):
        energy_mat[x][y] = norm_grads_x[x][y] + norm_grads_y[x][y]

    return energy_mat
