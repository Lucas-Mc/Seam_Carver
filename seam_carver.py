# Lucas McCullum
# Create a Seam Carving Algorithm

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from main_class import MainClass

from scipy import ndimage

mc = MainClass()
show_plot = False

img, xs, ys = mc.import_RGB_vals('Penalty_Kick.jpg')

r_vals, g_vals, b_vals = mc.seperate_RGB(img, xs, ys)

r_grads_x, r_grads_y = mc.calc_img_grads(r_vals, xs, ys)
g_grads_x, g_grads_y = mc.calc_img_grads(g_vals, xs, ys)
b_grads_x, b_grads_y = mc.calc_img_grads(b_vals, xs, ys)

norm_grads_x = mc.calc_grad_magnitude(r_grads_x, g_grads_x, b_grads_x, xs, ys)
norm_grads_y = mc.calc_grad_magnitude(r_grads_y, g_grads_y, b_grads_y, xs, ys)

pixel_energies = mc.calc_energy(norm_grads_x, norm_grads_y, xs, ys)
pixel_energies = [*zip(*pixel_energies)]

if (show_plot is True):
  img_file = Image.open('Penalty_Kick.jpg')
  img_file.show()

  plt.figure(1)
  rotated_img = ndimage.rotate(pixel_energies, -90)
  plt.imshow(rotated_img)
  plt.show()

previous_seam_energies_row = list(pixel_energies[0])
coord_list = []

# Skip the first row in the following loop.
for y in range(1, len(pixel_energies)):
    pixel_energies_row = pixel_energies[y]

    seam_energies_row = []
    temp_min = np.inf
    temp_ind = 0
    for x, pixel_energy in enumerate(pixel_energies_row):
        # Determine the range of x values to iterate over in the previous
        # row. The range depends on if the current pixel is in the middle of
        # the image, or on one of the edges.
        x_left = max(x - 1, 0)
        x_right = min(x + 1, len(pixel_energies_row) - 1)
        x_range = range(x_left, x_right + 1)

        for i,x_i in enumerate(x_range):
          temp_val = previous_seam_energies_row[x_i]
          if (temp_val < temp_min):
            temp_min = temp_val
            temp_ind = x_i

        min_seam_energy = pixel_energy + temp_min #min(previous_seam_energies_row[x_i] for x_i in x_range)
        seam_energies_row.append(min_seam_energy)
      
    coord_list.append(temp_ind)
    previous_seam_energies_row = seam_energies_row

lowest_seam = min(seam_energy for seam_energy in previous_seam_energies_row)

img_file = Image.open('Penalty_Kick.jpg')
img_file.show()

# rotated_img = ndimage.rotate(pixel_energies, -90)
plt.figure(1)
plt.imshow(pixel_energies)

plt.figure(2)
plt.scatter(coord_list,range(161))
plt.show()

