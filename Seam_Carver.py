# Lucas McCullum
# Create a Seam Carving Algorithm

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from main_class import MainClass

from scipy import ndimage

mc = MainClass()

img, xs, ys = mc.import_RGB_vals('Penalty_Kick.jpg')

r_vals, g_vals, b_vals = mc.seperate_RGB(img, xs, ys)

r_grads_x, r_grads_y = mc.calc_img_grads(r_vals, xs, ys)
g_grads_x, g_grads_y = mc.calc_img_grads(g_vals, xs, ys)
b_grads_x, b_grads_y = mc.calc_img_grads(b_vals, xs, ys)

norm_grads_x = mc.calc_grad_magnitude(r_grads_x, g_grads_x, b_grads_x, xs, ys)
norm_grads_y = mc.calc_grad_magnitude(r_grads_y, g_grads_y, b_grads_y, xs, ys)

energy_mat = mc.calc_energy(norm_grads_x, norm_grads_y, xs, ys)

img_file = Image.open('Penalty_Kick.jpg')
img_file.show()

plt.figure(1)
rotated_img = ndimage.rotate(energy_mat, -90)
plt.imshow(rotated_img)
plt.show()
