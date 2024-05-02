import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from skimage.transform import warp_polar
from skimage.util import img_as_float
from skimage import io

image = io.imread('https://upload.wikimedia.org/wikipedia/commons/7/70/Caravaggio_-_David_with_the_Head_of_Goliath_-_Vienna.jpg')

# double-invert the image as it falls on the retina
image = img_as_float(image)[::-1,::-1]
image_polar = warp_polar(image, radius=1100, scaling='log', multichannel=True)

fig, axes = plt.subplots(1, 1, figsize=(12, 8))
axes.set_title("Original")
axes.imshow(image[::-1,::-1])
axes.set_axis_off()

fig, axes = plt.subplots(1, 2, figsize=(24, 8), dpi=300)
ax = axes.ravel()
ax[0].set_title("Left-hemisphere V1 image")
ax[0].imshow(np.roll(image_polar, 0, 0)[image_polar.shape[0]//4:3*image_polar.shape[0]//4][::-1,::-1])
ax[0].set_axis_off()

ax[1].set_title("Right-hemisphere V1 image")
ax[1].imshow(np.roll(image_polar, image_polar.shape[0]//2, 0)[image_polar.shape[0]//4:3*image_polar.shape[0]//4])
ax[1].set_axis_off()
# ax[3].set_title("Polar-Transformed Rotated")
# ax[3].imshow(rotated_polar)
plt.axis('off')
plt.savefig('Caravaggio_logpolar.png')