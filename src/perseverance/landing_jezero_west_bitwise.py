"""
Subplots with bitwise operations between images
"""
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from __init__ import assetsdir, rgb2gray, threshold, inverted_threshold
from skimage import exposure
from skimage.filters import threshold_otsu, threshold_local, threshold_mean

img1 = rgb2gray(mpimg.imread(join(assetsdir, "Mars_Perseverance_descent_1.png")))
img_eq = exposure.equalize_hist(mpimg.imread(join(assetsdir, "Mars_Perseverance_descent_1.png")))
img_otsu = img1 > threshold_otsu(img1)

print(img1.shape, img_eq.shape, img_otsu.shape)

_, axs = plt.subplots(2, 3, constrained_layout=True)
plt.axis("off")

print(axs)

axs[0, 0].imshow(img1)
axs[0, 0].set_title('image')

axs[0, 1].imshow(img_eq)
axs[0, 1].set_title("histogram equalize")

axs[0, 2].set_title("otsu")
axs[0, 2].imshow(img_otsu)

#
# bitwise ops
#
axs[1, 0].set_title("NOT otsu")
axs[1, 0].imshow(np.bitwise_not(img_otsu))

axs[1, 1].set_title("NOT tresh mean")
axs[1, 1].imshow(np.bitwise_not(img1 > threshold_mean(img1)))

axs[1, 2].set_title("otsu XOR thresh mean")
axs[1, 2].imshow(np.bitwise_xor(img_otsu, img1 > threshold_mean(img1)))


plt.show()