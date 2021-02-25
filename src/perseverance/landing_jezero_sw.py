"""
IMAGE SEGMENTATION: image thresholding
on Mars Perseverance landing aerial image
(south west edge of Jezero Crater)
"""
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from __init__ import assetsdir, rgb2gray, threshold, inverted_threshold


img1 = mpimg.imread(join(assetsdir, "Mars_Perseverance_descent_3.png"))
grayscale_img1 = rgb2gray(img1)
print(np.median(img1), np.median(grayscale_img1))
print(np.average(img1), np.average(grayscale_img1))
print(np.mean(img1), np.mean(grayscale_img1))
print(np.std(img1), np.std(grayscale_img1))
print(np.var(img1), np.var(grayscale_img1))

_, axs = plt.subplots(2, 2, constrained_layout=True)

axs[0,0].set_title("Original")
axs[0,0].imshow(img1)

axs[0,1].set_title("grayscale")
axs[0,1].imshow(grayscale_img1, cmap="gray")

axs[1,0].set_title("grayscale - inverted threshold 127 grayscale")
axs[1,0].imshow(grayscale_img1 - inverted_threshold(grayscale_img1), cmap="gray")

axs[1,1].set_title("inverted threshold 127 grayscale - grayscale")
axs[1,1].imshow(inverted_threshold(grayscale_img1) - grayscale_img1, cmap="gray")

plt.show()