"""
From multi-channel to grayscale
"""
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from __init__ import assetsdir

def rgb2gray(img):
    r = img1[:, :, 0]
    g = img1[:, :, 1]
    b = img1[:, :, 2]
    return (
        0.2989 * r + 0.5879 * g + 0.114 * b
    )

img1 = mpimg.imread(join(assetsdir, "2.2.05.tiff"))
print(img1.shape)
print(rgb2gray(img1).shape)

plt.subplot(1, 2, 1)
plt.imshow(img1)
plt.subplot(1, 2, 2)
plt.imshow(rgb2gray(img1), cmap="gray")

plt.show()