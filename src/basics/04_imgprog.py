"""
From multi-channel to grayscale
"""
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from __init__ import assetsdir, rgb2gray

img1 = mpimg.imread(join(assetsdir, "Mars_Perseverance_descent_1.png"))
print(img1.shape)
print(rgb2gray(img1).shape)

plt.subplot(1, 2, 1)
plt.imshow(img1)
plt.subplot(1, 2, 2)
plt.imshow(rgb2gray(img1), cmap="gray")

plt.show()