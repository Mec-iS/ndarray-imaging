"""
IMAGE SEGMENTATION: image thresholding
on Mars Perseverance ground image
(Jezero Crater)
"""
from os.path import join, dirname

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from __init__ import assetsdir, rgb2gray, threshold, inverted_threshold


img1 = mpimg.imread(join(dirname(assetsdir), "ground", "Mars_Perseverance_NRE_0002_0667130601_395ECM_N0010052AUT_04096_00_0LLJ01.png"))
print(type(img1))
print(img1.shape)
print(img1.ndim)
print(img1.size)
print(img1.dtype)
print(img1.nbytes)
print(np.median(img1))
print(np.average(img1))
print(np.mean(img1))
print(np.std(img1))
print(np.var(img1))

_, axs = plt.subplots(2, 2, constrained_layout=True)

axs[0,0].set_title("Original")
axs[0,0].imshow(img1)

axs[0,1].set_title("grayscale")
axs[0,1].imshow(rgb2gray(img1), cmap="gray")

axs[1,0].set_title("inverted threshold 127")
axs[1,0].imshow(inverted_threshold(img1, thresh=200))

axs[1,1].set_title("inverted threshold 127 - original")
axs[1,1].imshow(inverted_threshold(rgb2gray(img1)) - rgb2gray(img1), cmap="gray")

plt.show()