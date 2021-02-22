"""
IMAGE SEGMENTATION: image thresholding
the converting of an image into two parts: background and foreground
"""
import matplotlib
matplotlib.use('Qt5Agg')

from os.path import dirname, realpath, join

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def threshold(img, thresh=127):
    return ((img > thresh) * 255).astype("uint8")

def inverted_threshold(img, thresh=127):
    return ((img < thresh) * 255).astype("uint8")

img1 = mpimg.imread(join(dirname(dirname(realpath(__file__))), "data", "aerials", "2.1.01.tiff"))

_, axs = plt.subplots(2, 2, constrained_layout=True)

axs[0,0].set_title("Original")
axs[0,0].imshow(img1)

axs[0,1].set_title("threshold 127")
axs[0,1].imshow(threshold(img1), cmap="gray")

axs[1,0].set_title("inverted threshold 127")
axs[1,0].imshow(inverted_threshold(img1), cmap="gray")

axs[1,1].set_title("inverted threshold 127 - img1")
axs[1,1].imshow(inverted_threshold(img1) - img1, cmap="gray")

plt.show()