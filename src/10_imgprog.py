"""
IMAGE SEGMENTATION: Intensity normalization
"""
from os.path import join
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from __init__ import assetsdir,rgb2gray

def normalize(img):
    # check image is grayscale
    print(img.shape, type(img))
    if len(img.shape) == 3:
        raise ValueError("image is not monochannel grayscale")

    lmin = float(img.min())
    lmax = float(img.max())
    return np.floor((img-lmin) / (lmax-lmin) * 255.0)


img1 = mpimg.imread(join(assetsdir, "2.2.05.tiff"))
print(img1.shape, type(img1))

_, axs = plt.subplots(1, 2, constrained_layout=True)

axs[0].set_title("Original")
axs[0].imshow(img1)

axs[1].set_title("normalize intensity")
axs[1].imshow(normalize(
    rgb2gray(img1)  # needs to be a grayscale image
    ), cmap="gray")

plt.axis("off")
plt.show()