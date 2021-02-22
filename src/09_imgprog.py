"""
IMAGE SEGMENTATION: RGB max
"""
from os.path import join
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from __init__ import assetsdir

def RGBmax(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    M = np.maximum(np.maximum(r, g), b)

    img.setflags(write=1)

    r[r<M] = 0
    g[g<M] = 0
    b[b<M] = 0

    return (np.dstack((r,g,b)))


img1 = mpimg.imread(join(assetsdir, "2.2.05.tiff"))
img2 = img1.copy()
print(img1.flags)
print(img2.flags)

_, axs = plt.subplots(1, 2, constrained_layout=True)

axs[0].set_title("Original")
axs[0].imshow(img1)

axs[1].set_title("RGB max filter")
axs[1].imshow(RGBmax(img2))

plt.axis("off")
plt.show()