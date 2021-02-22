"""
IMAGE SEGMENTATION: gradient
"""
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from __init__ import assetsdir

def gradient(img, reverse=False):
    cols = img.shape[1]
    if reverse is True:
        C = np.linspace(1, 0, cols)
    else:
        C = np.linspace(0, 1, cols)
    C = np.dstack((C, C, C))
    return (C * img).astype("uint8")

img1 = mpimg.imread(join(assetsdir, "2.1.01.tiff"))

_, axs = plt.subplots(2, 2, constrained_layout=True)

axs[0,0].set_title("Original")
axs[0,0].imshow(img1)

axs[0,1].set_title("gradient")
axs[0,1].imshow(gradient(img1))

axs[1,0].set_title("reverse gradient")
axs[1,0].imshow(gradient(img1, reverse=True))

plt.axis("off")
plt.show()