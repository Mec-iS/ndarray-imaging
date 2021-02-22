"""
IMAGE SEGMENTATION: tinting
"""
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from __init__ import assetsdir

def tinting(img, percent=0.5):
    return (img + (np.ones(img.shape) - img) * percent).astype("uint8")

img1 = mpimg.imread(join(assetsdir, "2.1.01.tiff"))

_, axs = plt.subplots(2, 2, constrained_layout=True)

axs[0,0].set_title("Original")
axs[0,0].imshow(img1)

axs[0,1].set_title("tinting 50")
axs[0,1].imshow(tinting(img1))

axs[1,0].set_title("tinting 25")
axs[1,0].imshow(tinting(img1, percent=0.25))

axs[1,1].set_title("tinting 75")
axs[1,1].imshow(tinting(img1, percent=0.75))

plt.show()