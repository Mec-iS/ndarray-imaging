"""
IMAGE SEGMENTATION: shading
"""
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from __init__ import assetsdir

def shading(img, percent=0.5):
    return (img * (1 - percent)).astype("uint8")

img1 = mpimg.imread(join(assetsdir, "2.1.01.tiff"))

_, axs = plt.subplots(2, 2, constrained_layout=True)

axs[0,0].set_title("Original")
axs[0,0].imshow(img1)

axs[0,1].set_title("shading 50")
axs[0,1].imshow(shading(img1))

axs[1,0].set_title("shading 25")
axs[1,0].imshow(shading(img1, percent=0.25))

axs[1,1].set_title("shading 75")
axs[1,1].imshow(shading(img1, percent=0.75))

plt.show()