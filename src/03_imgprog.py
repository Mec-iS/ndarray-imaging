"""
Subplots with image histograms
"""
import matplotlib
matplotlib.use('Qt5Agg')

from os.path import dirname, realpath, join

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img1 = mpimg.imread(join(dirname(dirname(realpath(__file__))), "data", "aerials", "2.2.05.tiff"))
r = img1[:, :, 0]
g = img1[:, :, 1]
b = img1[:, :, 2]

_, axs = plt.subplots(3, 2, constrained_layout=False)
plt.axis("off")
plt.subplots_adjust(hspace=0.5, wspace=0.5)

axs[0, 0].imshow(img1)
axs[0, 0].set_title('image 1')

#
# histograms for channels
#
hist, bins = np.histogram(
    r.ravel(),  # channel is a 2D numpy array, flattened
    bins=256, range=(0, 256))
axs[0, 1].set_title("RED histogram")
axs[0, 1].bar(bins[:-1], hist)

hist, bins = np.histogram(
    g.ravel(),
    bins=256, range=(0, 256))
axs[1, 1].set_title("GREEN histogram")
axs[1, 1].bar(bins[:-1], hist)

hist, bins = np.histogram(
    b.ravel(),
    bins=256, range=(0, 256))
axs[2, 1].set_title("BLUE histogram")
axs[2, 1].bar(bins[:-1], hist)

plt.show()