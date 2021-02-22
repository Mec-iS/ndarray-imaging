"""
Subplots with image trasformations
"""
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

from __init__ import assetsdir

img1 = plt.imread(join(assetsdir, "2.1.01.tiff"))
img2 = plt.imread(join(assetsdir, "2.1.03.tiff"))

plt.subplot(2, 2, 1)
plt.imshow(img1)

plt.subplot(2, 2, 2)
plt.imshow(img2)

plt.subplot(2, 2, 3)
#
# images subtraction, addition, flip, roll
#
# plt.imshow(img1 + img2)
# plt.imshow(img1 - img2)
# plt.imshow(np.flip(img1, 1))
plt.imshow(np.roll(img1, 2048))

plt.show()