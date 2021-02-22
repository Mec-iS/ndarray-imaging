"""
Subplots with bitwise operations between images
"""
import matplotlib
matplotlib.use('Qt5Agg')

from os.path import dirname, realpath, join

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img1 = mpimg.imread(join(dirname(dirname(realpath(__file__))), "data", "aerials", "2.1.01.tiff"))
img2 = mpimg.imread(join(dirname(dirname(realpath(__file__))), "data", "aerials", "2.1.03.tiff"))

_, axs = plt.subplots(3, 2, constrained_layout=True)
plt.axis("off")

print(axs)

axs[0, 0].imshow(img1)
axs[0, 0].set_title('image 1')

axs[0, 1].imshow(img2)
axs[0, 1].set_title("image2")

#
# bitwise ops
#
axs[1, 0].set_title("AND")
axs[1, 0].imshow(np.bitwise_and(img1, img2))

axs[1, 1].set_title("OR")
axs[1, 1].imshow(np.bitwise_or(img1, img2))

axs[2, 0].set_title("XOR")
axs[2, 0].imshow(np.bitwise_xor(img1, img2))

axs[2, 1].set_title("NOT Image 1")
axs[2, 1].imshow(np.bitwise_not(img1))

plt.show()