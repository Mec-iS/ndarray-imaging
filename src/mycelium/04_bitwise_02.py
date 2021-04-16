"""
Subplots with bitwise operations between images
"""
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from __init__ import assetsdir_2, rgb2gray, threshold, inverted_threshold
from skimage import exposure
from skimage.filters import threshold_otsu, threshold_local, threshold_mean

from os import walk

files = []
for _, _, filenames in walk(assetsdir_2):
    files.extend(filenames)
    break

print(len(files))
rows_n = len(files)
cols_n = 2
_, axs = plt.subplots(rows_n, cols_n, constrained_layout=True)
plt.axis("off")

print(axs)

i = 0
for x in axs:
    try:
        img1 = rgb2gray(mpimg.imread(join(assetsdir_2, files[i])))
        img_otsu = img1 > threshold_otsu(img1)

        # "otsu XOR thresh mean"
        x[0].set_title(files[i] + " original gray")
        x[1].set_title(files[i] + " otsu XOR thresh mean")
        x[0].imshow(img1)
        x[1].imshow(np.bitwise_xor(img_otsu, img1 > threshold_mean(img1)))
        
        i += 1
    except IndexError:
        break


plt.show()