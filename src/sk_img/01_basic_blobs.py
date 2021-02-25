"""
SK-Image: generate random blobs
"""
import matplotlib.pyplot as plt
import skimage.data as data
from skimage.color import convert_colorspace, rgb2gray

from random import randint, uniform

_, axs = plt.subplots(5, 5, constrained_layout=True)

for i in range(5):
    for j in range(5):
        fraction = uniform(0.1,1.0)
        axs[i,j].set_title(f"fr = {str(fraction)[1:4]}")
        axs[i,j].imshow(
            data.binary_blobs(length=128, blob_size_fraction=fraction, seed=5)
        )

plt.show()