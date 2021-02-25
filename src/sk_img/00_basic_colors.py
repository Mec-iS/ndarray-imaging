"""
SK-Image: convert colorspace
"""
import skimage.io as io
import skimage.data as data
from skimage.color import convert_colorspace, rgb2gray
import matplotlib.pyplot as plt

# img = data.binary_blobs(length=512, blob_size_fraction=0.1, seed=5)
img = data.coffee()

_, axs = plt.subplots(2, 2, constrained_layout=True)

axs[0,0].set_title("Original RGB")
axs[0,0].imshow(img)

axs[0,1].set_title("grayscale")
axs[0,1].imshow(rgb2gray(img), cmap="gray")

axs[1,0].set_title("HSV")
axs[1,0].imshow(convert_colorspace(img, "RGB", "HSV"))


plt.show()