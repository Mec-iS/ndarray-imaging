"""
SK-Image: Otsu Binarization
(automatic thresholding. foreground/background)
"""
import matplotlib.pyplot as plt
import skimage.data as data
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, threshold_local

# img = data.binary_blobs(length=512, blob_size_fraction=0.1, seed=5)
img = data.camera()
thresh = threshold_otsu(img)
binary = img > thresh

_, axs = plt.subplots(2, 3, constrained_layout=True)

axs[0,0].set_title("total threshold")
axs[0,0].imshow(img)

axs[0,1].set_title("bimodal histogram")
axs[0,1].hist(img.ravel(), bins=256, range=[0,255])
plt.xlim([0,255])

axs[0,2].set_title("otsu")
axs[0,2].imshow(binary, cmap="gray")

img = data.page()
adaptive = threshold_local(img, block_size=3, offset=30)
print(img.shape, adaptive.shape)

axs[1,0].set_title("local threshold")
axs[1,0].imshow(img, cmap="gray")

axs[1,1].set_title("bimodal histogram")
axs[1,1].hist(adaptive.ravel(), bins=256, range=[0,255])
plt.xlim([0,255])

axs[1,2].set_title("local threshold")
axs[1,2].imshow(adaptive, cmap="gray")

plt.axis("off")
plt.show()