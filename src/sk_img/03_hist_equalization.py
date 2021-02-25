"""
SK-Image: Histogram equalization.
adjusting the intensities in histogram such that we 
 get the contrast of an image enhanced.
"""
import matplotlib.pyplot as plt
import skimage.data as data
from skimage import exposure
from skimage.color import rgb2gray

_, axs = plt.subplots(3, 2, constrained_layout=True)

# original image and histogram
img = data.moon()

axs[0,0].set_title("original")
axs[0,0].imshow(img, cmap="gray")

axs[0,1].set_title("bimodal histogram")
axs[0,1].hist(img.ravel(), bins=256, range=[0,255])
plt.xlim([0,255])

# equalized
img_eq = exposure.equalize_hist(img)

axs[1,0].set_title("equalized")
axs[1,0].imshow(img_eq, cmap="gray")

axs[1,1].set_title("bimodal histogram")
axs[1,1].hist(img_eq.ravel(), bins=256, range=[0,255])
plt.xlim([0,255])

# equalize adapt histogram
img_adpt = exposure.equalize_adapthist(img, clip_limit=0.03)

axs[2,0].set_title("equalized adapt")
axs[2,0].imshow(img_adpt, cmap="gray")

axs[2,1].set_title("bimodal histogram")
axs[2,1].hist(img_adpt.ravel(), bins=256, range=[0,255])

plt.show()