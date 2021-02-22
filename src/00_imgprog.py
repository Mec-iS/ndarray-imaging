"""
Create subplots
"""
import matplotlib
matplotlib.use('Qt5Agg')

from os.path import dirname, realpath, join
import numpy as np
import matplotlib.pyplot as plt

fpath = join(dirname(dirname(realpath(__file__))), "data", "aerials", "2.1.01.tiff")

img1 = plt.imread(fpath)
print(type(img1))
print(img1.shape)
print(img1.ndim)
print(img1.size)
print(img1.dtype)
print(img1.nbytes)

# image statistics
print(np.median(img1))
print(np.average(img1))
print(np.mean(img1))
print(np.std(img1))
print(np.var(img1))


r = img1[:, :, 0]
g = img1[:, :, 1]
b = img1[:, :, 2]

output = [img1, r, g, b]

titles = ["image", "red", "green", "blue"]

for i in range(4):
    plt.subplot(2, 2, i+1)

    plt.axis("off")
    plt.title(titles[i])
    if i == 0:
        plt.imshow(output[i])
    else:
        plt.imshow(output[i], cmap="gray")


plt.show()