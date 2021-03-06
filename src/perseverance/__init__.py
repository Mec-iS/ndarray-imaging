import matplotlib
matplotlib.use('Qt5Agg')

from os.path import dirname, realpath, join

assetsdir = join(dirname(dirname(dirname(realpath(__file__)))), "data", "aerials")

def rgb2gray(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    return (
        0.2989 * r + 0.5879 * g + 0.114 * b
    )

def threshold(img, thresh=127):
    return ((img > thresh) * 255).astype("uint8")

def inverted_threshold(img, thresh=127):
    return ((img < thresh) * 255).astype("uint8")