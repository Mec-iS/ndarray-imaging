from os.path import dirname, realpath, join

assetsdir = join(dirname(dirname(dirname(realpath(__file__)))), "data", "aerials")

def rgb2gray(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    return (
        0.2989 * r + 0.5879 * g + 0.114 * b
    )