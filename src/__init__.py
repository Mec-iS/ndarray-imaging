import matplotlib
matplotlib.use('Qt5Agg')

from os.path import dirname, realpath, join

assetsdir = join(dirname(dirname(realpath(__file__))), "data", "aerials")