import matplotlib
matplotlib.use('Qt5Agg')

from os.path import dirname, realpath, join

assetsdir_1 = join(dirname(dirname(dirname(dirname(realpath(__file__))))), "data", "mycelium", "videos")
