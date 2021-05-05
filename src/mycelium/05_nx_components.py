"""
connected-component labeling on mycelium networks

Inspired by IPython Cookbook, Second Edition, by Cyrille Rossant, Packt Publishing 2018
https://github.com/ipython-books/cookbook-2nd/blob/master/LICENSE
"""

from os.path import join
import itertools
from random import randrange

import numpy as np
import networkx as nx
import matplotlib.colors as col
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.filters import threshold_otsu, threshold_mean
from colour import Color

from __init__ import assetsdir_3, rgb2gray, threshold, inverted_threshold


img_original_1 = mpimg.imread(join(assetsdir_3, "horizon-3.jpg"))
img_original_2 = mpimg.imread(join(assetsdir_3, "glass_1__sample_5_6.jpg"))

GRADIENT_N = 13


def prepare_image(img_o):
    """
    Grayscale image and normalise matrix for N 
    """
    img_o = rgb2gray(img_o.copy()).astype(int)
    print(img_o.shape)
    return (GRADIENT_N * (img_o - np.min(img_o)) / np.ptp(img_o)).astype(int)


def show_image(img, ax=None, **kwargs):
    ax.imshow(img, origin='lower',
              interpolation='none',
              **kwargs)
    ax.set_axis_off()


def show_graph(g, ax=None, **kwargs):
    pos = {(i, j): (j, i) for (i, j) in g.nodes()}
    node_color = [img[i, j] for (i, j) in g.nodes()]
    nx.draw_networkx(
        g,
        ax=ax,
        pos=pos,
        node_color='#000',
        linewidths=0.01,
        width=0.01,
        edge_color='#000',
        with_labels=False,
        node_size=0.1,
        **kwargs)


def compute_subgraph_components(img_o):
    """
    Draw a graph on pixels with same grayscale and
     filter subgraph with lower values.
    
    Returns subgraph and most connected components
    """
    # fully connected graph
    g = nx.grid_2d_graph(img_o.shape[0], img_o.shape[1])
    # compute subgraph for lower grayscale
    sub_g = g.subgraph(zip(*np.nonzero(img_o == 0)))
    # filter most connected components 
    components = [np.array(list(comp))
              for comp in nx.connected_components(sub_g)
              if len(comp) >= img_o.shape[0] // 3]
    
    return components


def generate_subgraph_image(img_o):
    """
    Create image to visualise subgraphs with color gradient
    """
    img_bis = img_o.copy()

    components = compute_subgraph_components(img_o)

    for i, comp in enumerate(components):
        img_bis[comp[:, 0], comp[:, 1]] = i + 3

    colors = [c.hex_l for c in list(Color("red").range_to(Color("green"), GRADIENT_N))]
    cmap = col.ListedColormap(colors, 'indexed')

    return img_bis, cmap

#
# main
#
fig, ax = plt.subplots(2, 3, constrained_layout=True)

ax[0][0].set_title("original")
show_image(img_original_1, ax=ax[0][0], vmin=-1)
ax[0][1].set_title("otsu XOR thresh mean (contours)")
img_bis_1, cmap = generate_subgraph_image(
    prepare_image(img_original_1)
)
img_otsu = rgb2gray(img_original_1) > threshold_otsu(rgb2gray(img_original_1))
ax[0][2].set_title("connected-component label (areas)")
show_image(
    np.bitwise_xor(img_otsu, rgb2gray(img_original_1) > threshold_mean(rgb2gray(img_original_1))),
    ax=ax[0][1]
)
show_image(img_bis_1, ax=ax[0][2], cmap=cmap)

show_image(img_original_2, ax=ax[1][0], vmin=-1)
img_bis_2, cmap = generate_subgraph_image(
    prepare_image(img_original_2)
)
img_otsu = rgb2gray(img_original_2) > threshold_otsu(rgb2gray(img_original_2))
show_image(
    np.bitwise_xor(img_otsu, rgb2gray(img_original_2) > threshold_mean(rgb2gray(img_original_2))),
    ax=ax[1][1]
)
show_image(img_bis_2, ax=ax[1][2], cmap=cmap)

plt.show()