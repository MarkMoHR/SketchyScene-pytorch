import argparse
import numpy as np
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt
import os


def visualize_semantic_segmentation(label_array, color_map, black_bg=False, save_path=None):
    """
    tool for visualizing semantic segmentation for a given label array

    :param label_array: [H, W], contains [0-nClasses], 0 for background
    :param color_map: array read from 'colorMapC46.mat'
    :param black_bg: the background is black if set True
    :param save_path: path for saving the image
    """
    visual_image = np.zeros((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)
    if not black_bg:
        visual_image.fill(255)

    ## read all colors
    colors_list = []
    for i in range(color_map.shape[0]):
        colors_list.append(color_map[i][1][0])
    colors_list = np.array(colors_list)

    ## assign color to drawing regions
    visual_image[label_array != 0] = colors_list[label_array[label_array != 0] - 1]

    plt.imshow(visual_image)
    plt.show()

    ## save visualization
    if save_path is not None:
        visual_image = Image.fromarray(visual_image, 'RGB')
        visual_image.save(save_path)