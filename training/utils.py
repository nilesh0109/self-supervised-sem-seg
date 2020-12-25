from typing import List, Tuple
import numpy as np
import matplotlib
from config import defaults

if defaults.MATPLOTLIB_NO_GUI:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import math
import torch
from pathlib import Path

from sem_seg.datasets.datautils import CityscapesUtils

DIRNAME = Path(__file__).parents[1].resolve()
OUTDIR = DIRNAME / "sem_seg" / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

cityscape_utils = CityscapesUtils()


def sanitize_imgs(images: List) -> List:
    """Converts the list of torch image Tensors in (N, C, H, W) format to list of numpy images in <N,H,W,C> format"""
    for i in range(len(images)):
        if torch.is_tensor(images[i]):
            images[i] = images[i].cpu().numpy().squeeze()
        if images[i].ndim > 3 and images[i].shape[-1] != 3:
            images[i] = np.moveaxis(images[i], 1, -1)
    return images


def is_mask(img: np.array) -> bool:
    """
    Checks if the image is a segmentation mask
    Criteria: If its a 2D image and has values in range of num_classes of the dataset
    :param img: Image
    :return: True if the image is a segmentation mask, False otherwise
    """
    return img.ndim == 2 and 0 <= len(np.unique(img)) < cityscape_utils.num_classes + 1


def plot_images(imgs, targets=None, preds=None, title='No title', num_cols=6) -> plt.Figure:
    """
    Plot the images, targets and predictions in a grid. Shows 24 images at max in the plot.
    :param imgs: List of images to be plotted.
    :param targets: Corresponding List of ground truths.
    :param preds: Corresponding List of network predictions
    :param title: Title of the figure
    :param num_cols: Number of columns in the grid. Default is 6
    :return: output plot figure object
    """
    inputs = [imgs]
    if targets is not None:
        inputs.append(targets)
    if preds is not None:
        inputs.append(preds)
    inputs = sanitize_imgs(inputs)
    num_types = len(inputs)
    num_rows, num_cols, plot_width, plot_height = get_plot_sizes(inputs, num_cols)
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(plot_width, plot_height), num=title, dpi=220,
                           gridspec_kw={'wspace': 0.05, 'hspace': 0.05, 'left': 0, 'top': 0.95})
    [axi.set_axis_off() for axi in ax.ravel()]
    for i in range(len(inputs[0])):
        r, c = (num_types * i) // num_cols, (num_types * i) % num_cols
        if r >= num_rows: break
        for j in range(num_types):
            img = inputs[j][i]
            img_to_show = cityscape_utils.label2color(img) if is_mask(img) else img
            ax[r][c + j].imshow(img_to_show)
    if title:
        fig.suptitle(title, fontsize=8)
    if defaults.MATPLOTLIB_NO_GUI:
        plt.savefig(str(OUTDIR) + '/plot2.png', bbox_inches='tight')
    else:
        plt.show()
    return fig


def get_plot_sizes(inputs: List, num_cols: int) -> Tuple:
    """ Given the input images, return the dimension of the plot i.e. num_rows, num_cols, Height, Width"""
    num_types, num_images = len(inputs), len(inputs[0])
    num_rows = min(4, math.ceil((num_images * num_types) / num_cols))
    if inputs[0].ndim == 4:
        N, C, H, W = inputs[0].shape
    elif inputs[0].ndim == 3:
        C, H, W = inputs[0].shape
    else:
        H, W = inputs[0].shape
    aspect_ratio = W / H
    return num_rows, num_cols, num_cols * aspect_ratio * 0.8, num_rows


def save_results(data, filename):
    filename = str(OUTDIR) + '/' + filename + '.csv'
    np.savetxt(filename, data, fmt='%0.6f')
    print(f'saved at {filename}')
