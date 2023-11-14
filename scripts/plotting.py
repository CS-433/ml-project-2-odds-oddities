"""plotting.py: helper functions for plotting."""
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker


def _is_color_image(array: np.ndarray) -> bool:
    return 3 in array.shape


def _is_chw(array: np.ndarray) -> bool:
    return array.shape[0] < array.shape[1]


def make_image_plottable(image: np.ndarray) -> np.ndarray:
    if _is_color_image(image) and _is_chw(image):
        return image.transpose(1, 2, 0)
    elif not _is_color_image(image) and _is_chw(image):
        return image.squeeze()
    return image


def show_images(axis: bool = True, tight_layout: bool = False, **images):
    """
    Plot images next to each other.

    :param axis: show if True
    :param tight_layout: self-explanatory
    :param images: kwargs as title=image
    """
    image_count = len(images)
    plt.figure(figsize=(image_count * 3, 3))
    for i, (name, image) in enumerate(images.items()):

        plt.subplot(1, image_count, i + 1)
        plt.axis('off') if not axis else None
        # get title from the parameter names
        plt.title(name.replace('_', ' ').title(), fontsize=14)
        plt.imshow(make_image_plottable(image), cmap="Greys_r")
    plt.tight_layout() if tight_layout else None
    plt.show()


def plot_metric_per_epoch(
        train: Union[list, np.ndarray],
        validation: Union[list, np.ndarray],
        y_label: str,
        title: str = None
):
    """
    Plot two-line graph with metric per epoch.

    :param train: with train metric
    :param validation: with validation metric
    :param y_label: usually the name of metric
    :param title: if necessary
    """
    plt.set_cmap('Set2')
    plt.rc('text', usetex=True)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Dark2.colors)

    # force epochs to integers
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

    x = np.arange(len(train))
    plt.plot(x, train, label='train')
    plt.plot(x, validation, label='validation')

    plt.xlabel('epoch', fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    if title:
        plt.title(title, fontsize=20)

    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    plt.grid(color="#d3d3d3", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=16)
    plt.tight_layout()


