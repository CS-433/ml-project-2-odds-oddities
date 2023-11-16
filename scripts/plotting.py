"""plotting.py: helper functions for plotting."""
from typing import Union

import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from distutils.spawn import find_executable

from scripts.array_manipulations import simplify_array
from scripts.evaluation import get_patched_f1, get_correct_mask, get_prediction
from scripts.preprocessing import get_patched_classification


def plot_images(axis: bool = True, tight_layout: bool = False, **images):
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
        plt.imshow(simplify_array(image), cmap="Greys_r")
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

    # use latex whenever possible
    plt.rc('text', usetex=bool(find_executable('latex')))
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


def plot_n_predictions(
        model,
        dataloader,
        num_images: int = 5,
        is_patched: bool = True,
        is_comparison: bool = True
):
    """
    Plot image with ground truth and prediction. If booleans are true,
        plot patched versions of the mask and predictions alongside
        the comparison of the two.

    :param model: trained
    :param dataloader: with batch_size=1
    :param num_images: to plot
    :param is_patched: if True -> plot patched mask and prediction
    :param is_comparison: if True -> plot patched comparison
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_count = 0

    for image, label in dataloader:
        image, label = image.to(device), label.to(device)
        predicted = get_prediction(model, image)
        print('f1: {:.2f}'.format(get_patched_f1(predicted, label)))

        extra_plots = {}
        if is_patched:
            extra_plots['ground_truth_16x16'] = get_patched_classification(label)
            extra_plots['predicted_16x16'] = get_patched_classification(predicted)
        if is_comparison:
            extra_plots['comparison'] = get_correct_mask(
                extra_plots['ground_truth_16x16'], extra_plots['predicted_16x16']
            )

        plot_images(
            axis=False,
            satellite=image,
            ground_truth=label,
            predicted=predicted,
            **extra_plots
        )

        image_count += 1
        if image_count >= num_images:
            break
