"""array_manipulations.py: helper scripts for working with Tensors and np.ndarray."""
import torch

import numpy as np

from skimage.util import view_as_blocks
from typing import Union

PATCH_SIZE = 16


def _is_color_image(array: np.ndarray) -> bool:
    """
    Check if there is 3 (color image) or 1 (mask) channels.

    :param array: no requirements to shape
    :return: True if is color image
    """
    return 3 in array.shape


def _is_chw(array: np.ndarray) -> bool:
    """
    Check if channel is first dimension in the array.

    :param array: of shape (x, x, x)
    :return: True of channel is the first dimension
    """
    return array.shape[0] < array.shape[1]


def simplify_array(image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    This function has 3 goals:
        1. convert Tensor to numpy array
        2. if color-image -> transpose to shape (height, width, channel)
        3. if binary image -> squeese to shape (height, width)

    NB! Defined twice in order to avoid circular imports.

    :param image: of arbitrary shape
    :return: array with simplified structure
    """
    image = image.cpu().numpy().squeeze() if isinstance(image, torch.Tensor) else image

    if _is_color_image(image) and _is_chw(image):
        return image.transpose(1, 2, 0)
    elif not _is_color_image(image) and _is_chw(image):
        return image.squeeze()
    return image


def get_patched_array(array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Return 16x16 blocks/patches as a np.ndarray of shape (N, 16, 16)

    :param array: of arbitrary shape
    :return: 3d array with patches
    """
    array = simplify_array(array)

    block_shape = (PATCH_SIZE, PATCH_SIZE)
    # calculate the number of blocks we should have
    num_blocks = int((array.shape[0] / PATCH_SIZE) ** 2)

    return view_as_blocks(array, block_shape=block_shape).reshape(num_blocks, PATCH_SIZE, PATCH_SIZE)
