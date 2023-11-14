import numpy as np
from skimage.util import view_as_blocks
from sklearn.metrics import f1_score

from scripts.plotting import make_image_plottable
from scripts.preprocessing import get_class


def get_patched_f1(output: np.ndarray, target: np.ndarray, side_length: int = 16) -> float:
    """
    Return the f1 score based on the patched logic. As the classification task
        requires label for every 16x16 patch, we need to calculate the final
        f1 using the very same logic.

    :param output: with predicted labels for each pixel
    :param target: ground truth
    :param side_length: usually 16 pixels
    :return: f1 score in [0, 1]
    """
    output, target = make_image_plottable(output), make_image_plottable(target)

    block_shape = (side_length, side_length)
    # calculate the number of blocks we should have
    num_blocks = int((target.shape[0] / side_length) ** 2)

    output_blocks = (view_as_blocks(output, block_shape=block_shape).
                     reshape(num_blocks, side_length, side_length))
    target_blocks = (view_as_blocks(target, block_shape=block_shape).
                     reshape(num_blocks, side_length, side_length))

    target_labels = [get_class(target_block) for target_block in target_blocks]
    output_labels = [get_class(output_block) for output_block in output_blocks]

    return f1_score(target_labels, output_labels)


def get_correct_mask(label: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    # Convert boolean array to RGB image
    bool_array = label == predicted
    shape = bool_array.shape

    image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

    image[bool_array] = [0, 127, 45]  # green
    image[~bool_array] = [227, 6, 19]  # red

    return image
