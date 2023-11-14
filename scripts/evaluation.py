"""evaluation.py: helper scripts for evaluation."""
import numpy as np
import torch

from sklearn.metrics import f1_score
from scripts.preprocessing import get_class, get_patched_array


@torch.no_grad()
def get_prediction(model, image) -> np.ndarray:
    """
    Return prediction for the specific image.

    :param model: used for inference
    :param image: torch.Tensor
    :return: segmented image
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = image.to(device)
    model.eval()
    logits = model(image.float())
    return np.where(logits.cpu().numpy().squeeze() >= 0.5, 1, 0)


def get_patched_f1(output: np.ndarray, target: np.ndarray) -> float:
    """
    Return the f1 score based on the patched logic. As the classification task
        requires label for every 16x16 patch, we need to calculate the final
        f1 using the very same logic.

    :param output: with predicted labels for each pixel
    :param target: ground truth
    :return: f1 score in [0, 1]
    """
    output_blocks = get_patched_array(output)
    target_blocks = get_patched_array(target)

    target_labels = [get_class(target_block) for target_block in target_blocks]
    output_labels = [get_class(output_block) for output_block in output_blocks]

    return f1_score(target_labels, output_labels)


def get_class_by_patch(array: np.ndarray) -> list:
    """
    Return classification results {0, 1} per patch for the whole image.

    :param array: of shape (height, width) and height=width
    :return: list with values in {0, 1} for (height / 16) * (width / 16) samples
    """
    blocks = get_patched_array(array)
    labels = [get_class(block) for block in blocks]
    return labels


def get_correct_mask(label: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Compare prediction with actual and convert boolean mask to red-green image.

    :param label: ground truth, notice that it must be the same size as 'predicted'
    :param predicted: self-explanatory
    :return: ndarray with similar shape to the inputs, but with 3 channels
    """
    bool_array = label == predicted

    shape = bool_array.shape
    image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

    image[bool_array] = [0, 127, 45]  # green
    image[~bool_array] = [227, 6, 19]  # red

    return image


def get_test_f1(model, dataloader) -> float:
    """
    Get f1 score for the whole test dataset.

    :param model: must be trained
    :param dataloader: with test data
    :return: f1 score
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    target_labels = []
    output_labels = []

    for image, label in dataloader:
        image, label = image.to(device), label.to(device)
        predicted = get_prediction(model, image)
        target_labels += get_class_by_patch(label)
        output_labels += get_class_by_patch(predicted)

    return f1_score(target_labels, output_labels)
