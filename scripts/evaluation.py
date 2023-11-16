"""evaluation.py: helper scripts for evaluation."""
import os
from typing import Union

import numpy as np
import albumentations as A
import torch
from skimage.util import view_as_blocks

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from scripts.array_manipulations import simplify_array
from scripts.preprocessing import get_class, RoadDataset


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
    prediction_sigmoid = logits.sigmoid().cpu().numpy().squeeze()
    return np.where(prediction_sigmoid >= 0.5, 1, 0)


def get_patched_f1(output: np.ndarray, target: np.ndarray) -> float:
    """
    Return the f1 score based on the patched logic. As the classification task
        requires label for every 16x16 patch, we need to calculate the final
        f1 using the very same logic.

    :param output: with predicted labels for each pixel
    :param target: ground truth
    :return: f1 score in [0, 1]
    """
    output_patches = get_image_as_patches_array(output)
    target_patches = get_image_as_patches_array(target)

    output_labels = [get_class_by_patch(output_block) for output_block in output_patches]
    target_labels = [get_class_by_patch(target_block) for target_block in target_patches]

    # flatten the array to (N, 1)
    flat_target_labels = np.concatenate(np.array(target_labels)).ravel().tolist()
    flat_output_labels = np.concatenate(np.array(output_labels)).ravel().tolist()

    return f1_score(flat_target_labels, flat_output_labels)


def get_image_as_patches_array(image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """TODO: update"""
    patches = view_as_blocks(simplify_array(image), (16, 16))
    return patches.reshape(-1, patches.shape[2], patches.shape[3])


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


def get_class_by_patch(array: np.ndarray) -> list:
    """
    Return classification results {0, 1} per patch for the whole image.

    :param array: of shape (height, width) and height=width
    :return: list with values in {0, 1} for (height / 16) * (width / 16) samples
    """
    patches = view_as_blocks(simplify_array(array), (16, 16))
    patches_array = patches.reshape(-1, patches.shape[2], patches.shape[3])
    labels = [get_class(block) for block in patches_array]
    return labels


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


def save_csv_aicrowd(filename, model):
    """

    :param filename:
    :param model:
    :return:
    """
    ROOT_PATH = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    ai_crowd_directory = os.path.join(ROOT_PATH, 'data', 'raw', 'test')
    ai_crowd_paths = [os.path.join(ai_crowd_directory, f'test_{i + 1}.png') for i in range(50)]

    ai_crowd_dataset = RoadDataset(ai_crowd_paths)
    ai_crowd_dataloader = DataLoader(ai_crowd_dataset)

    masks_to_submission(filename, model, ai_crowd_dataloader)


def mask_to_submission_string(image_number, prediction):
    """

    :param image_number:
    :param prediction:
    :return:
    """
    patch_size = 16
    for j in range(0, prediction.shape[1], patch_size):
        for i in range(0, prediction.shape[0], patch_size):
            patch = prediction[i:i + patch_size, j:j + patch_size]
            label = get_class(patch)
            yield "{:03d}_{}_{},{}".format(image_number, j, i, label)


def masks_to_submission(filename, model, dataloader):
    """

    :param filename:
    :param model:
    :param dataloader:
    :return:
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(filename, 'w') as f:
        f.write('id,prediction\n')

        for img_number, (image, _) in enumerate(dataloader, 1):
            image = image.to(device)
            predicted = get_prediction(model, image)
            f.writelines('{}\n'.format(s) for s in mask_to_submission_string(img_number, predicted))
