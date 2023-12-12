"""evaluation.py: helper scripts for evaluation."""
import json
import os
from collections import defaultdict
from typing import Union

import numpy as np
import pandas as pd

import torch
from skimage.util import view_as_blocks

from sklearn.metrics import f1_score

from scripts.array_manipulations import simplify_array
from scripts.inference import get_prediction
from scripts.preprocessing import get_class


class MetricMonitor:
    """
    Inspired from examples of Albumentation:
        https://albumentations.ai/docs/examples/pytorch_classification/
    """
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.metrics = {}
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def averages(self):
        """Return the average per metric (loss, f1)"""
        return tuple([metric['avg'] for (metric_name, metric) in self.metrics.items()])

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


class EvaluationMonitor:
    """Helper class for storing training and validation loss and f1 scores."""

    files = ['training_f1', 'training_loss', 'validation_f1', 'validation_loss']

    def __init__(self, jsons_path: str):
        self.jsons_path = jsons_path
        self.data = {}

        for metric in self.files:
            filepath = os.path.join(jsons_path, f'{metric}.json')
            self.data[metric] = self._get_dict(filepath)

    def get_not_updated_models(self) -> list:
        """Return the list of models that don't have metrics logged yet."""
        return [key for key, value in self.data['validation_f1'].items() if not value]

    def update_metrics(self, setup: str, **metrics):
        """Update the metrics in dictionary."""
        for name, metric in metrics.items():
            self.data[name][setup] = metric

    def update_metrics_by_fold(self, setup: str, fold: int, **metrics):
        """TODO: update"""
        for name, metric in metrics.items():

            if setup not in self.data[name]:
                self.data[name][setup] = [[]]

            # folds start from zero, add new fold if necessary
            if len(self.data[name][setup]) <= fold:
                self.data[name][setup].append([])

            self.data[name][setup][fold].append(metric)

    def update_jsons(self):
        """Update json based on dictionary."""
        for file in self.files:
            filepath = os.path.join(self.jsons_path, f'{file}.json')
            json_file = open(filepath, 'w+')
            data = self._tuple_key_to_string(self.data[file])
            json_file.write(json.dumps(data))
            json_file.close()

    def _get_dict(self, filepath: str):
        """Get dictionary from json. If necessary convert strings with '+'
            (combination in setup) to tuples."""
        json_file = open(filepath, 'r')
        data = json.load(json_file)
        json_file.close()
        return self._string_key_to_tuple(data)

    @staticmethod
    def _tuple_key_to_string(data: dict) -> dict:
        """Convert dictionary keys from the instance of tuple to strings concatenated with '+'.
            It's necessary due to json-s incapability to handle tuples."""
        if any(isinstance(key, tuple) for key in data.keys()):
            return {'+'.join(key): value for key, value in data.items()}
        return data

    @staticmethod
    def _string_key_to_tuple(data: dict) -> dict:
        """Convert dictionary keys from the instance of string concatenated with '+' to tuple.
            String is necessary due to json-s incapability with tuples."""
        if any('+' in key for key in data.keys()):
            return {tuple(key.split('+')): value for key, value in data.items()}
        return data


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
    """
    Take the image of arbitrary size and convert it to array of 16x16 patches.

    :param image: of arbitrary shape (simplify array can handle it)
    :return: array of shape (N, 16, 16)
    """
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


def get_best_f1_per_setup(setup: dict):
    """Based on the dictionary, return the best f1 per setup."""
    best_f1s = []
    std_devs = []
    setups = list(setup.keys())

    for setup, matrix in setup.items():
        matrix = np.array(matrix)
        mean_per_epoch = matrix.mean(axis=0)

        best_epoch = mean_per_epoch.argmax()
        best_f1 = mean_per_epoch.max()

        best_f1s.append(best_f1)
        std_devs.append(matrix[:, best_epoch].std())

    data = np.array([best_f1s, std_devs]).T

    return pd.DataFrame(data, index=setups, columns=['top_f1', 'std_dev'])
