"""evaluation.py: helper scripts for evaluation."""
import json
import os
import torch

import numpy as np
import pandas as pd

from PIL import Image
from typing import Union
from sklearn.metrics import f1_score
from skimage.util import view_as_blocks
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


def save_csv_aicrowd(filename: str, model):
    """
    Save the csv with predictions of the test set. The script assumes that
        you have followed the data extraction pipeline and have directory
        ../data/raw/test in your project.

    :param filename: absolute path for the output csv
    :param model: used for predictions
    """
    root = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    ai_crowd_directory = os.path.join(root, 'data', 'raw', 'test')
    ai_crowd_paths = [os.path.join(ai_crowd_directory, f'test_{i + 1}.png') for i in range(50)]

    ai_crowd_dataset = RoadDataset(ai_crowd_paths)
    ai_crowd_dataloader = DataLoader(ai_crowd_dataset)

    _masks_to_submission(filename, model, ai_crowd_dataloader)


def _mask_to_submission_string(image_number: int, prediction: np.ndarray) -> str:
    """
    Convert mask to lines in csv.

    :param image_number: [1, 50]
    :param prediction: binary array of shape (600, 600)
    :return: "translated" entries for csv
    """
    patch_size = 16
    for j in range(0, prediction.shape[1], patch_size):
        for i in range(0, prediction.shape[0], patch_size):
            patch = prediction[i:i + patch_size, j:j + patch_size]
            label = get_class(patch)
            yield "{:03d}_{}_{},{}".format(image_number, j, i, label)


def _masks_to_submission(filename, model, dataloader):
    """
    Generate csv from the predictions of the mask.

    :param filename: to save the csv (absolute path)
    :param model: used for prediction
    :param dataloader: with batch_size = 1
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(filename, 'w') as f:
        f.write('id,prediction\n')

        for img_number, (image, _) in enumerate(dataloader, 1):
            image = image.to(device)
            predicted = get_prediction(model, image)
            f.writelines('{}\n'.format(s) for s in _mask_to_submission_string(img_number, predicted))


def _binary_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert from {0, 1} to {0, 255}."""
    return (img * 255).round().astype(np.uint8)


def reconstruct_from_labels(filepath: str, image_id: int, is_save: bool = False):
    """
    Convert data from CSV submission back to image mask.

    :param filepath: absolute path to the csv
    :param image_id: one out of the 50 test images [1, 50]
    :param is_save: boolean whether to save the img
    :return: image
    """
    img_width, img_height = 600, 600
    patch_w, patch_h = 16, 16
    im = np.zeros((img_width, img_height), dtype=np.uint8)
    f = open(filepath)
    lines = f.readlines()
    image_id_str = '%.3d_' % image_id

    for i in range(1, len(lines)):
        line = lines[i]
        if not image_id_str in line:
            continue

        tokens = line.split(',')
        id = tokens[0]
        prediction = int(tokens[1])
        tokens = id.split('_')
        i = int(tokens[1])
        j = int(tokens[2])

        je = min(j+patch_w, img_width)
        ie = min(i+patch_h, img_height)
        if prediction == 0:
            adata = np.zeros((patch_w, patch_h))
        else:
            adata = np.ones((patch_w, patch_h))

        im[j:je, i:ie] = _binary_to_uint8(adata)

    if is_save:
        Image.fromarray(im).save('prediction_' + '%.3d' % image_id + '.png')

    return im


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
        """
        Add metrics to the data dict based on setup and fold.

        :param setup: name of model as 'encoder+decoder'
        :param fold: starting from 0
        :param metrics: kwargs with metric values
        """
        for name, metric in metrics.items():

            setup = (setup, ) if setup == 'ensembling' else tuple(setup.split('+'))

            if setup not in self.data[name].keys():
                self.data[name][setup] = [[]]

            # folds start from zero, add new fold if necessary
            while len(self.data[name][setup]) <= fold:
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


class Ensembler:
    """Helper class for storing training and validation predictions for ensembling."""

    attributes = [
        'training_predictions', 'training_masks',
        'validation_predictions', 'validation_masks'
    ]

    def __init__(self):
        self._model = None
        self.data = dict((attr, {}) for attr in self.attributes)

    def set_model(self, encoder, decoder):
        """Set the current model for Ensembler object."""
        self._model = (encoder, decoder)

        for attr in self.attributes:
            self.data[attr][self._model] = []

    def update(self, predictions: torch.Tensor, masks: torch.Tensor, mode: str):
        """Update predictions and ground_truth in data to save them into JSON eventually."""
        mask_matrix = masks.clone().cpu().detach().numpy().tolist()
        pred_matrix = predictions.clone().cpu().detach().numpy().tolist()

        self.data[f'{mode}_masks'][self._model] += mask_matrix
        self.data[f'{mode}_predictions'][self._model] += pred_matrix

    def get_majority_vote(self, mode: str):
        """
        Calculate the inference (contained in self.data) according to majority vote.

        :param mode: either 'training' or  'validation'
        """
        predictions = self.data[f'{mode}_predictions']
        arrays = [(np.array(pred) >= 0.5).astype(int) for pred in predictions.values()]
        threshold = len(predictions) // 2

        return (np.add(*arrays) > threshold).astype(int)

    def get_f1(self, mode: str):
        """
        Make the inference and calculate f1 score of the fold.

        :param mode: either 'training' or  'validation'
        :return: f1 score over the fold
        """
        ground_truth = np.array(list(self.data[f'{mode}_masks'].values())[0])

        ground_truth_arr = ground_truth.reshape(-1)
        predicted_arr = self.get_majority_vote(mode).reshape(-1)

        return f1_score(ground_truth_arr, predicted_arr)

