"""inference.py: helper Classes and functions for generating predictions."""
import os
from pathlib import Path

from PIL import Image
import numpy as np
import torch

import segmentation_models_pytorch as smp
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from scripts.preprocessing import RoadDataset, get_class, get_preprocessing
from segmentation_models_pytorch.encoders import get_preprocessing_fn


class Ensembler:
    """Helper class for storing training and validation predictions for ensembling."""

    attributes = [
        'training_predictions', 'training_masks',
        'validation_predictions', 'validation_masks'
    ]

    def __init__(self):
        self._model = None
        self.data = dict((attr, {}) for attr in self.attributes)
        self.inference = {}

    def set_model(self, encoder, decoder):
        """Set the class attribute that keeps the current model."""
        self._model = (encoder, decoder)

        for attr in self.attributes:
            self.data[attr][self._model] = []

    def update(self, predictions: torch.Tensor, masks: torch.Tensor, mode: str):
        """Update predictions and ground_truth in data to save them into JSON eventually."""
        mask_matrix = masks.clone().cpu().detach().numpy().tolist()
        pred_matrix = predictions.clone().cpu().detach().numpy().tolist()

        self.data[f'{mode}_masks'][self._model] += mask_matrix
        self.data[f'{mode}_predictions'][self._model] += pred_matrix

    def add_inference(self, predictions: np.ndarray, model: str):
        """Add predictions to the inference dict."""
        self.inference[model] = predictions

    def get_majority_vote(self, mode: str = None) -> np.ndarray:
        """
        Get majority vote of the models and using the mode.

        :param mode: either 'training' or 'validation'
        :return: ensembling result
        """

        if mode:
            predictions = self.data[f'{mode}_predictions']
            arrays = [(np.array(pred) >= 0.5).astype(int) for pred in predictions.values()]
        else:
            # used for final inference
            arrays = list(self.inference.values())

        # if we use only one model
        if len(arrays) > 1:
            threshold = len(arrays) // 2
            return (np.add(*arrays) > threshold).astype(int)

        return arrays[0].astype(int)

    def get_f1(self, mode: str):
        """Calculate f1 score for the ensembling result."""
        # it doesn't matter which one we take
        ground_truth = np.array(list(self.data[f'{mode}_masks'].values())[0])

        ground_truth_arr = ground_truth.reshape(-1)
        predicted_arr = self.get_majority_vote(mode).reshape(-1)

        return f1_score(ground_truth_arr, predicted_arr)


@torch.no_grad()
def get_prediction(model, image: torch.Tensor, class_threshold: float = 0.5) -> np.ndarray:
    """
    Return prediction for the specific image.

    :param model: used for inference
    :param image: torch.Tensor
    :param class_threshold: for deciding the class
    :return: segmented image
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = image.to(device)
    model.eval()
    logits = model(image.float())
    prediction_sigmoid = logits.sigmoid().cpu().numpy().squeeze()
    return np.where(prediction_sigmoid >= class_threshold, 1, 0)


def load_tuned_models(model_names: list, directory: str):
    """
    Load the tuned models from the

    :param model_names:
    :param directory: that contains model objects
    :return:
    """
    models = []
    for encoder, decoder in model_names:

        model = smp.create_model(decoder, encoder_name=encoder)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state_dict_path = os.path.join(directory, f"{encoder}+{decoder}.pth")
        state_dict = torch.load(state_dict_path, map_location=device)["state_dict"]
        model.load_state_dict(state_dict)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        model.eval()

        models.append(model)

    return models


def save_csv_aicrowd(filename, models, **kwargs):
    """
    Save the csv with predictions of the test set. The script assumes that
        you have followed the data extraction pipeline and have directory
        ../data/raw/test in your project.

    :param filename: absolute path for the output csv
    :param models: used for predictions
    """
    root = Path(__file__).parent.parent  # go 2 dirs back
    ai_crowd_directory = os.path.join(root, "data", "raw", "test")
    ai_crowd_paths = [
        os.path.join(ai_crowd_directory, f"test_{i + 1}.png") for i in range(50)
    ]

    # need to have separate datasets for models due to encoder specific transforms
    encoders = [model.name.split('-', 1)[1] for model in models]
    preproc_fns = [get_preprocessing_fn(encoder) for encoder in encoders]

    datasets = [RoadDataset(ai_crowd_paths, preprocess=get_preprocessing(fn)) for fn in preproc_fns]
    dataloaders = [DataLoader(ds) for ds in datasets]

    _masks_to_submission(filename, models, dataloaders, **kwargs)


def _mask_to_submission_string(image_number: int, prediction: np.ndarray, **kwargs) -> str:
    """
    Convert mask to lines in csv.

    :param image_number: [1, 50]
    :param prediction: binary array of shape (600, 600)
    :return: "translated" entries for csv
    """
    patch_size = 16
    for j in range(0, prediction.shape[1], patch_size):
        for i in range(0, prediction.shape[0], patch_size):
            patch = prediction[i : i + patch_size, j : j + patch_size]
            label = get_class(patch, kwargs['foreground_threshold'])
            yield "{:03d}_{}_{},{}".format(image_number, j, i, label)


def _masks_to_submission(filename, models, dataloaders, **kwargs):
    """
    Generate csv from the predictions of the mask.

    :param filename: to save the csv (absolute path)
    :param models: used for prediction
    :param dataloaders: for every model with batch_size=1
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(filename, "w") as f:
        f.write("id,prediction\n")

        for img_number, loaders in enumerate(zip(*dataloaders), 1):

            ensembler = Ensembler()

            for i, (model, (image, _)) in enumerate(zip(models, loaders)):
                decoder, encoder = model.name.split('-', 1)
                ensembler.set_model(encoder, decoder)  # as we don't care about the model type

                image = image.to(device)

                predicted = get_prediction(model, image, kwargs['class_threshold'])
                ensembler.add_inference(predicted, str(i))

            final_prediction = ensembler.get_majority_vote()

            f.writelines(
                "{}\n".format(s)
                for s in _mask_to_submission_string(img_number, final_prediction, **kwargs)
            )


def _binary_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert from {0, 1} to {0, 255}."""
    return (img * 255).round().astype(np.uint8)


def reconstruct_from_labels(filepath: str, image_id: int, is_save: bool = False):
    """
    Convert data from CSV submission back to image mask.
        Template provided by professors.

    :param filepath: absolute path to the csv
    :param image_id: one out of the 50 test images [1, 50]
    :param is_save: boolean whether to save the img
    :return: image
    """
    img_width, img_height = 608, 608
    patch_w, patch_h = 16, 16
    im = np.zeros((img_width, img_height), dtype=np.uint8)
    f = open(filepath)
    lines = f.readlines()
    image_id_str = "%.3d_" % image_id

    for i in range(1, len(lines)):
        line = lines[i]
        if not image_id_str in line:
            continue

        tokens = line.split(",")
        id = tokens[0]
        prediction = int(tokens[1])
        tokens = id.split("_")
        i = int(tokens[1])
        j = int(tokens[2])

        je = min(j + patch_w, img_width)
        ie = min(i + patch_h, img_height)
        if prediction == 0:
            adata = np.zeros((patch_w, patch_h))
        else:
            adata = np.ones((patch_w, patch_h))

        im[j:je, i:ie] = _binary_to_uint8(adata)

    if is_save:
        Image.fromarray(im).save("prediction_" + "%.3d" % image_id + ".png")

    return im
