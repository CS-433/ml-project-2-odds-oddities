"""preprocessing.py: helper functions and Class for preprocessing."""
import os

import numpy as np
from PIL import Image
from matplotlib import image as mpimg
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import albumentations as A


IMG_PATCH_SIZE = 16
IMG_WIDTH = 400
IMG_HEIGHT = 400
N_PATCHES_PER_IMAGE = (IMG_WIDTH / IMG_PATCH_SIZE) * (IMG_HEIGHT / IMG_PATCH_SIZE)


class RoadDataset(Dataset):
    """
    Dataset class for preprocessing satellite images.

    :param image_paths: local absolute path of images
    :param mask_paths: local absolute path of ground truths
    """

    def __init__(self, image_paths, mask_paths, transform=None):
        #super().__init__(image_paths, *args, **kwargs)

        # read images in
        self.images = [mpimg.imread(path) for path in image_paths]
        self.masks = [mpimg.imread(path) for path in mask_paths]
        self.transform = transform

    def __getitem__(self, i):
        image = np.array(Image.fromarray((self.images[i] * 255).astype(np.uint8)).resize((512, 512))) / 255
        trimap = np.array(Image.fromarray((self.masks[i] * 255).astype(np.uint8)).resize((512, 512)))
        mask = np.where(trimap > 128, 1, 0)
        
        image = image.astype(np.float32)    # Previous division creates float64 by default, but augmentations can only handle one of [uint8, float32]
        mask = mask.astype(np.uint8)        # Mask values are [0,1] so uint8 is more reasonable: np.uint32 -> np.uint8
        
        if self.transform:
            # Apply same transformation to image and mask
            # NB! This must be done before converting to Pytorch format
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        # convert to Pytorch format HWC -> CHW
        image = np.moveaxis(image, -1, 0)
        mask = np.expand_dims(mask, 0)

        return image, mask

    def __len__(self):
        return len(self.images)


def split_data(images_path: str, test_size: float):
    """
    Split list of absolute paths to training and test data by using sklearn.

    :param images_path: absolute path of the parent directory of images
    :param test_size: value [0, 1]
    :return: image_path_train, image_path_test, mask_path_train, mask_path_test
    """
    # specify image and ground truth full path
    image_directory = os.path.join(images_path, "images")
    labels_directory = os.path.join(images_path, "groundtruth")

    # specify absolute paths for all files
    image_paths = [os.path.join(image_directory, image) for image in sorted(os.listdir(image_directory))]
    mask_paths = [os.path.join(labels_directory, image) for image in sorted(os.listdir(labels_directory))]

    return train_test_split(image_paths, mask_paths, test_size=test_size)


def get_patched_array(array: np.ndarray, step: int) -> np.ndarray:
    """
    As the goal is to return label for 16x16 pixel patches, this function helps to
        patch the ground truth or prediction using this logic

    :param array: of shape (x, x)
    :param step: of size
    :return:
    """
    patched_img = np.zeros(array.shape)
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch

    for x in range(0, array.shape[0], step):
        for y in range(0, array.shape[1], step):
            patch_mean = np.mean(array[y:y+step, x:x+step])
            patched_img[y:y+step, x:x+step] = 1 if patch_mean > foreground_threshold else 0

    return patched_img


