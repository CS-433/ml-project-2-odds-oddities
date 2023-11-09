"""TODO: add description."""
import os

import numpy as np
from matplotlib import image as mpimg
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


IMG_PATCH_SIZE = 16
IMG_WIDTH = 400
IMG_HEIGHT = 400
N_PATCHES_PER_IMAGE = (IMG_WIDTH / IMG_PATCH_SIZE) * (IMG_HEIGHT / IMG_PATCH_SIZE)


class RoadDataset(Dataset):
    """Read images from the specified path.

    Args:
        image_paths (str): TODO: update
        image_paths (str): TODO: update
    """

    def __init__(self, image_paths, mask_paths):

        # read images in
        self.images = [mpimg.imread(path) for path in image_paths]
        self.masks = [mpimg.imread(path) for path in mask_paths]

        # generate batches from images
        img_batches = np.asarray([self.get_batches(image) for image in self.images])
        dim_1, dim_2, dim_3, dim_4, dim_5 = img_batches.shape
        self.image_batches = img_batches.reshape((dim_1 * dim_2, dim_3, dim_4, dim_5))

        # generate batches from ground truth
        mask_batches = np.asarray([self.get_batches(mask) for mask in self.masks])
        dim_1, dim_2, dim_3, dim_4 = mask_batches.shape
        self.mask_batches = mask_batches.reshape((dim_1 * dim_2, dim_3, dim_4))
        # 1-hot-encode the ground truth
        self.ohe_mask = np.asarray(
            [self.one_hot_encode(np.mean(batch)) for batch in self.mask_batches]
        ).astype(np.float32)

    def __getitem__(self, i):
        return self.image_batches[i], self.ohe_mask[i]

    def __len__(self):
        return len(self.images)

    @staticmethod
    def one_hot_encode(percentage):
        foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
        if percentage > foreground_threshold:  # road  TODO: wouldn't vice versa make sense?
            return [0, 1]
        else:  # background
            return [1, 0]

    @staticmethod
    def get_batches(image):
        list_patches = []

        for i in range(0, IMG_HEIGHT, IMG_PATCH_SIZE):
            for j in range(0, IMG_WIDTH, IMG_PATCH_SIZE):
                list_patches.append(image[j:j + IMG_PATCH_SIZE, i:i + IMG_PATCH_SIZE])

        return list_patches


def split_data(images_path: str, test_size: float):
    # specify image and ground truth full path
    image_directory = os.path.join(images_path, "images")
    labels_directory = os.path.join(images_path, "groundtruth")

    # specify absolute paths for all files
    image_paths = [os.path.join(image_directory, image) for image in sorted(os.listdir(image_directory))]
    mask_paths = [os.path.join(labels_directory, image) for image in sorted(os.listdir(labels_directory))]

    return train_test_split(image_paths, mask_paths, test_size=test_size)

