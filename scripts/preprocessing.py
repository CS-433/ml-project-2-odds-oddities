"""TODO: add description."""
import os

import numpy as np
from PIL import Image
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

    def __getitem__(self, i):
        image = np.array(Image.fromarray((self.images[i] * 255).astype(np.uint8)).resize((512, 512))) / 255
        trimap = np.array(Image.fromarray((self.masks[i] * 255).astype(np.uint8)).resize((512, 512)))
        mask = np.where(trimap > 128, 1, 0)

        # convert to Pytorch format HWC -> CHW
        image = np.moveaxis(image, -1, 0)
        mask = np.expand_dims(mask, 0)

        return image, mask

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


if __name__ == '__main__':
    ROOT_PATH = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    train_directory = os.path.join(ROOT_PATH, 'data', 'raw', 'training')
    image_path_train, image_path_test, mask_path_train, mask_path_test = split_data(train_directory, 0.2)

    train_dataset = RoadDataset(image_path_train, mask_path_train)
    x = train_dataset[1]

    kala = 1

