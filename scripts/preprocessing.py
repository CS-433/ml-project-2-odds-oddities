"""TODO: add description."""
import os

import numpy as np
from matplotlib import image as mpimg
from torch.utils.data import Dataset


IMG_PATCH_SIZE = 16
IMG_WIDTH = 400
IMG_HEIGHT = 400
N_PATCHES_PER_IMAGE = (IMG_WIDTH / IMG_PATCH_SIZE) * (IMG_HEIGHT / IMG_PATCH_SIZE)


def one_hot_encode(label):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label

    # Returns
        A 2D array with the same width and height as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in [0, 1]:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


class RoadDataset(Dataset):
    """Read images from the specified path.

    Args:
        image_path (str): directory containing true labels and images
    """

    def __init__(self, image_path):
        # specify image and ground truth full path
        image_directory = os.path.join(image_path, "images")
        labels_directory = os.path.join(image_path, "groundtruth")

        # specify absolute paths for all files
        image_paths = [os.path.join(image_directory, image) for image in sorted(os.listdir(image_directory))]
        mask_paths = [os.path.join(labels_directory, image) for image in sorted(os.listdir(labels_directory))]

        # read images in
        self.images = [mpimg.imread(path) for path in image_paths]
        self.masks = [mpimg.imread(path) for path in mask_paths]

        # generate batches from images
        img_batches = np.asarray([self.get_batches(image) for image in self.images])
        dim_1, dim_2, dim_3, dim_4, dim_5 = img_batches.shape
        self.batches = img_batches.reshape((dim_1 * dim_2, dim_3, dim_4, dim_5))

        # generate batches from ground truth
        mask_batches = np.asarray([self.get_batches(mask) for mask in self.masks])
        masks = mask_batches.reshape((dim_1 * dim_2, dim_3, dim_4, dim_5))

    def __getitem__(self, i):
        mask = None
        return self.batches[i], mask

    def __len__(self):
        return len(self.images)

    @staticmethod
    def value_to_class(percentage):
        foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
        if percentage > foreground_threshold:  # road
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


def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches


def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(gt_imgs)
    gt_patches = [
        img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ]
    data = np.asarray(
        [
            gt_patches[i][j]
            for i in range(len(gt_patches))
            for j in range(len(gt_patches[i]))
        ]
    )
    labels = np.asarray(
        [value_to_class(np.mean(data[i])) for i in range(len(data))]
    )

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)


if __name__ == '__main__':
    ROOT_PATH = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    train_directory = os.path.join(ROOT_PATH, 'data', 'raw', 'training')

    labels = extract_labels(os.path.join(train_directory, 'groundtruth/'), 1)
    data = RoadDataset(train_directory)

    kala = 1


