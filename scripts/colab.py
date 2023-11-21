"""colab.py: helper function for more seamless training in Google Colab."""
import os
import sys


def get_root_path(is_colab=False) -> str:
    """
    Setup for local and Colab training. With colab, we need to cd to the correct directory
        and specify the root path

    :param is_colab: if colab is used or not
    :return: root path
    """
    if is_colab:
        # noinspection PyUnresolvedReferences
        from google.colab import drive
        drive.mount('/content/drive')

        # go to the drive directory
        root_path = '/content/drive/MyDrive/EPFL/Machine Learning/ml-project-2-odds-oddities'
        os.chdir(root_path)
        root_path = root_path

    else:
        root_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)

    sys.path.append(root_path)
    return root_path