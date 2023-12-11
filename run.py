"""TODO: specify"""
import os
from pathlib import Path

from scripts.inference import load_tuned_models


model_names = [("resnet18", "unet")]

root_path = Path(__file__).parent
state_dict_root = os.path.join(root_path, "data", "results", "hyperopt")


if __name__ == "__main__":
    models = load_tuned_models(model_names, state_dict_root)

    # 1. get inference data one by one (dataloader)
    # 2. predict for all 3 models
    # 3. apply ensembling
    # 4. write down to file
