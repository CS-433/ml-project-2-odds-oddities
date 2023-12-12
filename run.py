"""run.py: Predict using the pretrained models and save it down to csv."""
import os
from pathlib import Path

from scripts.inference import load_tuned_models, save_csv_aicrowd

model_names = [("resnet18", "unet"), ("resnet18", "unet"), ("resnet18", "unet")]

root_path = Path(__file__).parent
state_dict_root = os.path.join(root_path, "data", "results", "final_models")


if __name__ == "__main__":
    models = load_tuned_models(model_names, state_dict_root)
    save_csv_aicrowd('out.csv', models)
