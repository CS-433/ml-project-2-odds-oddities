"""run.py: Predict using the pretrained models and save it down to csv."""
import os
from pathlib import Path

from scripts.inference import load_tuned_models, save_csv_aicrowd
from scripts.training import setup_seed

model_names = [
    ("efficientnet-b4", "UnetPlusPlus"),
    ("inceptionv4", "UnetPlusPlus"),
    ("mit_b2", "Unet"),
]

# hyperparams
foreground_threshold = 0.2
class_threshold = 0.1
seed = 16


if __name__ == "__main__":
    setup_seed(seed)

    # get the project root path
    root_path = Path(__file__).parent
    state_dict_root = os.path.join(root_path, "data", "results", "final_models")

    models = load_tuned_models(model_names, state_dict_root)
    save_csv_aicrowd(
        f"out_fgt{foreground_threshold}_ct{class_threshold}.csv",
        models=models,
        foreground_threshold=foreground_threshold,
        class_threshold=class_threshold,
    )
