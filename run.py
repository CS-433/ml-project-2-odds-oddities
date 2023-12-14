"""run.py: Predict using the pretrained models and save it down to csv."""
import os
from pathlib import Path

from scripts.inference import load_tuned_models, save_csv_aicrowd

# inceptionnet-0.2-0.4

model_names = [
    # ("efficientnet-b4", "UnetPlusPlus")
    ("inceptionv4", "UnetPlusPlus"),
    # ("mit_b2", "Unet")
]

root_path = Path(__file__).parent
state_dict_root = os.path.join(root_path, "data", "results", "final_models")

foreground_threshold = 0.2  # 0.2  best so far
class_threshold = 0.25  # 0.4  best so far


if __name__ == "__main__":
    models = load_tuned_models(model_names, state_dict_root)
    save_csv_aicrowd(
        f'inceptionv4_new_fgt{foreground_threshold}_ct{class_threshold}.csv',
        models=models,
        foreground_threshold=foreground_threshold,
        class_threshold=class_threshold
    )
