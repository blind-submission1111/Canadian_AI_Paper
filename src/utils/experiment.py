import random
import time
import os

def get_new_experiment_data_dir(experiment_dir: str, dataset: str = None, model: str = None) -> str:
    random_num = random.randint(0, 9999999)
    timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{random_num}"
    folder_name = timestamp
    if dataset:
        folder_name += f"_{dataset}"
    if model:
        folder_name += f"_{model}"
    experiment_data_dir = os.path.join(experiment_dir, folder_name)
    os.makedirs(experiment_data_dir, exist_ok=True)
    return experiment_data_dir