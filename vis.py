import torch
from dataset import FasterRCNNDataset
from pathlib import Path
from utils import load_config
from utils import load_augmentations

if __name__ == "__main__":

    # Path to data
    p_to_data = Path("/home/dts/Desktop/master_thesis/FasterRCNN/cfg/mvdd_params.yaml")

    if not p_to_data.exists():
        raise Exception("Path to data does not exist")
    
    cfg = load_config(p_to_data)

    # load training scenes
    train_scenes = cfg["train_scenes"]
    
    scenes = [Path(cfg["train_path"]) / scene for scene in train_scenes]

    train_aug, test_aug = load_augmentations()
    
    dataset = FasterRCNNDataset(scenes=scenes, transform=train_aug, infer=False)