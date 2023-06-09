import torch
from core.pldatamodule import FasterRCNNDataModule
from pathlib import Path
from core.utils import load_config
from core.utils import load_augmentations

if __name__ == "__main__":

    # Path to data
    path_to_yaml = Path("/home/dts/Desktop/master_thesis/FasterRCNN/cfg/mvdd_params.yaml")

    if not path_to_yaml.exists():
        raise Exception("Path to data does not exist")

    train_aug, test_aug = load_augmentations()

    data_module = FasterRCNNDataModule(path_to_yaml=path_to_yaml, 
                                       train_transform=train_aug, 
                                       test_transform=test_aug)
    
    data_module.prepare_data()

    train_dataloader = data_module.train_dataloader()

    for idx, (images, targets) in enumerate(train_dataloader):
        
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        break