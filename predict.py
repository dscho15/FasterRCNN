from core.plmodel import FasterRCNNLightning
from pathlib import Path
from core.utils import load_augmentations
from core.pldatamodule import FasterRCNNDataModule

from core.plmodel import FasterRCNNLightning
from pytorch_lightning.loggers import TensorBoardLogger

import pytorch_lightning as pl
import torch

if __name__ == "__main__":

    # Path to model
    path_to_model = Path("/home/dts/Desktop/master_thesis/FasterRCNN/logs/my-model-epoch=00-val_loss=0.08.ckpt")

    # Path to data
    path_to_yaml = Path("/home/dts/Desktop/master_thesis/FasterRCNN/cfg/mvdd_params.yaml")

    if not path_to_yaml.exists():
        raise Exception("Path to data does not exist")

    train_aug, test_aug = load_augmentations()

    data_module = FasterRCNNDataModule(path_to_yaml=path_to_yaml, 
                                       train_transform=train_aug, 
                                       test_transform=test_aug)
    
    data_module.prepare_data()

    test_dataloader = data_module.test_dataloader()
    
    model = FasterRCNNLightning(str(path_to_model), num_classes=3)

    for idx, (images, targets) in enumerate(test_dataloader):

        model.model.eval()

        with torch.no_grad():

            images = list(image for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]

            outputs = model(images)
