import os
from pathlib import Path
from torch import utils

import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader

from core.dataset import FasterRCNNDataset

def collate_fn(batch):
    return tuple(zip(*batch))

class FasterRCNNDataModule(pl.LightningDataModule):

    def __init__(self,
                 path_to_yaml=Path("./cfg/mvdd_params.yaml"),
                 **kwargs):

        super().__init__()

        # Assign params
        self.path_to_yaml = path_to_yaml

        self.train_scenes = []
        self.val_scenes = []
        self.test_scenes = []

        self.train_transform = kwargs.get("train_transform", None)
        self.test_transform = kwargs.get("test_transform", None)

        self.obj_ids = kwargs.get("obj_ids", [1, 2, 3])

        self.num_workers = min(os.cpu_count() - 4, 8)

    def prepare_data(self):

        # Load yaml files
        with open(str(self.path_to_yaml)) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        train_scenes = data['train_scenes']
        train_path = Path(data['train_path'])

        val_scenes = data['val_scenes']
        val_path = Path(data['val_path'])

        test_scenes = data.get('test_scenes')
        test_path = Path(data['test_path'])

        self.train_scenes = [train_path / scene for scene in train_scenes]
        self.val_scenes = [val_path / scene for scene in val_scenes]
        self.test_scenes = [test_path / scene for scene in test_scenes]

    def train_dataloader(self):
        train_split = FasterRCNNDataset(scenes=self.train_scenes,
                                        transform=self.test_transform,
                                        obj_ids=self.obj_ids,
                                        infer=True)

        return DataLoader(train_split,
                          batch_size=16,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=collate_fn)

    def val_dataloader(self):
        valid_split = FasterRCNNDataset(scenes=self.val_scenes,
                                        transform=self.test_transform,
                                        obj_ids=self.obj_ids,
                                        infer=True)

        return DataLoader(valid_split,
                          batch_size=16,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=collate_fn)

    def test_dataloader(self):
        test_split = FasterRCNNDataset(scenes=self.test_scenes,
                                       transform=self.test_transform,
                                       obj_ids=self.obj_ids,
                                       infer=True)

        return DataLoader(test_split,
                          batch_size=1,
                          shuffle=False,
                          num_workers=1,
                          collate_fn=collate_fn)

    def __repr__(self):
        return f"{self.__class__.__name__}, \npath_to_yaml: {self.path_to_yaml}"
