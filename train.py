from core.plmodel import FasterRCNNLightning
from pathlib import Path
from core.utils import load_config
from core.utils import load_augmentations
from core.pldatamodule import FasterRCNNDataModule

from core.plmodel import FasterRCNNLightning
from pytorch_lightning.loggers import TensorBoardLogger

import pytorch_lightning as pl

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
    valid_dataloader = data_module.val_dataloader()
    
    model = FasterRCNNLightning(num_classes=3)

    # logger
    create_path = Path('logs/my_model/').mkdir(parents=True, exist_ok=True)

    # check how many versions are already created in the logdir
    version = len([dir for dir in Path('logs/my_model/').iterdir() if dir.is_dir()])

    tb_logger = TensorBoardLogger('logs/', name='my_model', version=str(version))

    # log the best model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=f"logs/my_model/{version}/",
        filename='my-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(gpus=1,
                         max_epochs=25,
                         gradient_clip_algorithm='norm',
                         val_check_interval=0.1,
                         gradient_clip_val=3,
                         logger=tb_logger,
                         callbacks=[checkpoint_callback])
    
    trainer.fit(model, train_dataloader, valid_dataloader)

