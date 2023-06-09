import pytorch_lightning as pl
import torch

from torch import nn
from torch.nn import functional as F

import torchvision
from torchvision.models import ResNet18_Weights
from torchvision.models.detection.faster_rcnn import _validate_trainable_layers, _resnet_fpn_extractor, RPNHead, FastRCNNConvFCHead, _default_anchorgen
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

class FasterRCNNLightning(pl.LightningModule):

    def __init__(self, 
                 model_path: str = None, 
                 num_classes=3):
        
        super().__init__()


        self.model = self.__faster_rcnn_model(num_classes)

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))

    def forward(self, x):

        return self.model(x)
    
    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)

        return loss
    
    def predict_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self.model(x)

        return y_hat

    
    def configure_optimizers(self):
        
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
    
    def __faster_rcnn_model(self, n_classes):
        
        backbone = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, progress=True)
        backbone = _resnet_fpn_extractor(backbone, 3, norm_layer=nn.BatchNorm2d)

        rpn_anchor_generator = _default_anchorgen()
        
        rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
        
        box_head = FastRCNNConvFCHead((backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d)

        model = FasterRCNN(
            backbone,
            num_classes=n_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            rpn_head=rpn_head,
            box_head=box_head,
            min_size=2**8,
            max_size=2**10
        )

        return model
