import pytorch_lightning as pl
import torch

from torch import nn
from torch.nn import functional as F

import torchvision
from core.fasterrcnn import FasterRCNN

from torchvision.models import ResNet18_Weights
from torchvision.models.detection.faster_rcnn import RPNHead, FastRCNNConvFCHead, _default_anchorgen, FastRCNNPredictor
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

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

        images, targets = batch

        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        self.log('train_loss', losses)

        return losses
    
    def validation_step(self, batch, batch_idx):

        images, targets = batch

        self.model.train()

        with torch.no_grad():

            images = list(image for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
            self.log('val_loss', losses)

        return losses
    
    def predict_step(self, batch, batch_idx):

        images, targets = batch

        with torch.no_grad():

            images = list(image for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]
        
            loss_dict = self.model(images, targets)

            print(loss_dict)
            losses = sum(loss for loss in loss_dict.values())
        
            self.log('val_loss', losses)

        return losses

    
    def configure_optimizers(self):
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        return [optimizer], [lr_scheduler]
    
    def __faster_rcnn_model(self, n_classes):
        
        backbone = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, progress=True)
        backbone = self.__resnet_fpn_extractor(backbone, 3, norm_layer=nn.BatchNorm2d)

        rpn_anchor_generator = _default_anchorgen()
        
        rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)

        box_head = FastRCNNConvFCHead((backbone.out_channels, 7, 7), [128, 128, 128, 128], [256], norm_layer=nn.BatchNorm2d)

        model = FasterRCNN(
            backbone,
            num_classes=n_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            rpn_head=rpn_head,
            box_head=box_head,
            box_roi_pool=roi_pooler,
            min_size=2**8,
            max_size=2**10
        )

        # model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(num_classes=n_classes, pretrained_backbone=True, trainable_backbone_layers=1, min_size=2**8, max_size=2**10)

        return model
    
    def __resnet_fpn_extractor(
            self,
            backbone: nn.Module,
            trainable_layers: int,
            returned_layers = None,
            extra_blocks = None,
            norm_layer = None,
        ):

        # select layers that wont be frozen
        if trainable_layers < 0 or trainable_layers > 5:
            raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
        
        layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]

        if trainable_layers == 5:
            layers_to_train.append("bn1")

        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        if returned_layers is None:
            returned_layers = [1, 2, 3, 4]

        if min(returned_layers) <= 0 or max(returned_layers) >= 5:
            raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")
        
        return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

        in_channels_stage2 = backbone.inplanes // 8
        in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
        out_channels = 128

        return BackboneWithFPN(
            backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
        )
