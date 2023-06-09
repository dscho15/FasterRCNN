import yaml
import albumentations as A
from augmentations.debayering import DebayerArtefacts
from augmentations.unsharp_mask import Unsharpen

def load_config(path):

    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config

def load_augmentations(resize=(1080, 1080)):

    train_album = A.Compose([
                                A.GaussianBlur(p=0.5),
                                A.ISONoise(p=0.5),
                                A.GaussNoise(p=0.5),
                                A.CLAHE(p=0.5),
                                A.ColorJitter(p=0.5, brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
                                A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.5),
                                DebayerArtefacts(p=0.5),
                                Unsharpen(p=0.5),
                                A.BBoxSafeRandomCrop(p=0.5),
                                A.Resize(resize[0], resize[1], p=1.0),
                            ],
                            bbox_params=A.BboxParams(format="coco",
                                                     min_visibility=0.8,
                                                     label_fields=["labels"]))
    
    test_album = A.Compose([
                                A.Resize(resize[0], resize[1], p=1.0),
                            ],
                            bbox_params=A.BboxParams(format="coco",
                                                     min_visibility=0.1,
                                                     label_fields=["labels"]))

    return train_album, test_album