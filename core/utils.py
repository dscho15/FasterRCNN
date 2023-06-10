import yaml
import albumentations as A
import cv2
from distinctipy import distinctipy
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
                                A.ColorJitter(p=0.5),
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

def draw_boxes(image_, boxes, labels, scores):

    distinctipy_colors = distinctipy.get_colors(3)

    image = image_.detach().cpu().numpy().transpose(1, 2, 0)

    for box, label, score in zip(boxes, labels, scores):

        if score < 0.7:
            continue

        x1, y1, x2, y2 = box

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        r, g, b = [int(c * 255) for c in distinctipy_colors[label-1]]
        
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (r, g, b), 2)
    
    image = image.astype('uint8')

    return image