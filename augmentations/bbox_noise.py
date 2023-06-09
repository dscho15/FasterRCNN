
import numpy as np
from albumentations import DualTransform

class BoundingBoxNoise(DualTransform):
    """Crop a random part of the input.
    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(self, mode: str = "uniform", always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        assert mode in ["uniform", "gaussian"]
        self.mode = mode

    def apply(self, img, **params):
        return img

    def get_params(self):
        return {}

    def apply_to_bbox(self, bbox, **params):
        # bbox = [x_min, y_min, x_max, y_max] in normalized coordinates 
        
        dx = bbox[2] - bbox[0] # x_max - x_min
        dy = bbox[3] - bbox[1] # y_max - y_min
        
        if self.mode == "uniform":
            xc = np.random.uniform(low=bbox[0], high=bbox[2])
            yc = np.random.uniform(low=bbox[1], high=bbox[3])
        if self.mode == "gaussian":
            xc = np.random.normal(bbox[0]+dx/2, scale=dx/8)
            yc = np.random.normal(bbox[1]+dy/2, scale=dy/8)
        
        return (xc - dx, yc - dy, xc + dx, yc + dy)

    def get_transform_init_args_names(self):
        return "mode"