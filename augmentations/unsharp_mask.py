import numpy as np
import albumentations as A
import cv2

class Unsharpen(A.ImageOnlyTransform):
    def __init__(self, k_limits=(3, 7), strength_limits=(0., 2.), p=0.5):
        super().__init__()
        self.k_limits = k_limits
        self.strength_limits = strength_limits
        self.p = p

    def apply(self, img, **params):
        if np.random.rand() > self.p:
            return img
        k = np.random.randint(self.k_limits[0] // 2, self.k_limits[1] // 2 + 1) * 2 + 1
        s = k / 3
        blur = cv2.GaussianBlur(img, (k, k), s)
        strength = np.random.uniform(*self.strength_limits)
        unsharpened = cv2.addWeighted(img, 1 + strength, blur, -strength, 0)
        return unsharpened