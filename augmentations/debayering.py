import albumentations as A
import numpy as np
import cv2

class DebayerArtefacts(A.ImageOnlyTransform):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def apply(self, img, **params):
        
        if np.random.rand() > self.p:
            return img
        
        assert img.dtype == np.uint8
        # permute channels before bayering/debayering to cover different bayer formats
        channel_idxs = np.random.permutation(3)
        channel_idxs_inv = np.empty(3, dtype=int)
        channel_idxs_inv[channel_idxs] = 0, 1, 2

        # assemble bayer image
        bayer = np.zeros(img.shape[:2], dtype=img.dtype)
        bayer[::2, ::2] = img[::2, ::2, channel_idxs[2]]
        bayer[1::2, ::2] = img[1::2, ::2, channel_idxs[1]]
        bayer[::2, 1::2] = img[::2, 1::2, channel_idxs[1]]
        bayer[1::2, 1::2] = img[1::2, 1::2, channel_idxs[0]]

        # debayer
        debayer_method = np.random.choice((cv2.COLOR_BAYER_BG2BGR, cv2.COLOR_BAYER_BG2BGR_EA))
        debayered = cv2.cvtColor(bayer, debayer_method)[..., channel_idxs_inv]
        return debayered