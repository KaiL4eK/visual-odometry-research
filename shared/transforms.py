import cv2
import numpy as np

from albumentations.augmentations import functional as F
from albumentations.core.transforms_interface import DualTransform

class ResizeKeepingRatio(DualTransform):
    """Rescale an image so that maximum side is equal to max_size, keeping the aspect ratio of the initial image.

    Args:
        target_wh (tuple): target size to embed image.
        interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, target_wh=(1024, 768), interpolation=1, always_apply=False, p=1):
        super(ResizeKeepingRatio, self).__init__(always_apply, p)
        self.interpolation = interpolation
        self.target_wh = np.array(target_wh).astype(np.float32)

    def apply(self, img, interpolation=1, **params):
        im_sz = np.array([params["cols"], params["rows"]])
        scale = min(self.target_wh/im_sz)
        return F.scale(img, scale=scale, interpolation=interpolation)
    
    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox
    
    def apply_to_keypoint(self, keypoint, **params):
        im_sz = np.array([params["cols"], params["rows"]])
        scale = min(self.target_wh/im_sz)
        return F.keypoint_scale(keypoint, scale, scale)

    def get_transform_init_args_names(self):
        return ("target_wh", "interpolation")              

