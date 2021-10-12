import cv2
import numpy as np
import os
from os.path import join as ospj
from os.path import dirname as ospd

import sys
sys.path.append('/home/Anonymized/AL_OOD/code/util/')

from evaluation import BoxEvaluator
from evaluation import MaskEvaluator
from evaluation import configure_metadata
from utility import t2n

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224


def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam


class CAMComputer(object):
    def __init__(self, model, loader, metadata_root, mask_root,
                 iou_threshold_list, dataset_name, split,
                 multi_contour_eval, cam_curve_interval=.01, log_folder=None):
        self.model = model
        self.model.eval()
        self.loader = loader
        self.split = split
        self.log_folder = log_folder

        metadata = configure_metadata(metadata_root)
        cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))

        self.evaluator = BoxEvaluator(metadata=metadata,
                                          dataset_name=dataset_name,
                                          split=split,
                                          cam_threshold_list=cam_threshold_list,
                                          iou_threshold_list=iou_threshold_list,
                                          mask_root=mask_root,
                                          multi_contour_eval=multi_contour_eval)

    def compute_and_evaluate_cams(self, folder_name):
        print("Computing and evaluating cams.")
        for images, targets, image_ids in self.loader:
            image_size = images.shape[2:]
            images = images.cuda()
            cams = t2n(self.model(images, targets, return_cam=True))
            for cam, image_id in zip(cams, image_ids):
                cam_resized = cv2.resize(cam, image_size,
                                         interpolation=cv2.INTER_CUBIC)
                cam_normalized = normalize_scoremap(cam_resized)
                #if self.split in ('val', 'test'):
                #    cam_path = ospj(self.log_folder, folder_name, image_id)
                #    if not os.path.exists(ospd(cam_path)):
                #        os.makedirs(ospd(cam_path))
                #    np.save(ospj(cam_path), cam_normalized)
                    #cv2.imwrite(ospj(cam_path), cam_normalized)
                self.evaluator.accumulate(cam_normalized, image_id)
        return self.evaluator.compute()
