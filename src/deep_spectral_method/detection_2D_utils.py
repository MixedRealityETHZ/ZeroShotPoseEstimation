from .extract import extract
from .extract import extract_utils as utils
import torch
import numpy as np
import cv2
import lmfit
import matplotlib.pyplot as plt

# TODO: extend this class to load dino model on the constructor


class UnsupBbox:
    def __init__(self, downscale_factor=0.6, device="cpu") -> None:
        downscale_factor=0.6
        self.model_name = "dino_vits16"
        self.num_workers = 0  # decrease this if out_of_memory error
        self.downscale_factor = downscale_factor
        self.device = device
        self.fitting_model = lmfit.models.Gaussian2dModel() #Model to fit the gaussian curve

        if device=="cpu":
            self.on_GPU = False
        elif device=="mps":
            self.on_GPU = False
        elif device=="cuda":
            self.on_GPU = True
        elif device=="gpu":
            self.on_GPU = True
        else:
            print("\n unknown device")

        (
            self.model,
            self.val_transform,
            self.patch_size,
            self.num_heads,
        ) = utils.get_model(self.model_name)

        self.model = self.model.to(device)

    def downscale_image(self, image):
        return cv2.resize(
            image, (0, 0), fx=self.downscale_factor, fy=self.downscale_factor
        )

    def infer_2d_bbox(self, image, K):
        self.K = K
        image_half = self.downscale_image(image)
        image_half = self.val_transform(image_half)
        c, h, w = image_half.shape
        image_half = image_half.reshape((1, c, h, w))
        feature_dict = extract.extract_features(
            model=self.model,
            patch_size=self.patch_size,
            num_heads=self.num_heads,
            images=image_half,
            device=self.device,
        )

        eigs_dict = extract._extract_eig(
            K=4, data_dict=feature_dict, device=self.device
        )

        eigs_dict_gaussian = eigs_dict.copy()

        # small segmentation, use of fitting curve
        segmap = extract.gaussian_fitting(
        feature_dict=feature_dict,
        eigs_dict=eigs_dict_gaussian,
        fitting_model=self.fitting_model
        )

        bbox = extract.extract_bboxes(
        feature_dict=feature_dict,
        segmap=segmap,
        )
        
        #If we cannot find a solution with the filter then use the original bbox
        if not bbox['bboxes']:
            # small Segmentation
            segmap = extract.extract_single_region_segmentations(
                feature_dict=feature_dict, eigs_dict=eigs_dict
            )
            # Bounding boxes
            bbox = extract.extract_bboxes(feature_dict=feature_dict, segmap=segmap)
        
        bbox_orig_res = (
            np.array(bbox["bboxes_original_resolution"][0]) / self.downscale_factor
        )

        # small Segmentation
        segmap_2 = extract.extract_single_region_segmentations(
            feature_dict=feature_dict, eigs_dict=eigs_dict
        )
        # Bounding boxes
        bbox_2 = extract.extract_bboxes(feature_dict=feature_dict, segmap=segmap_2)

        if not bbox_2['bboxes']:
            return bbox_orig_res, []
        else:
            bbox_orig_res_2 = (
            np.array(bbox_2["bboxes_original_resolution"][0]) / self.downscale_factor
            )
            return bbox_orig_res, bbox_orig_res_2
