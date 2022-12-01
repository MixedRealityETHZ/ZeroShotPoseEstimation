from .extract import extract
from .extract import extract_utils as utils
import os
import cv2

# TODO: extend this class to load dino model on the constructor

class UnsupBbox():
    def __init__(self, feature_dir, full_seg_dir=None, downscale_factor=0.3, on_GPU=False) -> None:
        self.model_name = "dino_vits16"
        self.feature_dir = feature_dir
        self.full_seg_dir = full_seg_dir
        self.num_workers = 0 # decrease this if out_of_memory error
        self.downscale_factor = downscale_factor
        self.on_GPU = on_GPU
        self.model, self.val_transform, self.patch_size, self.num_heads = utils.get_model(self.model_name)

    def downscale_image(self, image):
        return cv2.resize(image, (0, 0), fx=self.downscale_factor, fy=self.downscale_factor)

    def infer_2d_bbox(self, image, K): 
        self.K = K   
        image_half = self.downscale_image(image)
        feature_dict = extract.extract_features(
            model=self.model,
            patch_size=self.patch_size,
            num_heads=self.num_heads,
            images=image_half,
        )

        eigs_dict = extract._extract_eig(K=4, data_dict=feature_dict, on_gpu=self.on_GPU)

        # small Segmentation
        segmap = extract.extract_single_region_segmentations(feature_dict=feature_dict, eigs_dict=eigs_dict)

        # Bounding boxes
        bbox = extract.extract_bboxes(feature_dict=feature_dict, segmap=segmap)
        bbox_orig_res = bbox["bboxes_original_resolution"][0]
        return bbox_orig_res


