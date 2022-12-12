from .extract import extract
from .extract import extract_utils as utils
import torch
import numpy as np
import cv2

# TODO: extend this class to load dino model on the constructor


class UnsupBbox:
    def __init__(self, downscale_factor=0.3, on_GPU=True) -> None:
        self.model_name = "dino_vits16"
        self.num_workers = 0  # decrease this if out_of_memory error
        self.downscale_factor = downscale_factor
        self.on_GPU = on_GPU
        (
            self.model,
            self.val_transform,
            self.patch_size,
            self.num_heads,
        ) = utils.get_model(self.model_name)
        self.model = self.model.to(
            "cuda" if on_GPU and torch.cuda.is_available() else "cpu"
        )

    def downscale_image(self, image):
        return cv2.resize(
            image, (0, 0), fx=self.downscale_factor, fy=self.downscale_factor
        )

    def infer_2d_bbox(self, image, K, viz=False):
        if viz: 
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
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
            on_GPU=self.on_GPU,
        )

        eigs_dict = extract._extract_eig(
            K=2, data_dict=feature_dict, on_gpu=self.on_GPU
        )

        # small Segmentation
        segmap = extract.extract_single_region_segmentations(
            feature_dict=feature_dict, eigs_dict=eigs_dict
        )

        # Bounding boxes
        bbox = extract.extract_bboxes(feature_dict=feature_dict, segmap=segmap)
        bbox_orig_res = (
            np.array(bbox["bboxes_original_resolution"][0]) // self.downscale_factor
        )

        if viz:
            fig = plt.figure(num=42)
            plt.clf()
            plt.imshow(image, alpha=0.9)
            plt.gca().add_patch(
                Rectangle(
                    (bbox_orig_res[0], bbox_orig_res[1]),
                    bbox_orig_res[2] - bbox_orig_res[0],
                    bbox_orig_res[3] - bbox_orig_res[1],
                    edgecolor="red",
                    facecolor="none",
                    lw=4,
                )
            )
            plt.show(block=False)
            plt.pause(0.0001)
        return bbox_orig_res
