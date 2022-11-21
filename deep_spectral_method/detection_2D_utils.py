from .extract import extract
from extract import extract_utils as utils
from src.local_feature_2D_detector import crop_img_by_bbox

class UnsupBbox():
    def __init__(self, feature_dir, full_seg_dir=None) -> None:
        self.model_name = "dino_vits16"
        self.feature_dir = feature_dir
        self.full_seg_dir = full_seg_dir
        self.model, self.val_transform, self.patch_size, self.num_heads = utils.get_model(self.model_name)


    def infer_2d_bbox(self, query_img, images_root, K): 
        self.K = K            
        dataset = utils.ImagesDataset(
            filenames=query_img, images_root=images_root, transform=self.val_transform
        )

        feature_dict = extract.extract_features(
            output_dir=self.feature_dir,
            batch_size=1,
            num_workers=0,
            model_name=self.model_name,
            model=self.model,
            patch_size=self.patch_size,
            num_heads=self.num_heads,
            dataset=dataset,
        )

        eigs_dict = extract.extract_eigs(
            which_matrix="laplacian", K=2, data_dict=feature_dict, image_file=dataset,
        )

        # small Segmentation
        segmap = extract.extract_single_region_segmentations(
            feature_dict=feature_dict, eigs_dict=eigs_dict,
        )

        # Bounding boxes
        bbox = extract.extract_bboxes(feature_dict=feature_dict, segmap=segmap,)
        bbox_orig_res = bbox["bboxes_original_resolution"][0]

        self.crop_size = int(max(bbox_orig_res[0]-bbox_orig_res[2], bbox_orig_res[1]-bbox_orig_res[3]))

        return crop_img_by_bbox(query_img_path=images_root, bbox=bbox_orig_res, K=self.K, crop_size=self.crop_size)


