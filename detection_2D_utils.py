from extract import extract
from extract import extract_utils as utils

class UnsupBbox():
    def __init__(self, feature_dir, full_seg_dir) -> None:
        self.model_name = "dino_vits16"
        self.feature_dir = feature_dir
        self.full_seg_dir = full_seg_dir
        self.model, self.val_transform, self.patch_size, self.num_heads = utils.get_model(self.model_name)

        pass


    def infer_2d_bbox(self, query_img, images_root):    
        dataset = utils.ImagesDataset(
            filenames=query_img, images_root=images_root, transform=self.val_transform
        )

        # Load image with PIL, maybe we can optimize this part
        #image_PIL = Image.open(images_root + "/" + filenames[k])

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

        # Segmentation
        segmap = extract.extract_single_region_segmentations(
            feature_dict=feature_dict, eigs_dict=eigs_dict,
        )

        # Bounding boxes
        bbox = extract.extract_bboxes(feature_dict=feature_dict, segmap=segmap,)
        limits = bbox["bboxes_original_resolution"][0]
        return limits
