import glob
import torch
import hydra
from tqdm import tqdm
import os
import cv2
import time
import os.path as osp
import numpy as np
import natsort
import torchvision

from loguru import logger
from torch.utils.data import DataLoader
from src.utils import data_utils, path_utils, eval_utils, vis_utils
from src.utils.bbox_3D_utils import compute_3dbbox_from_sfm
from src.utils.model_io import load_network
from src.local_feature_2D_detector import LocalFeatureObjectDetector
from deep_spectral_method.detection_2D_utils import UnsupBbox
from bbox_3D_estimation.utils import (
    Detector3D,
    read_list_poses,
    sort_path_list,
    predict_3D_bboxes,
)
from pytorch_lightning import seed_everything

"""Inference & visualize"""
from src.datasets.normalized_dataset import NormalizedDataset
from src.sfm.extract_features import confs

seed_everything(12345)


if torch.cuda.is_available():
    device = "cuda"
    compute_on_GPU = True
    logger.info("Running OnePose with GPU, will it work?")
else:
    device = "cpu"
    compute_on_GPU = False
    logger.info("Running OnePose with CPU")


def get_default_paths(cfg, data_root, data_dir, sfm_model_dir):
    anno_dir = osp.join(
        sfm_model_dir, f"outputs_{cfg.network.detection}_{cfg.network.matching}", "anno"
    )
    avg_anno_3d_path = osp.join(anno_dir, "anno_3d_average.npz")
    clt_anno_3d_path = osp.join(anno_dir, "anno_3d_collect.npz")
    idxs_path = osp.join(anno_dir, "idxs.npy")
    sfm_ws_dir = osp.join(
        sfm_model_dir,
        f"outputs_{cfg.network.detection}_{cfg.network.matching}",
        "sfm_ws",
        "model",
    )

    img_lists = []
    color_dir = osp.join(data_dir, "color_full")
    img_lists += glob.glob(color_dir + "/*.png", recursive=True)

    img_lists = natsort.natsorted(img_lists)

    # Visualize detector:
    vis_detector_dir = osp.join(data_dir, "detector_vis")
    if osp.exists(vis_detector_dir):
        os.system(f"rm -rf {vis_detector_dir}")
    os.makedirs(vis_detector_dir, exist_ok=True)
    det_box_vis_video_path = osp.join(data_dir, "det_box.mp4")

    # Visualize pose:
    vis_box_dir = osp.join(data_dir, "pred_vis")
    if osp.exists(vis_box_dir):
        os.system(f"rm -rf {vis_box_dir}")
    os.makedirs(vis_box_dir, exist_ok=True)
    demo_video_path = osp.join(data_dir, "demo_video.mp4")

    intrin_full_path = osp.join(data_dir, "intrinsics.txt")
    paths = {
        "data_root": data_root,
        "data_dir": data_dir,
        "sfm_model_dir": sfm_model_dir,
        "sfm_ws_dir": sfm_ws_dir,
        "avg_anno_3d_path": avg_anno_3d_path,
        "clt_anno_3d_path": clt_anno_3d_path,
        "idxs_path": idxs_path,
        "intrin_full_path": intrin_full_path,
        "vis_box_dir": vis_box_dir,
        "vis_detector_dir": vis_detector_dir,
        "det_box_vis_video_path": det_box_vis_video_path,
        "demo_video_path": demo_video_path,
    }
    return img_lists, paths


def load_model(cfg):
    """Load model"""

    def load_matching_model(model_path):
        """Load onepose model"""
        from src.models.GATsSPG_lightning_model import LitModelGATsSPG

        trained_model = LitModelGATsSPG.load_from_checkpoint(checkpoint_path=model_path)
        trained_model.to(device)
        trained_model.eval()

        return trained_model

    def load_extractor_model(cfg, model_path):
        """Load extractor model(SuperPoint)"""
        from src.models.extractors.SuperPoint.superpoint import SuperPoint
        from src.sfm.extract_features import confs

        extractor_model = SuperPoint(confs[cfg.network.detection]["conf"])
        extractor_model.to(device)
        extractor_model.eval()
        load_network(extractor_model, model_path, force=True)

        return extractor_model

    matching_model = load_matching_model(cfg.model.onepose_model_path)
    extractor_model = load_extractor_model(cfg, cfg.model.extractor_model_path)
    return matching_model, extractor_model


def load_2D_matching_model(cfg):
    def load_2D_matcher(cfg):
        from src.models.matchers.SuperGlue.superglue import SuperGlue
        from src.sfm.match_features import confs

        match_model = SuperGlue(confs[cfg.network.matching]["conf"])
        match_model.eval()
        match_model.to(device)
        load_network(match_model, cfg.model.match_model_path)
        return match_model

    matcher = load_2D_matcher(cfg)
    return matcher


def pack_data(avg_descriptors3d, clt_descriptors, keypoints3d, detection, image_size):
    """Prepare data for OnePose inference"""
    keypoints2d = torch.Tensor(detection["keypoints"])
    descriptors2d = torch.Tensor(detection["descriptors"])

    inp_data = {
        "keypoints2d": keypoints2d[None].to(device),  # [1, n1, 2]
        "keypoints3d": keypoints3d[None].to(device),  # [1, n2, 3]
        "descriptors2d_query": descriptors2d[None].to(device),  # [1, dim, n1]
        "descriptors3d_db": avg_descriptors3d[None].to(device),  # [1, dim, n2]
        "descriptors2d_db": clt_descriptors[None].to(device),  # [1, dim, n2*num_leaf]
        "image_size": image_size,
    }

    return inp_data


def inference_core(
    cfg,
    data_root,
    seq_dir,
    sfm_model_dir,
    object_det_type="detection",
    box_3D_detect_type="image_based",
    verbose=False,
):

    BboxPredictor = UnsupBbox(downscale_factor=0.3, on_GPU=compute_on_GPU)

    # Load models and prepare data:
    matching_model, extractor_model = load_model(cfg)
    matching_2D_model = load_2D_matching_model(cfg)
    img_lists, paths = get_default_paths(cfg, data_root, seq_dir, sfm_model_dir)

    # sort images
    im_ids = [int(osp.basename(i).replace(".png", "")) for i in img_lists]
    im_ids.sort()
    img_lists = [
        osp.join(osp.dirname(img_lists[0]), f"{im_id}.png") for im_id in im_ids
    ]
    K, _ = data_utils.get_K(paths["intrin_full_path"])

    sfm_ws_dir = paths["sfm_ws_dir"]
    anno_3d_box = data_root + "/box3d_corners.txt"

    if box_3D_detect_type == "image_based" or not os.path.exists(anno_3d_box):
        logger.info(
            f"3d bbox estimated with {box_3D_detect_type} method, reading from file"
        )

        intriscs_path = seq_dir + "/intrinsics.txt"

        K_anno, _ = data_utils.get_K(intriscs_path)
        poses_list_anno = glob.glob(
            os.path.join(os.getcwd(), f"{seq_dir}/poses", "*.txt")
        )
        poses_list_anno = sort_path_list(poses_list_anno)
        img_lists_anno = glob.glob(
            os.path.join(os.getcwd(), f"{seq_dir}/color_full", "*.png")
        )
        img_lists_anno = sort_path_list(img_lists_anno)

        predict_3D_bboxes(
            BboxPredictor=BboxPredictor,
            img_lists=img_lists_anno,
            poses_list=poses_list_anno,
            K=K_anno,
            data_root=data_root,
            step=10,
        )

        logger.info(f"built from file {anno_3d_box}")
        print("done")

    else:
        logger.info(f"reading from file {anno_3d_box}")

    box3d_path = path_utils.get_3d_box_path(data_root, annotated=False)

    local_feature_obj_detector = LocalFeatureObjectDetector(
        extractor_model,
        matching_2D_model,
        n_ref_view=10,
        sfm_ws_dir=sfm_ws_dir,
        output_results=False,
        detect_save_dir=paths["vis_detector_dir"],
        device=device,
    )

    # Prepare 3D features:
    num_leaf = cfg.num_leaf
    avg_data = np.load(paths["avg_anno_3d_path"])
    clt_data = np.load(paths["clt_anno_3d_path"])
    idxs = np.load(paths["idxs_path"])

    keypoints3d = torch.Tensor(clt_data["keypoints3d"]).to(device)
    num_3d = keypoints3d.shape[0]
    # load average 3D features:
    avg_descriptors3d, _ = data_utils.pad_features3d_random(
        avg_data["descriptors3d"], avg_data["scores3d"], num_3d
    )
    # load corresponding 2D features of each 3D point:
    clt_descriptors, _ = data_utils.build_features3d_leaves(
        clt_data["descriptors3d"], clt_data["scores3d"], idxs, num_3d, num_leaf
    )

    dataset = NormalizedDataset(
        img_lists, confs[cfg.network.detection]["preprocessing"]
    )
    loader = DataLoader(dataset, num_workers=1)

    for id, data in enumerate(tqdm(loader)):
        img_path = data["path"][0]
        with torch.no_grad():
            inp = data["image"].to(device)
            # Detect object:
            # Use 3D bbox and previous frame's pose to yield current frame 2D bbox:
            start = time.time()
            if object_det_type == "features":
                bbox, inp_crop, K_crop = local_feature_obj_detector.detect(
                    inp,
                    img_path,
                    K,
                )
            elif object_det_type == "detection":
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                bbox_orig_res = BboxPredictor.infer_2d_bbox(image=image, K=K)

                inp_crop, K_crop = local_feature_obj_detector.crop_img_by_bbox(
                    query_img_path=img_path,
                    bbox=bbox_orig_res,
                    K=K,
                    crop_size=512,
                )
                inp_crop = torchvision.transforms.functional.to_tensor(
                    inp_crop
                ).unsqueeze(0)
                inp_crop = inp_crop.to(device)

            if verbose:
                logger.info(
                    f"feature matching runtime: {(time.time() - start)%60} seconds"
                )

            # Detect query image(cropped) keypoints and extract descriptors:
            pred_detection = extractor_model(inp_crop)
            pred_detection = {k: v[0].cpu().numpy() for k, v in pred_detection.items()}

            # 2D-3D matching by GATsSPG:
            inp_data = pack_data(
                avg_descriptors3d,
                clt_descriptors,
                keypoints3d,
                pred_detection,
                data["size"],
            )
            pred, _ = matching_model(inp_data)
            matches = pred["matches0"].detach().cpu().numpy()
            valid = matches > -1
            kpts2d = pred_detection["keypoints"]
            kpts3d = inp_data["keypoints3d"][0].detach().cpu().numpy()
            confidence = pred["matching_scores0"].detach().cpu().numpy()
            mkpts2d, mkpts3d, mconf = (
                kpts2d[valid],
                kpts3d[matches[valid]],
                confidence[valid],
            )

            # Estimate object pose by 2D-3D correspondences:
            _, pose_pred_homo, inliers = eval_utils.ransac_PnP(
                K_crop, mkpts2d, mkpts3d, scale=1000
            )

        pose_opt = pose_pred_homo

        # Visualize:
        vis_utils.save_demo_image(
            pose_opt,
            K,
            image_path=img_path,
            box3d_path=box3d_path,
            draw_box=len(inliers) > 3,
            save_path=osp.join(paths["vis_box_dir"], f"{id}.jpg"),
        )

    # Output video to visualize estimated poses:
    vis_utils.make_video(paths["vis_box_dir"], paths["demo_video_path"])


def inference(cfg):
    data_dirs = cfg.input.data_dirs
    sfm_model_dirs = cfg.input.sfm_model_dirs
    if isinstance(data_dirs, str) and isinstance(sfm_model_dirs, str):
        data_dirs = [data_dirs]
        sfm_model_dirs = [sfm_model_dirs]

    for data_dir, sfm_model_dir in tqdm(
        zip(data_dirs, sfm_model_dirs), total=len(data_dirs)
    ):
        splits = data_dir.split(" ")
        data_root = splits[0]
        for seq_name in splits[1:]:
            seq_dir = osp.join(data_root, seq_name)
            logger.info(f"Eval {seq_dir}")
            inference_core(cfg, data_root, seq_dir, sfm_model_dir)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg):
    globals()[cfg.type](cfg)


if __name__ == "__main__":
    main()
