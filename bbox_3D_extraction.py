import glob
from tqdm import tqdm
import os
import collections
from pathlib import Path
import time
import sys
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import data_utils
from deep_spectral_method.detection_2D_utils import UnsupBbox
from bbox_3D_estimation.detection_3D_utils import Detector3D
from bbox_3D_estimation.utils import sort_path_list
from bbox_3D_estimation.utils import read_list_poses

def predict_3D_bboxes(BboxPredictor, img_lists, poses_list, K):
    DetectorBox3D = Detector3D(K)
    for id, img_path in enumerate(img_lists):
        if id%50==0 or id==0:
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"\nprocessing id:{id}")
            bbox_orig_res = BboxPredictor.infer_2d_bbox(image=image, K=K)

            poses = read_list_poses([poses_list[id]])
            DetectorBox3D.add_view(bbox_orig_res, poses)
                
    DetectorBox3D.detect_3D_box()
    print(f"\nSaving... in {data_root}")
    DetectorBox3D.save_3D_box(data_root)
    print(f"\nSaved")

if __name__ == "__main__":
    data_root = os.getcwd() + "/data/onepose_datasets/test_moccona"
    feature_dir = data_root + "/DSM_features"
    segment_dir = data_root + "/test_moccona-annotate"
    intriscs_path = segment_dir + "/intrinsics.txt"

    BboxPredictor = UnsupBbox(feature_dir=feature_dir)
    K, _ = data_utils.get_K(intriscs_path)
    
    poses_list = glob.glob(os.path.join(os.getcwd(), f"{segment_dir}/poses", "*.txt"))
    poses_list = sort_path_list(poses_list)
    img_lists = glob.glob(os.path.join(os.getcwd(), f"{segment_dir}/color_full", "*.png"))
    img_lists = sort_path_list(img_lists)

    predict_3D_bboxes(BboxPredictor, img_lists, poses_list, K)
