import cv2
import torch
import rclpy
import torchvision
import numpy as np
from rclpy.node import Node
from resource.src.sfm.extract_features import confs
from resource.src.utils.model_io import load_network
from resource.src.utils.eval_utils import ransac_PnP
from resource.src.utils.data_utils import get_K_crop_resize, get_image_crop_resize, pad_features3d_random, build_features3d_leaves
from test_msgs.msg import BoundingBoxStamped, PosedImageStamped
from resource.deep_spectral_method.detection_2D_utils import UnsupBbox
from resource.src.models.GATsSPG_lightning_model import LitModelGATsSPG
from resource.src.models.extractors.SuperPoint.superpoint import SuperPoint

SUPPORTED_FORMATS = ["BGR", "RGB", "RGBA", "BGRA"]

def draw_3d_box(image, corners_2d, linewidth=3, color="g"):
    """Draw 3d box corners
    @param corners_2d: [8, 2]
    """

    lines = np.array(
        [
            [0, 1, 5, 4, 2, 3, 7, 6, 0, 1, 4, 5],
            [1, 5, 4, 0, 3, 7, 6, 2, 2, 3, 6, 7],
        ]
    ).T

    colors = {"g": (0, 255, 0), "r": (0, 0, 255), "b": (255, 0, 0)}
    if color not in colors.keys():
        color = (42, 97, 247)
    else:
        color = colors[color]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for id, line in enumerate(lines):
        pt1 = corners_2d[line[0]].astype(int)
        pt2 = corners_2d[line[1]].astype(int)
        cv2.line(image, tuple(pt1), tuple(pt2), color, linewidth)

    return image

def reproj(K, pose, pts_3d):
    """
    Reproj 3d points to 2d points
    @param K: [3, 3] or [3, 4]
    @param pose: [3, 4] or [4, 4]
    @param pts_3d: [n, 3]
    """
    assert K.shape == (3, 3) or K.shape == (3, 4)
    assert pose.shape == (3, 4) or pose.shape == (4, 4)

    if K.shape == (3, 3):
        K_homo = np.concatenate([K, np.zeros((3, 1))], axis=1)
    else:
        K_homo = K

    if pose.shape == (3, 4):
        pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
    else:
        pose_homo = pose

    pts_3d = pts_3d.reshape(-1, 3)
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1)
    pts_3d_homo = pts_3d_homo.T

    reproj_points = K_homo @ pose_homo @ pts_3d_homo
    reproj_points = reproj_points[:] / reproj_points[2:]
    reproj_points = reproj_points[:2, :].T
    return reproj_points  # [n, 2]

def save_demo_image(
    pose_pred, K, image, box3d, draw_box=True
):
    """
    Project 3D bbox by predicted pose and visualize
    """
    if draw_box:
        reproj_box_2d = reproj(K, pose_pred, box3d)
        image = draw_3d_box(image, reproj_box_2d, color="b", linewidth=10)

    return image

def deserialize_image_msg(msg: PosedImageStamped):

    if str(msg.encoding).upper() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unknown image encoding. Supported encodings: {SUPPORTED_FORMATS}"
        )

    pose = msg.pose
    intrinsics = msg.intrinsics.reshape(3, 3)
    image = np.array(msg.data).reshape(msg.height, msg.width, msg.step)
    if str(msg.encoding).upper() == "RGBA":
        image = image[..., :3]
    elif str(msg.encoding).upper() == "BGRA":
        image = image[..., :3]
        msg.encoding = "BGR"
    if str(msg.encoding).upper() == "BGR":
        image = image[..., ::-1]
    

    return image, intrinsics, pose

def load_model(cfg):
    """Load model"""

    def load_matching_model(model_path):
        """Load onepose model"""

        trained_model = LitModelGATsSPG.load_from_checkpoint(
            checkpoint_path=model_path)
        trained_model.to(cfg["device"])
        trained_model.eval()

        return trained_model

    def load_extractor_model(model_path):
        """Load extractor model(SuperPoint)"""

        extractor_model = SuperPoint(confs["superpoint"]["conf"])
        extractor_model.to(cfg["device"])
        extractor_model.eval()
        load_network(extractor_model, model_path, force=True)

        return extractor_model

    matching_model = load_matching_model(cfg["onepose_model_path"])
    extractor_model = load_extractor_model(cfg["superpoint_path"])
    return matching_model, extractor_model

def crop_img_by_bbox(image, bbox, K=None, crop_size=512):
    """
    Crop image by detect bbox
    Input:
        query_img_path: str,
        bbox: np.ndarray[x0, y0, x1, y1],
        K[optional]: 3*3
    Output:
        image_crop: np.ndarray[crop_size * crop_size],
        K_crop[optional]: 3*3
    """
    x0, y0 = bbox[0], bbox[1]
    x1, y1 = bbox[2], bbox[3]
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    resize_shape = np.array([int(y1 - y0), int(x1 - x0)])
    if K is not None:
        K_crop, _ = get_K_crop_resize(bbox, K, resize_shape)
    image_crop, _ = get_image_crop_resize(gray_img, bbox, resize_shape)

    bbox_new = np.array([0, 0, x1 - x0, y1 - y0])
    resize_shape = np.array([crop_size, crop_size])
    if K is not None:
        K_crop, _ = get_K_crop_resize(bbox_new, K_crop, resize_shape)
    image_crop, _ = get_image_crop_resize(image_crop, bbox_new, resize_shape)

    return image_crop, K_crop if K is not None else None

class MinimalServer(Node):

    def __init__(self):
        super().__init__('minimal_server')
        self.get_logger().info("Started server")

        self.image_subscriber_ = self.create_subscription(
            msg_type=PosedImageStamped,
            topic="/posed_images",
            callback=self.image_callback,
            qos_profile=10,
        )

        self.bbox_publisher_ = self.create_publisher(
            msg_type=BoundingBoxStamped,
            topic="/bounding_boxes",
            qos_profile=10,
        )

        self.detector = UnsupBbox()
        self.data_dir = "/home/ale/ZeroShotPoseEstimation/data"

        self.cfg = {
            "num_leaf": 8,
            "device": "cuda" if torch.cuda.is_available() else "cpu", 
            "onepose_model_path": f"{self.data_dir}/models/checkpoints/onepose/GATsSPG.ckpt",
            "superpoint_path": f"{self.data_dir}/models/extractors/SuperPoint/superpoint_v1.pth",
            "avg_anno_path": f"{self.data_dir}/sfm_model/demo/outputs_superpoint_superglue/anno/anno_3d_average.npz",
            "clt_anno_path": f"{self.data_dir}/sfm_model/demo/outputs_superpoint_superglue/anno/anno_3d_collect.npz",
            "idxs_path": f"{self.data_dir}/sfm_model/demo/outputs_superpoint_superglue/anno/idxs.npy",
            "box3d_path": f"{self.data_dir}/sfm_model/demo/box3d_corners.txt"
        }

        self.matching_model, self.extractor_model = load_model(self.cfg)
        self.onepose_input = self.onepose_input_setup()
        self.box3d = np.loadtxt(self.cfg["box3d_path"])

    def image_callback(self, msg: PosedImageStamped):
        time = self.get_clock().now().to_msg()
        self.get_logger().info(f"received image at {time}")
        image, intrinsics, pose = deserialize_image_msg(msg)
        res_bbox = self.process_image(image, intrinsics, pose)
        res_bbox.header = msg.header
        self.bbox_publisher_.publish(res_bbox)

    def process_image(self, image, intrinsics, pose) -> BoundingBoxStamped:
        with torch.no_grad():
            bbox = self.detector.infer_2d_bbox(image=image, K=2, viz=True)
            image_cropped, K_crop = crop_img_by_bbox(image=image, bbox=bbox, K=intrinsics)

            image_cropped = torchvision.transforms.functional.to_tensor(
                image_cropped
            ).unsqueeze(0)
            image_cropped = image_cropped.to(self.cfg["device"])

            pred_detection = self.extractor_model(image_cropped)
            pred_detection = {k: v[0].cpu().numpy() for k, v in pred_detection.items()}
            keypoints2d = torch.Tensor(pred_detection["keypoints"])
            descriptors2d = torch.Tensor(pred_detection["descriptors"])

            self.onepose_input["keypoints2d"] = keypoints2d[None].to(self.cfg["device"])
            self.onepose_input["descriptors2d_query"] = descriptors2d[None].to(self.cfg["device"])
            
            pred, _ = self.matching_model(self.onepose_input)
            matches = pred["matches0"].detach().cpu().numpy()
            valid = matches > -1
            kpts3d = self.onepose_input["keypoints3d"][0].detach().cpu().numpy()
            kpts2d = pred_detection["keypoints"]
            confidence = pred["matching_scores0"].detach().cpu().numpy()
            mkpts2d, mkpts3d, _ = (
                    kpts2d[valid],
                    kpts3d[matches[valid]],
                    confidence[valid],
                )

            _, pose_pred_homo, inliers = ransac_PnP(
                    K_crop, mkpts2d, mkpts3d, scale=1000
                )

        demo_img = save_demo_image (
            box3d=self.box3d,
            pose_pred=pose_pred_homo,
            K=intrinsics,
            image=image,
            draw_box=len(inliers) > 3
        )

        result = BoundingBoxStamped()
        result.pose.position.x = bbox[0]
        result.pose.position.y = bbox[1]
        result.height, result.width = bbox[3]-bbox[1], bbox[2]-bbox[0]
        result.length = 0.0

        return result

    def onepose_input_setup(self):
        # Prepare 3D features:
        num_leaf = self.cfg["num_leaf"]
        avg_data = np.load(self.cfg["avg_anno_path"])
        clt_data = np.load(self.cfg["clt_anno_path"])
        idxs = np.load(self.cfg["idxs_path"])

        keypoints3d = torch.Tensor(clt_data["keypoints3d"]).to(self.cfg["device"])
        num_3d = keypoints3d.shape[0]
        # load average 3D features:
        avg_descriptors3d, _ = pad_features3d_random(
            avg_data["descriptors3d"], avg_data["scores3d"], num_3d
        )
        # load corresponding 2D features of each 3D point:
        clt_descriptors, _ = build_features3d_leaves(
            clt_data["descriptors3d"], clt_data["scores3d"], idxs, num_3d, num_leaf
        )

        inp_data = {
        "keypoints3d": keypoints3d[None].to(self.cfg["device"]),  # [1, n2, 3]
        "descriptors3d_db": avg_descriptors3d[None].to(self.cfg["device"]),  # [1, dim, n2]
        "descriptors2d_db": clt_descriptors[None].to(self.cfg["device"]),
        "image_size": torch.Tensor([[720, 1280]]),
        }

        return inp_data

def main(args=None):
    rclpy.init(args=args)

    minimal_server = MinimalServer()

    rclpy.spin(minimal_server)

    minimal_server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
