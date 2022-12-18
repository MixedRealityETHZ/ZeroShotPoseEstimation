import cv2
import rclpy
import numpy as np
import matplotlib.pyplot as plt

from rclpy.node import Node
from test_msgs.msg import PosedImageStamped
from inference_live import deserialize_image_msg
from scipy.spatial.transform import Rotation as Rotmat

class MinimalClient(Node):

    def __init__(self):
        super().__init__('image_saver_node')
        self.get_logger().info("Started image saver")

        self.image_subscriber_ = self.create_subscription(
            msg_type=PosedImageStamped,
            topic="/posed_images",
            callback=self.image_saver_callback,
            qos_profile=10,
        )

        self.image_index = 0
        self.data_root = "/home/ale/ZeroShotPoseEstimation/data/onepose_datasets/demo/demo_test_flu"

    def image_saver_callback(self, msg: PosedImageStamped):
        image_path = f"{self.data_root}/color_full/{self.image_index}.png"
        pose_path = f"{self.data_root}/poses/{self.image_index}.txt"
        self.image_index += 1
        image, intrinsics, pose = deserialize_image_msg(msg=msg)
        plt.imsave(fname=image_path, arr=image)
        np.savetxt(fname=pose_path, X=pose)
        print(f"{self.image_index}")


def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()
    rclpy.spin(minimal_client)

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
