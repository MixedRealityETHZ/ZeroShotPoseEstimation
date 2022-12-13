import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Header
from test_msgs.msg import PosedImageStamped
from scipy.spatial.transform import Rotation as Rotmat
from resource.src.utils.data_utils import get_K

class MinimalClient(Node):

    def __init__(self):
        super().__init__('minimal_client')
        self.get_logger().info("Started client")

        self.image_publisher_ = self.create_publisher(
            PosedImageStamped, "/posed_images", 10)

        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.image_index = 0
        self.data_root = "/home/ale/ZeroShotPoseEstimation/dev_ws/src/dsmpose/resource/data/onepose_datasets/val_data/0606-tiger-others/tiger-2"
        self.intrinsics, _ = get_K(f"{self.data_root}/intrinsics.txt")

    def timer_callback(self):
        image_path = f"{self.data_root}/color_full/{self.image_index}.png"
        pose_path = f"{self.data_root}/poses/{self.image_index}.txt"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        # image = image[180:1261, :, :]
        # image = cv2.resize(image, (1280, 720))
        pose = np.loadtxt(pose_path)
        quat = Rotmat.from_matrix(pose[0:3, 0:3]).as_quat()
        header = Header()
        header.frame_id = str(0)
        header.stamp = self.get_clock().now().to_msg()

        msg = PosedImageStamped()
        msg.header = header
        msg.encoding = "BGRA"
        msg.height, msg.width, msg.step = image.shape
        msg.pose.position.x = pose[0, 3]
        msg.pose.position.y = pose[1, 3]
        msg.pose.position.z = pose[2, 3]
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]

        msg.intrinsics = self.intrinsics.reshape(self.intrinsics.size)

        msg.data = [int(i) for i in image.reshape(image.size)]

        self.get_logger().info(f'Publishing: {self.image_index}')
        self.image_publisher_.publish(msg)
        self.image_index += 1


def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()
    rclpy.spin(minimal_client)

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
