import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Header
from rclpy.qos import qos_profile_sensor_data
from test_msgs.msg import BoundingBoxStamped, PosedImageStamped


class MinimalClient(Node):

    def __init__(self):
        super().__init__('images_publisher')
        self.get_logger().info("Started client")

        self.image_publisher_ = self.create_publisher(
            PosedImageStamped, "/posed_images", qos_profile_sensor_data)

        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.image = cv2.imread("../resource/test_img.jpg")

    def timer_callback(self):

        header = Header()
        header.frame_id = str(np.random.randint(0, 5))
        header.stamp = self.get_clock().now().to_msg()

        msg = PosedImageStamped()
        msg.header = header
        msg.encoding = "BGR"
        msg.height, msg.width, msg.step = self.image.shape
        msg.pose.position.x = 0.0
        msg.pose.position.y = 0.42
        msg.pose.position.z = 5 * np.random.random()
        msg.data = [int(i) for i in self.image.reshape(self.image.size)]

        self.get_logger().info(f'Publishing: {self.i}')
        self.image_publisher_.publish(msg)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()
    rclpy.spin(minimal_client)

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
