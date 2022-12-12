import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from test_msgs.msg import BoundingBoxStamped


class MinimalClient(Node):

    def __init__(self):
        super().__init__('minimal_client')
        self.get_logger().info("Started client")

        self.bbox_publisher_ = self.create_publisher(
            BoundingBoxStamped, "/bounding_boxes", 10)

        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        bbox = BoundingBoxStamped()
        bbox.header.frame_id = "pippo"
        bbox.height = 0.1
        bbox.width = 0.1
        bbox.length = 0.1
        self.bbox_publisher_.publish(bbox)


def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()
    rclpy.spin(minimal_client)

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
