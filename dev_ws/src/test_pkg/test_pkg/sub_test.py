import rclpy
from rclpy.node import Node

from test_msgs.msg import BoundingBox3D


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscriber_ = self.create_subscription(
            msg_type=BoundingBox3D, topic="/topic", callback=self.sub_callback, qos_profile=10)

    def sub_callback(self, msg: BoundingBox3D):
        self.get_logger().info(message=str(msg))


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalPublisher()

    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
