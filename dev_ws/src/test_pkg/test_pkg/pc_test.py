import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data
from test_msgs.msg import BoundingBoxStamped, PosedImageStamped

SUPPORTED_FORMATS = ["BGR", "RGB", "ARGB"]


def deserialize_image_msg(msg: PosedImageStamped):

    if str(msg.encoding).upper() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unknown image encoding. Supported encodings: {SUPPORTED_FORMATS}"
        )

    pose = msg.pose
    intrinsics = msg.intrinsics.reshape(3, 3)
    image = np.array(msg.data).reshape(msg.height, msg.width, msg.step)

    if str(msg.encoding).upper() == "BGR":
        image = image[..., ::-1]

    return image, intrinsics, pose


def process_image(image, intrinsics, pose) -> BoundingBoxStamped:
    result = BoundingBoxStamped()

    result.pose = pose
    result.height, result.width = image.shape[0]/100, image.shape[1]/100
    result.length = 1 + intrinsics[0][0]

    return result


class MinimalServer(Node):

    def __init__(self):
        super().__init__('minimal_server')
        self.get_logger().info("Started server")

        # self.qos_profile = QoSProfile(
        #     reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
        #     history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
        #     depth=1,
        # )

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

    def image_callback(self, msg: PosedImageStamped):
        time = self.get_clock().now().to_msg()
        self.get_logger().info(f"received image at {time}")
        image, intrinsics, pose = deserialize_image_msg(msg)
        res_bbox = process_image(image, intrinsics, pose)
        res_bbox.header = msg.header
        self.bbox_publisher_.publish(res_bbox)


def main(args=None):
    rclpy.init(args=args)

    minimal_server = MinimalServer()

    rclpy.spin(minimal_server)

    minimal_server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
