import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import rosbag2_py
import numpy as np
import cv2
import os

class ROSBag_Publisher(Node):
    def __init__(self, bag_path):
        super().__init__("rosbag_publisher")
        self.publisher_ = self.create_publisher(CompressedImage, "/video_frames", 10)
        self.bridge = CvBridge()

        # Initialize ROS2 SequentialReader to read the bag
        self.reader = rosbag2_py.SequentialReader()

        # Specify the path to your ROS2 bag and the storage format
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"
        )
        self.reader.open(storage_options, converter_options)

        self.timer_period = 0.1  # Adjust timer period for frame publishing
        self.timer = self.create_timer(self.timer_period, self.publish_frame)

    def publish_frame(self):
        if self.reader.has_next():
            (topic, data, timestamp) = self.reader.read_next()

            if topic == "/camera/front/compressed_image":
                try:
                    # Check if data is already deserialized or in raw byte format
                    if hasattr(data, 'format') and hasattr(data, 'data'):
                        # If 'format' and 'data' attributes exist, it's likely a CompressedImage
                        img_format = data.format  # Could be "jpeg", "png", etc.
                        byte_data = data.data
                        self.get_logger().info(f"Image format: {img_format}")
                    else:
                        # Otherwise, treat 'data' as raw bytes
                        byte_data = data
                        img_format = 'jpeg'  # Defaulting to jpeg if we can't detect the format

                    # Convert byte array to NumPy array
                    np_arr = np.frombuffer(byte_data, np.uint8)

                    # Decode compressed image to OpenCV format based on format
                    if np_arr.size > 0:  # Ensure the array is not empty
                        if img_format == 'jpeg':
                            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        elif img_format == 'png':
                            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                        else:
                            raise ValueError(f"Unsupported image format: {img_format}")

                        if cv_image is None:
                            raise ValueError("cv2.imdecode failed to decode image.")

                        # Convert back to ROS2 CompressedImage format
                        msg = self.bridge.cv2_to_compressed_imgmsg(cv_image, img_format)
                        self.publisher_.publish(msg)
                    else:
                        raise ValueError("Received empty NumPy array from buffer.")

                except Exception as e:
                    self.get_logger().error(f"Error processing frame: {e}")
            else:
                self.get_logger().warn(f"Unknown topic: {topic}")
        else:
            self.get_logger().info("End of bag reached.")
            self.timer.cancel()  # Stops the timer once the bag is fully read
            self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    # Update this path to your actual bag file name
    bag_path = os.path.join(os.path.dirname(__file__), 'FrontFisheyeRectified20Hz.db3')
    node = ROSBag_Publisher(bag_path)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
