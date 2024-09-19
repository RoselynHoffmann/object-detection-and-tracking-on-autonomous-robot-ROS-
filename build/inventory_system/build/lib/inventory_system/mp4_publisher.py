import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import os
import time

class MP4_Publisher(Node):

    def __init__(self, video_path):
        super().__init__('mp4_publisher_node')

        # Initialize video capture from mp4 file
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open video file: {video_path}")
            return

        # Frame publisher
        self.publisher_ = self.create_publisher(Image, '/video_frames', 10)

        # Timer callback to publish frames at correct FPS
        self.timer_period = 1.0 / self.cap.get(cv2.CAP_PROP_FPS)
        self.timer = self.create_timer(self.timer_period, self.publish_frame)

        # FPS counter and logger setup
        self.frame_count = 0
        self.start_time = time.time()

        # Log active topics
        self.log_active_topics()

    def publish_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().info("End of video file reached.")
            self.timer.cancel()  # Stop the timer when the video ends
            return

        # Convert the frame to a ROS Image message
        msg = Image()
        msg.height, msg.width, _ = frame.shape
        msg.encoding = "bgr8"
        msg.data = frame.tobytes()

        # Publish the frame
        self.publisher_.publish(msg)

        # Log the published frame size
        self.get_logger().info(f"Published frame of size {len(msg.data)} bytes")

        # FPS counter
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time
            self.get_logger().info(f"FPS: {fps}")

    def log_active_topics(self):
        # Log the active topics for debugging
        active_topics = self.get_topic_names_and_types()
        self.get_logger().info(f"Active topics: {active_topics}")


def main(args=None):
    rclpy.init(args=args)

    # Set the path to your mp4 file
    video_path = os.path.join(os.path.dirname(__file__), 'FrontFisheyeRectified20Hz.mp4')
    node = MP4_Publisher(video_path)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
