#!/usr/bin/env python3
import cv2
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os

class MP4_Publisher(Node):

    def __init__(self, video_path):
        super().__init__('mp4_publisher_node')

        #ppublisher for the video frames, we publish to the /video_frames topic
        self.publisher_ = self.create_publisher(Image, '/video_frames', 10)

        #initialize cv2 bridge for ROS <-> OpenCV conversion
        self.bridge = CvBridge()

        #open the mp4 video file
        self.cap = cv2.VideoCapture(video_path)

        # if video file doesn't open, show an error
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open video file: {video_path}")
            return

        # logging the active topics, just for fun
        self.log_active_topics()

        # Timer for reading and publishing frames from the video
        self.timer_period = 1/30  # aiming for 30 FPS
        self.timer = self.create_timer(self.timer_period, self.publish_frame)

        #fps stuff: count frames and keep track of time
        self.frame_count = 0
        self.start_time = time.time()

    def log_active_topics(self):
        #get active topics (just logging info, it's nice to know)
        active_topics = self.get_publisher_names_and_types_by_node(self.get_name(), self.get_namespace())
        self.get_logger().info(f"Active topics: {active_topics}")

    def publish_frame(self):
        # Read frame from the video file
        ret, frame = self.cap.read()

        #If video ends, shut down
        if not ret:
            self.get_logger().info("End of video file reached.")
            self.cap.release()  # close video file
            self.timer.cancel()  # stop the timer
            rclpy.shutdown()
            return

        #Increase frame count
        self.frame_count += 1

        # FPS calculation every frame (pretty much)
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            current_fps = self.frame_count / elapsed_time
            self.get_logger().info(f"FPS: {current_fps}")

        #publish the frame as a ROS message
        try:
            #converting cv2 frame (image) to ROS message
            msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.publisher_.publish(msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting frame: {e}")

def main(args=None):
    rclpy.init(args=args)

    # update this path to point to the mp4 video file
    video_path = os.path.join(os.path.dirname(__file__), 'FrontFisheyeRectified20Hz.mp4')

    #Create MP4 publisher node
    node = MP4_Publisher(video_path)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
