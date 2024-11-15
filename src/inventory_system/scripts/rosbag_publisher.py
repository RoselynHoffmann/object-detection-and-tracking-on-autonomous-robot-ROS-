#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import rosbag
import numpy as np
import cv2
import os

class ROSBagPublisher:
    def __init__(self, bag_path, publish_topic='/video_frames', source_topic='/camera/front/compressed_image', rate_hz=5):
        rospy.init_node('rosbag_publisher', anonymous=True)
        self.publisher = rospy.Publisher(publish_topic, CompressedImage, queue_size=10)
        self.bridge = CvBridge()
        self.rate_hz = rate_hz
        self.publish_interval = 1.0 / self.rate_hz

        #attempt to open therosbag
        if not os.path.exists(bag_path):
            rospy.logfatal("ROS bag file does not exist at path: {}".format(bag_path))
            rospy.signal_shutdown("Invalid bag path.")
            return

        try:
            self.bag = rosbag.Bag(bag_path, 'r')
            rospy.loginfo("Opened ROS bag file: {}".format(bag_path))
        except Exception as e:
            rospy.logfatal("Could not open ROS bag file: {}".format(e))
            rospy.signal_shutdown("Failed to open bag file.")
            return

        #get iterator
        self.bag_iterator = self.bag.read_messages(topics=[source_topic])

        #timer for publishing speed
        self.timer = rospy.Timer(rospy.Duration(self.publish_interval), self.publish_frame)

    def publish_frame(self, event):
        try:
            #next msg
            topic, msg, t = next(self.bag_iterator)
            
            #decode img
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR if msg.format.lower().endswith('jpeg') else cv2.IMREAD_UNCHANGED)
            
            if cv_image is not None:
                #reenconde
                encoded_image = cv2.imencode('.jpg', cv_image)[1].tobytes()
                compressed_msg = self.bridge.cv2_to_compressed_imgmsg(cv_image, dst_format=msg.format)
                self.publisher.publish(compressed_msg)
                rospy.logdebug("Published a frame at time: {}".format(t.to_sec()))
            else:
                rospy.logwarn("Failed to decode image at time: {}".format(t.to_sec()))
        except StopIteration:
            rospy.loginfo("Reached end of ROS bag.")
            self.shutdown_node()
        except Exception as e:
            rospy.logerr("Failed to process a frame: {}".format(e))

    def on_shutdown(self):
        rospy.loginfo("Shutting down ROSBagPublisher node.")
        if hasattr(self, 'bag') and self.bag:
            self.bag.close()
            rospy.loginfo("Closed ROS bag.")
        rospy.loginfo("ROSBagPublisher node shutdown complete.")

    def shutdown_node(self):
        """Initiate node shutdown."""
        rospy.signal_shutdown("End of ROS bag reached.")

def main():
    bag_path = os.path.expanduser('/home/rose/ros_env/src/inventory_system/FrontFisheyeRectified20Hz.bag')
    publish_topic = '/video_frames'
    source_topic = '/camera/front/compressed_image'
    desired_fps = 15  

    bag_publisher = ROSBagPublisher(bag_path=bag_path, publish_topic=publish_topic, source_topic=source_topic, rate_hz=desired_fps)
    rospy.on_shutdown(bag_publisher.on_shutdown)
    rospy.spin()

if __name__ == '__main__':
    main()
