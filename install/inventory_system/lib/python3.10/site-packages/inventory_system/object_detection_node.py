#!/usr/bin/env python3
import cv2
import time
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from deep_sort_realtime.deepsort_tracker import DeepSort

class Inventory_System(Node):

    def __init__(self):
        super().__init__('inventory_system')
        self.subscription = self.create_subscription(Image, '/video_frames', self.image_callback, 10)

        # Initialize YOLOv8 model, which over here is a pre-trained nano version
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()

        # Initialize DeepSORT tracker
        self.tracker = DeepSort(max_age=30, n_init=3)

        # Inventory system
        self.detected_items = {}
        self.frame_count = 0
        self.start_time = time.time()

        # Log active topics
        self.log_active_topics()

    def image_callback(self, msg):
        # Conversion of ROS to OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS Image to OpenCV: {e}")
            return

        # YOLOv8 Detection
        results = self.model(frame)

        # Prepare detections for DeepSORT
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)

                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                width = x2 - x1
                height = y2 - y1

                detections.append([[x1, y1, width, height], confidence, class_id])

        # Update DeepSORT tracker
        if detections:
            tracked_objects = self.tracker.update_tracks(detections, frame=frame)

            for track in tracked_objects:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                class_id = track.det_class

                # Update inventory system
                class_name = self.model.names[class_id]
                if class_name not in self.detected_items:
                    self.detected_items[class_name] = set()
                self.detected_items[class_name].add(track_id)

                # Draw tracking info
                cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id} - {class_name}", (int(ltrb[0]), int(ltrb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Show the video with detections and tracking
        cv2.imshow("YOLOv8 with DeepSORT Tracking", frame)

        # FPS counter
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time
            self.get_logger().info(f"FPS: {fps}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.print_inventory()
            rclpy.shutdown()

    def print_inventory(self):
        # Print the final inventory
        self.get_logger().info("\nFinal List of Detected Items:")
        for class_name, track_ids in self.detected_items.items():
            self.get_logger().info(f"{class_name}: {len(track_ids)} unique objects")

    def log_active_topics(self):
        # Log the active topics for debugging
        active_topics = self.get_node_topics_interface().get_publisher_names_and_types_by_node(self.get_name(), self.get_namespace())
        self.get_logger().info(f"Active topics: {active_topics}")

def main(args=None):
    rclpy.init(args=args)
    node = Inventory_System()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.print_inventory()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
