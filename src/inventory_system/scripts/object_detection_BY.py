#!/usr/bin/env python
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
import time
import torch
import numpy as np
import threading
from ultralytics import YOLO

class InventorySystem:
    def __init__(self):
        #initialise the ROS node
        rospy.init_node('inventory_system', anonymous=True)


        #retrieves the yolov8 model uses yolovm
        model_path = rospy.get_param('~model_path', "yolov8m.pt")

        #initialise the device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rospy.loginfo(f"Using device: {self.device}")

        #laods model
        try:
            self.model = YOLO(model_path).to(self.device)
            rospy.loginfo("YOLOv8 model loaded successfully.")

        except Exception as e:
            rospy.logerr(f"Failed to load YOLOv8 model: {e}")
            rospy.signal_shutdown("YOLOv8 model loading failed.")
            return

        #initialise cvbridge for video capturinv
        self.bridge = CvBridge()

        #subscribe to the ROS topic to grab the compressed images
        self.subscriber = rospy.Subscriber('/video_frames', CompressedImage, self.image_callback, queue_size=10)

        #variables for fpscalculation
        self.start_time = time.time()
        self.frame_count = 0

        #shared frame variable and lock for thread safety
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        #initialize tracking variables
        self.max_age = 30  # maxnum of frames to keep lost tracks
        self.min_hits = 3  #maxnum of frames to consider a track as confirmed

        #defines target classes relevant to our lab/warehouse environments
        self.target_classes = [
            0, 1, 2, 3, 25, 26, 27, 31, 32, 33, 56, 62, 58,
            63, 64, 65, 66, 67, 68, 73, 74, 75, 76, 77, 79

        ]

        #map class id to names (from YOLO model)
        self.class_names = self.model.names

        #inventory counts
        self.inventory_detected_items = {}  # {class_name: set of track IDs}

        #confidence threshold
        self.confidence_threshold = 0.55  # Filter out low-confidence detections

        #tracker type NTOE:bytetrack is default in ultralytics YOLOv8
        self.tracker = 'bytetrack'

        #register shutdown hook
        rospy.on_shutdown(self.shutdown_node)

    def image_callback(self, msg):
        try:
            #convert ROS CompressedImage message to OpenCV format
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        #check if frame is valid
        if frame is None:
            rospy.logerr("Received an empty frame.")
            return

        #  resise the frame to reduce memory usage
        frame = cv2.resize(frame, (640, 480)) 

        try:
            with torch.no_grad():
                #perform object detection and tracking with YOLOv8
                results = self.model.track(
                    source=frame,
                    conf=self.confidence_threshold,
                    persist=True,
                    classes=self.target_classes,
                    device=self.device,
                    tracker=self.tracker
                )
        except Exception as e:
            rospy.logerr(f"YOLOv8 Inference Error: {e}")
            return

        #update frame count and calculate FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

        # extract detections and track information
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                # access class IDs, confidences, bounding boxes, and tracking IDs
                class_ids = boxes.cls.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy().astype(float)
                track_ids = boxes.id.cpu().numpy().astype(int)

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = xyxy[i]
                    cls_id = class_ids[i]
                    score = confidences[i]
                    track_id = track_ids[i]

                    # print detection info for debugging
                    rospy.loginfo(f"Detection: Class ID={cls_id}, Class Name={self.class_names.get(cls_id, 'Unknown')}, Score={score}, Track ID={track_id}")

                    # filter out low-confidence detections
                    if score < self.confidence_threshold:
                        continue

                    if cls_id not in self.target_classes:
                        continue  

                    class_name = self.class_names.get(cls_id, 'Unknown')

                    #ensure bounding box is within frame boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)

                    #draw to vid
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {track_id} - {class_name}",
                                (int(x1), max(0, int(y1) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    #updates inventory counts
                    if class_name not in self.inventory_detected_items:
                        self.inventory_detected_items[class_name] = set()
                    self.inventory_detected_items[class_name].add(track_id)

            else:
                rospy.logdebug("No detections in this frame.")

        #displays inventory counts on the frame
        y_offset = 20
        for idx, (class_name, IDs) in enumerate(self.inventory_detected_items.items()):
            text = f"{class_name}: {len(IDs)} unique objects"
            cv2.putText(frame, text, (10, y_offset + idx * 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

        #updates the latest frame for display in a thread-safe manner
        with self.frame_lock:
            self.latest_frame = frame.copy()

        #log FPS
        rospy.loginfo(f"Current FPS: {fps:.2f}")

    def shutdown_node(self):
        rospy.loginfo("\nFinal Inventory of Detected Items:")
        if not self.inventory_detected_items:
            rospy.loginfo("No items detected.")
        else:
            for class_name, IDs in self.inventory_detected_items.items():
                rospy.loginfo(f"{class_name}: {len(IDs)} unique objects")
        #destroy OpenCV windows
        cv2.destroyAllWindows()

def main():
    ic = InventorySystem()
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        with ic.frame_lock:
            if ic.latest_frame is not None:
                frame_to_display = ic.latest_frame.copy()
            else:
                frame_to_display = None
        if frame_to_display is not None:
            cv2.imshow("YOLOv8 with ByteTracker", frame_to_display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rospy.loginfo("Detected 'q' key press. Initiating shutdown.")
                rospy.signal_shutdown("Shutdown requested via GUI.")
                break
        rate.sleep()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
