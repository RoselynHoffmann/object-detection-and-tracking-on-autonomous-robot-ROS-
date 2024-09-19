#!/usr/bin/env python3
import cv2
import time
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from deep_sort_realtime.deepsort_tracker import DeepSort

class Inventory_System(Node):

    def __init__(self):
        super().__init__('inventory_system')
        #subscribing to video frames topic
        self.subscription = self.create_subscription(Image, '/video_frames', self.image_callback, 10)

        #initialize the YOLOv8 model, using a pre-trained 'nano' version
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()

        # deep sort tracker init. `max_age` is how long it remembers lost objects 
        #`n_init` needs this many detections to start tracking, set to default
        self.tracker = DeepSort(max_age=30, n_init=3)

        # keeping a list of items
        self.detected_items = []

        #inventory system's count of detected items
        self.inventory_detected_items = {}

        #FPS related stuff
        self.frame_count = 0
        self.start_time = time.time()

    def image_callback(self, msg):
        #vonverting ROS image to OpenCV format (from ros msg to cv2 image)
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        #increase frame count and calculate current fps
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            current_fps = self.frame_count / elapsed_time
            print(f"fps now: {current_fps:.2f}")

        #yolo detection on the current frame
        results = self.model(frame)

        # DeepSORT tracking setup: format detections properly 
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)  # confidence needs to be a float
                
                # getting box coordinates and converting to int
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                width = x2 - x1
                height = y2 - y1

                #appending detections in format for DeepSORT
                detections.append([[x1, y1, width, height], confidence, class_id])

        #process detections with DeepSORT, if any detections exist
        if detections:
            tracked_objects = self.tracker.update_tracks(detections, frame=frame)

            if tracked_objects:
                print(f"Tracker active, {len(tracked_objects)} objects tracked")
            else:
                print("no objects are being tracked...")

            #drawing boxes around tracked objects and assigning IDs
            for track in tracked_objects:
                if not track.is_confirmed():  # if object is confirmed
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()  # bounding box (left top right bottom)
                class_id = track.det_class  #detected class ID

                class_name = self.model.names[class_id]
                
                #updating the inventory with tracked objects
                if class_name not in self.inventory_detected_items:
                    self.inventory_detected_items[class_name] = set()
                if track_id not in self.inventory_detected_items[class_name]:
                    self.inventory_detected_items[class_name].add(track_id)

                # drawing bounding boxes and displaying track ID
                cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id} - {class_name}", (int(ltrb[0]), int(ltrb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        #show the video with the tracked objects
        cv2.imshow("YOLOv8 with DeepSORT Tracking", frame)

        #quit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = Inventory_System()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()  #cleanly close the window
        rclpy.shutdown()
    finally:
        #rint the final list of detected items
        print("\nFinal List of Detected Items:")
        for item in node.detected_items:
            print(item)

        #print the inventory system with object counts
        print("\nInventory System:")
        for class_name, IDs in node.inventory_detected_items.items():
            print(f"{class_name}: {len(IDs)} unique objects")

if __name__ == '__main__':
    main()
