#!/usr/bin/env python
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO
import time
import torch
import threading
import numpy as np


from deep_sort_realtime.deepsort_tracker import DeepSort

class InventorySystem:
    def __init__(self):
        #initialize the ROS node
        rospy.init_node('inventory_system', anonymous=True)
        
        #retrieve model path from ROS parameter or use default
        model_path = rospy.get_param('~model_path', "/home/rose/ros_ws/src/inventory_system/models/yolov8m.pt")
        
        #load the YOLOv8 model
        if torch.cuda.is_available():
            self.model = YOLO(model_path).to('cuda')  #use GPU if available
            rospy.loginfo("YOLOv8 model loaded on GPU.")
        else:
            self.model = YOLO(model_path).to('cpu')  #fallback to CPU
            rospy.loginfo("YOLOv8 model loaded on CPU.")
        
        #initialize CvBridge
        self.bridge = CvBridge()
        
        #subscriber to the ROS topic publishing compressed images
        self.subscriber = rospy.Subscriber('/video_frames', CompressedImage, self.image_callback, queue_size=10)
        
        #dictionary to hold the inventory of detected items
        self.inventory_detected_items = {}
        
        #Variables for FPS calculation
        self.start_time = time.time()
        self.frame_count = 0
        
        #shared frame variable and lock for thread safety
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        #flag to control the display thread
        self.display_active = True
        
        #start the display thread
        self.display_thread = threading.Thread(target=self.display_frames)
        self.display_thread.start()
        
        #initialize DeepSort
        self.tracker = DeepSort(max_age=1000)
        rospy.loginfo("Deep SORT tracker initialized.")
        
        #initialize last_processed_time if needed (commented out if not used)
        #self.last_processed_time = time.time()
        
        #register shutdown hook
        rospy.on_shutdown(self.shutdown_node)

    def image_callback(self, msg):
        try:
            #convert ROS CompressedImage message to OpenCV format
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            #perform object detection
            results = self.model(frame)
            
            #prepare detection data for DeepSort
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls)
                        confidence = float(box.conf)
                        #corrected line for bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        #deepSort expects bbox in format [x, y, w, h]
                        w = x2 - x1
                        h = y2 - y1
                        detections.append(([x1, y1, w, h], confidence, class_id))
            
            #update the tracker
            tracks = self.tracker.update_tracks(detections, frame=frame)
            
            #draw tracks
            if tracks:
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)
                    class_id = track.det_class
                    class_name = self.model.names.get(class_id, 'Unknown')
                    
                    #update inventory
                    if class_name not in self.inventory_detected_items:
                        self.inventory_detected_items[class_name] = set()
                    self.inventory_detected_items[class_name].add(track_id)
                    
                    #draw bounding box and tracking ID
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {track_id} - {class_name}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                rospy.loginfo("No tracks output by DeepSort.")
            
            #update the latest frame for display in a thread-safe manner
            with self.frame_lock:
                self.latest_frame = frame

            #update frame count and calculate FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            rospy.loginfo(f"Current FPS: {fps:.2f}")

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except AttributeError as e:
            rospy.logerr(f"Attribute Error during image processing: {e}")
        except TypeError as e:
            rospy.logerr(f"Type Error during image processing: {e}")
        except Exception as e:
            rospy.logerr(f"Error during image processing: {e}")
    
    def display_frames(self):
        """Thread function to display frames using OpenCV."""
        rospy.loginfo("Display thread started.")
        while not rospy.is_shutdown() and self.display_active:
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_to_display = self.latest_frame.copy()
                else:
                    frame_to_display = None
            if frame_to_display is not None:
                cv2.imshow("YOLOv8 with DeepSort Tracking", frame_to_display)
            #handle GUI events and check for 'q' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rospy.loginfo("Detected 'q' key press. Initiating shutdown.")
                self.shutdown_node()
                break
        rospy.loginfo("Display thread exiting.")
    
    def shutdown_node(self):
        """Shutdown the ROS node, printing inventory and closing any GUI elements."""
        if not self.display_active:
            return  #already shutting down
        self.display_active = False  #signal the display thread to stop
        
        rospy.loginfo("\nFinal Inventory of Detected Items:")
        for class_name, IDs in self.inventory_detected_items.items():
            rospy.loginfo(f"{class_name}: {len(IDs)} unique objects")
        
        #destroy OpenCV windows
        cv2.destroyAllWindows()
        
        #signal ROS to shutdown
        rospy.signal_shutdown("Shutdown requested via GUI.")
    
    def join_threads(self):
        """Wait for the display thread to finish."""
        self.display_thread.join()

def main():
    ic = InventorySystem()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("KeyboardInterrupt received. Shutting down.")
    finally:
        ic.shutdown_node()
        ic.join_threads()

if __name__ == '__main__':
    main()
