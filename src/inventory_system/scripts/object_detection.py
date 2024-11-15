#!/usr/bin/env python
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
import time
import torch
import numpy as np
import threading
import torch.nn.functional as F
from torch import nn
import torchvision.ops as ops
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment  #for Hungarian Algorithm

class YOLOWithFeatureExtraction(YOLO):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.hook_handles = []
        self.layer_strides = {}
        self.register_hooks()

    def register_hooks(self):
        """
        Register forward hooks to capture feature maps from YOLOv8's C2f layers in the neck.
        Assign strides programmatically based on the cumulative stride.
        """
        rospy.loginfo("YOLOv8 Model Architecture:")
        current_stride = 1

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                stride = module.stride[0]
                if stride > 1:
                    current_stride *= stride
            elif isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d):
                stride = module.stride
                if stride > 1:
                    current_stride *= stride
            elif isinstance(module, nn.Upsample):
                scale_factor = module.scale_factor
                current_stride /= scale_factor

            #if the module is a C2f layer, register a hook
            if module.__class__.__name__ == 'C2f':
                handle = module.register_forward_hook(self.hook_fn)
                self.hook_handles.append(handle)
                rospy.loginfo(f"Registered forward hook to layer: {name} with stride {current_stride}")
                #store the stride associated with this layer
                self.layer_strides[module] = current_stride

    def hook_fn(self, module, input, output):
        """
        Forward hook function to capture feature maps.
        Stores features along with their corresponding strides.
        """
        if not hasattr(self, 'current_features'):
            self.current_features = []
        stride = self.layer_strides.get(module, None)
        if stride is not None:
            self.current_features.append((output.detach(), stride))
            rospy.logdebug(f"Captured feature from module {module} with stride {stride} and shape {output.shape}")
        else:
            rospy.logwarn(f"Stride not found for module {module}")

    def remove_hooks(self):
        """
        Remove all forward hooks.
        """
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

class Track:
    def __init__(self, track_id, bbox, feature, class_id, class_name):
        self.track_id = track_id
        self.bbox = bbox  #[x1, y1, x2, y2]
        self.features = [feature]  #list of feature vectors
        self.mean_feature = feature  #mean feature vector
        self.class_id = class_id
        self.class_name = class_name
        self.age = 1  #total frames since the track was created
        self.time_since_update = 0  #frames since last update
        self.inactive_time = 0  #frames since becoming inactive
        self.is_active = True  #flag indicating if the track is active

    def update(self, bbox, feature):
        """
        Update the track with a new detection.
        """
        self.bbox = bbox
        self.features.append(feature)
        self.time_since_update = 0
        self.inactive_time = 0  #reset inactive time
        self.is_active = True
        self.age += 1

        #update mean feature vector
        self.mean_feature = np.mean(self.features, axis=0)

    def mark_inactive(self):
        """
        Mark the track as inactive when it's not updated.
        """
        self.is_active = False
        self.inactive_time += 1
        self.time_since_update += 1
        self.age += 1

class InventorySystem:
    def __init__(self):
        #initialize the ROS node
        rospy.init_node('inventory_system', anonymous=True)

        #retrieve YOLOv8 model path from ROS parameter or use default
        model_path = rospy.get_param('~model_path', "yolov8m.pt")

        #initialize device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rospy.loginfo(f"Using device: {self.device}")

        #load the YOLOv8 model with feature extraction
        try:
            self.model = YOLOWithFeatureExtraction(model_path).to(self.device)
            rospy.loginfo("YOLOv8 model with feature extraction loaded successfully.")
        except Exception as e:
            rospy.logerr(f"Failed to load YOLOv8 model: {e}")
            rospy.signal_shutdown("YOLOv8 model loading failed.")
            return

        #initialize CvBridge
        self.bridge = CvBridge()

        #subscribe to the ROS topic publishing compressed images
        self.subscriber = rospy.Subscriber('/video_frames', CompressedImage, self.image_callback, queue_size=10)

        #variables for FPS calculation
        self.start_time = time.time()
        self.frame_count = 0  #initialize frame count

        #shared frame variable and lock for thread safety
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        #initialize tracking variables
        self.tracks = []
        self.next_track_id = 0
        self.max_missing_frames = 10  #frames before marking a track as inactive
        self.max_inactive_frames = 1000  #frames to keep an inactive track for possible re-identification
        self.max_age = 1000  #maximum age of a track before deletion

        #define target classes relevant to lab/warehouse environments
        #adjust as needed (Removed class ID 68 for 'microwave')
        self.target_classes = [
            0, 1, 2, 3, 25, 26, 27, 31, 32, 33, 56, 58, 62,
            63, 64, 65, 66, 67, 73, 74, 75, 76, 77, 79
        ]

        #map class IDs to names (from YOLO model)
        self.class_names = self.model.names

        #inventory counts
        self.inventory_detected_items = {}  #{class_name: set of track IDs}

        #similarity threshold
        self.similarity_threshold = 0.45  #adjust for accuracy

        #confidence threshold
        self.confidence_threshold = 0.55  #filter out low-confidence detections

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

        #optionally resize the frame to reduce memory usage
        frame = cv2.resize(frame, (640, 480))  #adjust dimensions as needed

        #clear features before inference
        self.model.current_features = []

        try:
            with torch.no_grad():
                #perform object detection with YOLOv8 using standard inference
                preds = self.model(frame, conf=self.confidence_threshold)
        except Exception as e:
            rospy.logerr(f"YOLOv8 Inference Error: {e}")
            return

        #collect features from current inference
        features = self.model.current_features

        #ensure that features were captured
        if not features:
            rospy.logwarn("Feature maps are empty. No features captured from YOLOv8.")
            return

        #update frame count and calculate FPS
        self.frame_count += 1  #increment frame count
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

        #extract detections and corresponding features
        detections = []
        for r in preds:
            if r.boxes is not None and len(r.boxes) > 0:
                #access class IDs, confidences, and bounding boxes
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                confidences = r.boxes.conf.cpu().numpy()
                boxes = r.boxes.xyxy.cpu().numpy().astype(float)

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    cls_id = class_ids[i]
                    score = confidences[i]

                    #filter out low-confidence detections
                    if score < self.confidence_threshold:
                        continue

                    if cls_id not in self.target_classes:
                        continue  #skip non-target classes

                    class_name = self.class_names.get(cls_id, 'Unknown')

                    #optional: Skip 'microwave' class explicitly (additional safeguard)
                    if class_name == 'microwave':
                        continue  #skip microwave detections

                    #print detection info for debugging
                    rospy.loginfo(f"Detection: Class ID={cls_id}, Class Name={class_name}, Score={score}")

                    #ensure bounding box is within frame boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)

                    #initialize list to store feature vectors from all scales
                    feature_vectors = []

                    #extract features from each feature map
                    for feat, stride in features:
                        #compute spatial scale
                        spatial_scale = 1.0 / stride

                        #prepare ROI for ROI Align
                        roi = torch.tensor([[0, x1 * spatial_scale, y1 * spatial_scale, x2 * spatial_scale, y2 * spatial_scale]], dtype=torch.float32).to(self.device)

                        #create RoIAlign with appropriate spatial_scale
                        roi_align = ops.RoIAlign(
                            output_size=(7, 7),
                            spatial_scale=spatial_scale,
                            sampling_ratio=2,
                            aligned=True
                        )

                        #apply ROI Align
                        pooled_feature = roi_align(feat, roi)

                        #flatten the pooled features to get a feature vector
                        feature_vector = pooled_feature.view(-1)  #shape: (channels * 7 * 7)

                        #normalize the feature vector for cosine similarity
                        feature_vector = F.normalize(feature_vector, p=2, dim=0).cpu().numpy()

                        #append the feature vector
                        feature_vectors.append(feature_vector)

                    #concatenate feature vectors from all scales
                    final_feature_vector = np.concatenate(feature_vectors)

                    #append to detections list
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'score': score,
                        'class_id': cls_id,
                        'class_name': class_name,
                        'feature': final_feature_vector,
                    })
            else:
                rospy.logdebug("No detections in this frame.")

        #clear features after processing
        self.model.current_features = []

        #update tracks
        self.update_tracks(detections)

        #draw tracks
        for track in self.tracks:
            if track.age >= 3 and track.is_active:  #confirmed track after 3 frames
                bbox = track.bbox
                x1, y1, x2, y2 = map(int, bbox)
                class_name = track.class_name
                track_id = track.track_id
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id} - {class_name}",
                            (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        #display inventory counts on the frame
        y_offset = 20
        for idx, (class_name, IDs) in enumerate(self.inventory_detected_items.items()):
            count = len(IDs)
            if class_name == 'person' and self.frame_count <= 3:
                count -= 1  #deduct one 'person' in the first few frames
                if count < 0:
                    count = 0
            text = f"{class_name}: {count} unique objects"
            cv2.putText(frame, text, (10, y_offset + idx * 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

        #update the latest frame for display in a thread-safe manner
        with self.frame_lock:
            self.latest_frame = frame.copy()

        #log FPS
        rospy.loginfo(f"Current FPS: {fps:.2f}")

    def update_tracks(self, detections):
        #predict new locations of existing tracks (not using motion model here for simplicity)
        for track in self.tracks:
            if track.is_active:
                track.time_since_update += 1
                if track.time_since_update > self.max_missing_frames:
                    track.mark_inactive()
            else:
                track.inactive_time += 1

        #match detections to tracks
        matches, unmatched_detections, unmatched_tracks = self.associate_detections_to_tracks(detections, self.tracks)

        #update matched tracks
        for det_idx, trk_idx in matches:
            detection = detections[det_idx]
            track = self.tracks[trk_idx]
            track.update(detection['bbox'], detection['feature'])

        #create new tracks for unmatched detections
        for idx in unmatched_detections:
            detection = detections[idx]

            #ignore 'person' class in the first few frames
            if detection['class_name'] == 'person' and self.frame_count <= 3:
                continue  #skip adding 'person' tracks in the first few frames

            new_track = Track(
                track_id=self.next_track_id,
                bbox=detection['bbox'],
                feature=detection['feature'],
                class_id=detection['class_id'],
                class_name=detection['class_name']
            )
            self.next_track_id += 1
            self.tracks.append(new_track)

            #update inventory counts
            class_name = detection['class_name']
            if class_name not in self.inventory_detected_items:
                self.inventory_detected_items[class_name] = set()
            self.inventory_detected_items[class_name].add(new_track.track_id)
            rospy.loginfo(f"New {class_name} detected with ID: {new_track.track_id}")

        #remove tracks that have been inactive for too long or are too old
        tracks_to_remove = []
        for track in self.tracks:
            if not track.is_active and track.inactive_time > self.max_inactive_frames:
                tracks_to_remove.append(track)
            elif track.age > self.max_age:
                tracks_to_remove.append(track)

        for track in tracks_to_remove:
            rospy.loginfo(f"Removing {track.class_name} with ID: {track.track_id} due to inactivity or max age.")
            self.tracks.remove(track)
            #do NOT remove the track ID from inventory_detected_items to keep the count of unique items

    def associate_detections_to_tracks(self, detections, tracks):
        """
        Associate detections to existing tracks using Cosine Similarity.
        Only compare detections and tracks of the same class.
        Includes both active and inactive tracks for matching.
        Returns matched pairs, unmatched detections, and unmatched tracks.
        """
        matches = []
        unmatched_detections = []
        unmatched_tracks = []

        #group detections and tracks by class ID
        detections_by_class = {}
        for idx, det in enumerate(detections):
            class_id = det['class_id']
            if class_id not in detections_by_class:
                detections_by_class[class_id] = []
            detections_by_class[class_id].append((idx, det))

        tracks_by_class = {}
        for idx, trk in enumerate(tracks):
            class_id = trk.class_id
            if class_id not in tracks_by_class:
                tracks_by_class[class_id] = []
            tracks_by_class[class_id].append((idx, trk))

        #for each class, associate detections and tracks
        for class_id in detections_by_class.keys():
            detections_in_class = detections_by_class[class_id]
            tracks_in_class = tracks_by_class.get(class_id, [])

            if not tracks_in_class:
                unmatched_detections.extend([det_idx for det_idx, _ in detections_in_class])
                continue

            num_detections = len(detections_in_class)
            num_tracks = len(tracks_in_class)

            #compute Cosine Similarity cost
            cosine_cost = np.zeros((num_detections, num_tracks), dtype=np.float32)
            for d_idx, (d_det_idx, det) in enumerate(detections_in_class):
                for t_idx, (t_trk_idx, trk) in enumerate(tracks_in_class):
                    #ensure feature vectors have the same size
                    if det['feature'].shape != trk.mean_feature.shape:
                        rospy.logwarn("Feature vector size mismatch between detection and track.")
                        cosine_similarity = 0  #treat as dissimilar
                    else:
                        #compute cosine similarity
                        numerator = np.dot(det['feature'], trk.mean_feature)
                        denominator = np.linalg.norm(det['feature']) * np.linalg.norm(trk.mean_feature)
                        if denominator == 0:
                            cosine_similarity = 0
                        else:
                            cosine_similarity = numerator / denominator
                    cosine_cost[d_idx, t_idx] = 1 - cosine_similarity  #lower cost for higher similarity

            #apply Hungarian Algorithm for optimal assignment
            row_ind, col_ind = linear_sum_assignment(cosine_cost)

            for d_idx, t_idx in zip(row_ind, col_ind):
                if cosine_cost[d_idx, t_idx] > (1 - self.similarity_threshold):
                    continue
                det_idx, _ = detections_in_class[d_idx]
                trk_idx, _ = tracks_in_class[t_idx]
                matches.append((det_idx, trk_idx))

            #determine unmatched detections and tracks
            matched_det_indices = set([det_idx for det_idx, _ in matches])
            matched_trk_indices = set([trk_idx for _, trk_idx in matches])

            unmatched_detections.extend([det_idx for det_idx, _ in detections_in_class if det_idx not in matched_det_indices])
            unmatched_tracks.extend([trk_idx for trk_idx, _ in tracks_in_class if trk_idx not in matched_trk_indices])

        #remove duplicates in unmatched_tracks (since a track can be unmatched in multiple classes)
        unmatched_tracks = list(set(unmatched_tracks))

        return matches, unmatched_detections, unmatched_tracks

    def shutdown_node(self):
        """Shutdown the ROS node, printing inventory and closing any GUI elements."""
        rospy.loginfo("\nFinal Inventory of Detected Items:")
        if not self.inventory_detected_items:
            rospy.loginfo("No items detected.")
        else:
            for class_name, IDs in self.inventory_detected_items.items():
                count = len(IDs)
                if class_name == 'person' and self.frame_count <= 3:
                    count -= 1  #deduct one 'person' in the first few frames
                    if count < 0:
                        count = 0
                rospy.loginfo(f"{class_name}: {count} unique objects")
        #destroy OpenCV windows
        cv2.destroyAllWindows()
        #remove hooks to avoid memory leaks
        self.model.remove_hooks()

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
            cv2.imshow("YOLOv8 with Feature Tracking", frame_to_display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rospy.loginfo("Detected 'q' key press. Initiating shutdown.")
                rospy.signal_shutdown("Shutdown requested via GUI.")
                break
        rate.sleep()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
