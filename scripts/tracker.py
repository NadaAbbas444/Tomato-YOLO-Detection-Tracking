import cv2
import os
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import random

class Tracker:
    """
    This Tracker class is designed to process a sequence of images and their corresponding bounding box labels,
    track objects across frames, and visualize the tracking results by drawing bounding boxes with unique colors 
    for each object ID. The class also saves the annotated images to the specified directory and generates a summary file.

    Inputs:
    - image_folder_path: Path to the folder containing the input images.
    - label_folder_path: Path to the folder containing the bounding box label files in .txt format.
    - summary_path: Path to save the summary of tracked objects.

    Outputs:
    - Annotated images saved in the label folder path, with bounding boxes and labels drawn on each image.
    - A summary file containing information about the tracked objects for each image.

    How to use:
    1. Initialize the Tracker class with the path to the summary file:
       `tracker = Tracker(summary_path='path/to/summary.txt')`
    
    2. Call the `process_folder` method with the paths to the image and label folders:
       `tracker.process_folder(image_folder_path='path/to/images', label_folder_path='path/to/labels')`
    
    3. The processed images with annotations will be saved in the label folder, and a summary file will be generated.
    """

    def __init__(self, summary_path):
        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=0.3)
        self.unique_ids = set()  # Set to track unique object IDs
        self.summary_path = summary_path
        self.color_map = {}  # Dictionary to store colors for each ID

    def read_bboxes(self, bbox_file):
        """Reads bounding boxes from a file."""
        with open(bbox_file, 'r') as file:
            bboxes = [list(map(float, line.strip().split())) for line in file]
        return bboxes

    def convert_to_ltrb(self, bbox, image_shape):
        """Converts bounding box format from center_x, center_y, width, height to left, top, right, bottom."""
        class_id, center_x, center_y, width, height = bbox
        img_h, img_w = image_shape[:2]
        
        left = int((center_x - width / 2) * img_w)
        top = int((center_y - height / 2) * img_h)
        right = int((center_x + width / 2) * img_w)
        bottom = int((center_y + height / 2) * img_h)
        
        return [left, top, right, bottom], 1.0, int(class_id)
    
    def filter_similar_detections(self, detections, similarity_threshold=0.6):
        """Filters out detections that are too similar based on IoU."""
        filtered_detections = []
        for i, detA in enumerate(detections):
            similar = False
            for j, detB in enumerate(detections):
                if i != j:
                    iou_value = self.iou(detA[0], detB[0])
                    if iou_value > similarity_threshold:
                        similar = True
                        break
            if not similar:
                filtered_detections.append(detA)
        return filtered_detections
    
    def get_color(self, track_id):
        """Assigns a consistent color for each track ID."""
        if track_id not in self.color_map:
            self.color_map[track_id] = tuple(random.choices(range(256), k=3))
        return self.color_map[track_id]
    
    def iou(self, boxA, boxB):
        """Calculates Intersection over Union (IoU) for two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        return interArea / float(boxAArea + boxBArea - interArea)
    
    def draw_detections(self, image, detections, tracks, image_name, label_folder_path):
        """Draws bounding boxes and labels on the image and saves it."""
        class_names = ['fully_ripened_normal', 'half_ripened_normal', 'green_normal', 
                    'fully_ripened_cherry', 'half_ripened_cherry', 'green_cherry']

        for track in tracks:
            track_id = track.track_id
            track_bbox = track.to_ltrb()

            # Find the best matching detection for the current track
            best_iou = 0
            best_det = None
            for det in detections:
                det_bbox, _, det_class_id = det
                iou_value = self.iou(track_bbox, det_bbox)
                if iou_value > best_iou:
                    best_iou = iou_value
                    best_det = det

            if best_det is not None:
                det_bbox, _, det_class_id = best_det
                color = self.get_color(track_id)
                cv2.rectangle(image, (int(det_bbox[0]), int(det_bbox[1])), 
                            (int(det_bbox[2]), int(det_bbox[3])), color, 3)  # Thicker bounding boxes
                label = f'ID: {track_id} - {class_names[det_class_id]}'
                cv2.putText(image, label, (int(det_bbox[0]), int(det_bbox[1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

        output_path = os.path.join(label_folder_path, image_name)
        cv2.imwrite(output_path, image)


    def process_folder(self, image_folder_path, label_folder_path):        
        """Processes a folder of images and their corresponding labels."""
        image_files = sorted([f for f in os.listdir(image_folder_path) 
                              if f.endswith('.jpg') or f.endswith('.png')])
        bbox_files = sorted([f for f in os.listdir(label_folder_path) if f.endswith('.txt')])

        if not image_files or not bbox_files:
            print("No images or bbox files found in the folder.")
            return

        with open(self.summary_path, 'w') as summary_file:
            for i in range(len(image_files) - 1):
                if i >= len(bbox_files) - 1:
                    print("Reached the last available bbox file.")
                    break

                image1_path = os.path.join(image_folder_path, image_files[i])
                image2_path = os.path.join(image_folder_path, image_files[i + 1])
                bbox1_file = os.path.join(label_folder_path, bbox_files[i])
                bbox2_file = os.path.join(label_folder_path, bbox_files[i + 1])

                image1 = cv2.imread(image1_path)
                image2 = cv2.imread(image2_path)

                if image1 is None or image2 is None:
                    print(f"Error: One of the images could not be loaded.")
                    continue

                bboxes1 = self.read_bboxes(bbox1_file)
                bboxes2 = self.read_bboxes(bbox2_file)

                detections1 = [self.convert_to_ltrb(bbox, image1.shape) for bbox in bboxes1]
                detections2 = [self.convert_to_ltrb(bbox, image2.shape) for bbox in bboxes2]

                detections1 = self.filter_similar_detections(detections1)
                detections2 = self.filter_similar_detections(detections2)

                tracks1 = self.tracker.update_tracks(detections1, frame=image1)
                
                summary_file.write(f"Image: {image_files[i]}")
                for track in tracks1:
                    track_id = track.track_id
                    if track_id not in self.unique_ids:
                        self.unique_ids.add(track_id)
                    bbox = track.to_ltrb()
                    class_id = track.get_det_class()
                    summary_file.write(f"; {track_id} {bbox} {class_id}")

                summary_file.write("\n")

                # Draw and save the detection for the current image
                self.draw_detections(image1,detections1, tracks1, image_files[i], label_folder_path)

                tracks2 = self.tracker.update_tracks(detections2, frame=image2)

                # Draw and save the detection for the next image
                self.draw_detections(image2, detections2,tracks2, image_files[i + 1], label_folder_path)

            print(f"Total unique objects detected: {len(self.unique_ids)}")
