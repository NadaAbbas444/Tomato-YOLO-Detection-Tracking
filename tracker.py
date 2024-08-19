import cv2
import os
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self):
        # Initialize the DeepSort tracker
        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
        print("Initialized DeepSort tracker.")

    def read_bboxes(self, bbox_file):
        print(f"Reading bounding boxes from: {bbox_file}")
        with open(bbox_file, 'r') as file:
            # Read the bounding boxes from the file
            bboxes = [list(map(float, line.strip().split())) for line in file]
        print(f"Loaded {len(bboxes)} bounding boxes.")
        return bboxes

    def convert_to_ltrb(self, bbox, image_shape):
        # print(f"Converting bounding box: {bbox}")
        # Convert from YOLO format to DeepSort expected format
        class_id, center_x, center_y, width, height = bbox
        img_h, img_w = image_shape[:2]
        
        left = int((center_x - width / 2) * img_w)
        top = int((center_y - height / 2) * img_h)
        right = int((center_x + width / 2) * img_w)
        bottom = int((center_y + height / 2) * img_h)
        
        converted_bbox = [left, top, right - left, bottom - top]
        # print(f"Converted to LTRB: {converted_bbox}")
        return converted_bbox, 1.0, int(class_id)  # default confidence to 1.0

    def process_folder(self, folder_path, output_file):
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')])
        bbox_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])

        if not image_files or not bbox_files:
            print("No images or bbox files found in the folder.")
            return

        unique_detections = []
        total_detections = 0

        for i in range(len(image_files) - 1):
            image1_path = os.path.join(folder_path, image_files[i])
            image2_path = os.path.join(folder_path, image_files[i+1])
            bbox1_file = os.path.join(folder_path, bbox_files[i])
            bbox2_file = os.path.join(folder_path, bbox_files[i+1])

            print(f"Processing images: {image1_path} and {image2_path}")

            image1 = cv2.imread(image1_path)
            image2 = cv2.imread(image2_path)

            if image1 is None or image2 is None:
                print(f"Error: One of the images could not be loaded.")
                continue

            bboxes1 = self.read_bboxes(bbox1_file)
            bboxes2 = self.read_bboxes(bbox2_file)

            detections1 = [self.convert_to_ltrb(bbox, image1.shape) for bbox in bboxes1]
            detections2 = [self.convert_to_ltrb(bbox, image2.shape) for bbox in bboxes2]

            # Update tracker with first frame detections
            print(f"Updating tracker with first frame detections.")
            tracks1 = self.tracker.update_tracks(detections1, frame=image1)
            print(f"Tracks after first frame: {len(tracks1)}")

            # Store the unique detections from the first image
            for track in tracks1:
                if track.is_confirmed():
                    track_id = track.track_id
                    bbox = track.to_ltrb()
                    unique_detections.append(f"{image_files[i]}; {track_id} {bbox} {track.cls}")
                    total_detections += 1

            # Update tracker with second frame detections
            print(f"Updating tracker with second frame detections.")
            tracks2 = self.tracker.update_tracks(detections2, frame=image2)
            print(f"Tracks after second frame: {len(tracks2)}")

            # Process tracks in the second frame to detect unique objects
            for track in tracks2:
                if track.is_confirmed():
                    track_id = track.track_id
                    bbox = track.to_ltrb()
                    if f"{track_id} {bbox}" not in unique_detections:
                        unique_detections.append(f"{image_files[i+1]}; {track_id} {bbox} {track.cls}")
                        total_detections += 1

        # Save unique detections to the text file
        with open(output_file, 'w') as file:
            for detection in unique_detections:
                file.write(detection + "\n")

        print(f"Unique detections saved to {output_file}")
        print(f"Total unique objects detected: {total_detections}")

tracker = Tracker()

folder_path = "/home/nadaabbas/Downloads/Tomatoes_detection/Tomato_Detection_Counting_Tracking_yolo/test"
output_file = "/home/nadaabbas/Downloads/Tomatoes_detection/Tomato_Detection_Counting_Tracking_yolo/test/unique_detections.txt"

tracker.process_folder(folder_path, output_file)
