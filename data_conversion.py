import os
import shutil
import json

"""
data_conversion.py

This script converts dataset annotations from COCO format to YOLO format and organizes the images and labels 
into respective directories for training and testing. The conversion ensures that the annotations are in a format 
suitable for YOLO model training, where the bounding boxes are represented as normalized coordinates.

Functions:
- convert_to_yolo_format: Converts COCO bounding box format to YOLO format.
- process_annotations: Processes the annotations for a dataset (train/test) and saves them in YOLO format, 
                       while copying the associated images to a new directory.
"""

# Define paths for the dataset
ROOT_DIR = '/home/nadaabbas/Downloads/Tomatoes_detection'
TRAIN_IMAGES_PATH = os.path.join(ROOT_DIR, 'Tomato_Detection_Counting_Tracking_yolo/laboro_tomato/train')
TEST_IMAGES_PATH = os.path.join(ROOT_DIR, 'Tomato_Detection_Counting_Tracking_yolo/laboro_tomato/test')
TRAIN_ANNOTATIONS_FILE = os.path.join(ROOT_DIR, 'Tomato_Detection_Counting_Tracking_yolo/laboro_tomato/annotations/train.json')
TEST_ANNOTATIONS_FILE = os.path.join(ROOT_DIR, 'Tomato_Detection_Counting_Tracking_yolo/laboro_tomato/annotations/test.json')
OUTPUT_TRAIN_LABELS_PATH = os.path.join(ROOT_DIR, 'Tomato_Detection_Counting_Tracking_yolo/labels/train')
OUTPUT_TEST_LABELS_PATH = os.path.join(ROOT_DIR, 'Tomato_Detection_Counting_Tracking_yolo/labels/test')
OUTPUT_TRAIN_IMAGES_PATH = os.path.join(ROOT_DIR, 'Tomato_Detection_Counting_Tracking_yolo/images/train')
OUTPUT_TEST_IMAGES_PATH = os.path.join(ROOT_DIR, 'Tomato_Detection_Counting_Tracking_yolo/images/test')

# Ensure the output directories exist
os.makedirs(OUTPUT_TRAIN_LABELS_PATH, exist_ok=True)
os.makedirs(OUTPUT_TEST_LABELS_PATH, exist_ok=True)
os.makedirs(OUTPUT_TRAIN_IMAGES_PATH, exist_ok=True)
os.makedirs(OUTPUT_TEST_IMAGES_PATH, exist_ok=True)

def convert_to_yolo_format(annotation, img_width, img_height):
    """
    Converts COCO bounding box format to YOLO format.

    Args:
        annotation (dict): The annotation data containing the bounding box information.
        img_width (int): The width of the image.
        img_height (int): The height of the image.

    Returns:
        str: The annotation in YOLO format.
    """
    x_min, y_min, width, height = annotation['bbox']
    class_id = annotation['category_id'] - 1  # Adjusting category_id for 0-based indexing in YOLO
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width /= img_width
    height /= img_height
    return f"{class_id} {x_center} {y_center} {width} {height}"

def process_annotations(images_path, annotations_file, labels_output_path, images_output_path):
    """
    Processes the annotations for a dataset (train/test) and saves them in YOLO format, 
    while copying the associated images to a new directory.

    Args:
        images_path (str): Path to the images directory.
        annotations_file (str): Path to the annotations JSON file.
        labels_output_path (str): Path to save the converted YOLO labels.
        images_output_path (str): Path to save the images in the YOLO format structure.
    """
    with open(annotations_file) as f:
        annotations = json.load(f)
    
    for image_info in annotations['images']:
        image_id = image_info['id']
        image_file_name = image_info['file_name']
        img_width, img_height = image_info['width'], image_info['height']
        
        txt_file_path = os.path.join(labels_output_path, os.path.splitext(image_file_name)[0] + '.txt')
        
        with open(txt_file_path, 'w') as txt_file:
            for annotation in annotations['annotations']:
                if annotation['image_id'] == image_id:
                    yolo_format = convert_to_yolo_format(annotation, img_width, img_height)
                    txt_file.write(yolo_format + '\n')
        
        # Move the image file to the images directory
        src = os.path.join(images_path, image_file_name)
        dst = os.path.join(images_output_path, image_file_name)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Warning: {src} not found. Skipping...")

# Process both train and test datasets
process_annotations(TRAIN_IMAGES_PATH, TRAIN_ANNOTATIONS_FILE, OUTPUT_TRAIN_LABELS_PATH, OUTPUT_TRAIN_IMAGES_PATH)
process_annotations(TEST_IMAGES_PATH, TEST_ANNOTATIONS_FILE, OUTPUT_TEST_LABELS_PATH, OUTPUT_TEST_IMAGES_PATH)

print("YOLO annotation files have been saved to the labels directory and images copied to the images directory.")
