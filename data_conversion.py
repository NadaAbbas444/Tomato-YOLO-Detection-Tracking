import os
import shutil
import json

# Paths to your dataset
train_images_path = '/home/nadaabbas/Downloads/Tomatoes_detection/yolo/laboro_tomato/train'
test_images_path = '/home/nadaabbas/Downloads/Tomatoes_detection/yolo/laboro_tomato/test'
train_annotations_file = '/home/nadaabbas/Downloads/Tomatoes_detection/yolo/laboro_tomato/annotations/train.json'
test_annotations_file = '/home/nadaabbas/Downloads/Tomatoes_detection/yolo/laboro_tomato/annotations/test.json'
output_train_labels_path = '/home/nadaabbas/Downloads/Tomatoes_detection/yolo/labels/train'
output_test_labels_path = '/home/nadaabbas/Downloads/Tomatoes_detection/yolo/labels/test'
output_train_images_path = '/home/nadaabbas/Downloads/Tomatoes_detection/yolo/images/train'
output_test_images_path = '/home/nadaabbas/Downloads/Tomatoes_detection/yolo/images/test'

# Ensure the output directories exist
os.makedirs(output_train_labels_path, exist_ok=True)
os.makedirs(output_test_labels_path, exist_ok=True)
os.makedirs(output_train_images_path, exist_ok=True)
os.makedirs(output_test_images_path, exist_ok=True)

def convert_to_yolo_format(annotation, img_width, img_height):
    x_min, y_min, width, height = annotation['bbox']
    class_id = annotation['category_id'] - 1
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width /= img_width
    height /= img_height
    return f"{class_id} {x_center} {y_center} {width} {height}"

def process_annotations(images_path, annotations_file, labels_output_path, images_output_path):
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

# Process both train and test data
process_annotations(train_images_path, train_annotations_file, output_train_labels_path, output_train_images_path)
process_annotations(test_images_path, test_annotations_file, output_test_labels_path, output_test_images_path)

print(f"YOLO annotation files have been saved to the labels directory and images copied to the images directory.")
