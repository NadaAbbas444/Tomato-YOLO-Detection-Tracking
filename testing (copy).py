import torch
import cv2
import os
from pathlib import Path

# Path to the directory with images
images_dir = '/home/nadaabbas/Downloads/Tomatoes_detection/yolo/testing_images'

# Path to the directory where results will be saved
results_dir = '/home/nadaabbas/Downloads/Tomatoes_detection/yolo/results_images'
os.makedirs(results_dir, exist_ok=True)

# Path to the trained YOLO model
model_path = '/home/nadaabbas/Downloads/Tomatoes_detection/yolo/yolov5/runs/train/exp22/weights/best.pt'

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Get the list of images
image_files = list(Path(images_dir).glob('*.jpg'))  # Adjust to your image format if needed

# Function to process and save images with detections
def save_images_with_detections(image_files, model, results_dir):
    for img_path in image_files:
        # Load image
        img = cv2.imread(str(img_path))

        # Inference
        results = model(img)

        # Convert BGR to RGB for saving
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Plot detections on the image
        for *xyxy, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            score = conf.item()

            # Draw rectangle and label
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_rgb, f"{label} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Convert back to BGR for saving
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Save the image
        save_path = os.path.join(results_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, img_bgr)

# Run the function to process and save images with detections
save_images_with_detections(image_files, model, results_dir)
