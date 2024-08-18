# Tomato Detection, Counting, and Tracking with YOLO

## Project Overview

This project aims to create a robust and efficient model for detecting, counting, and tracking tomatoes using the YOLO (You Only Look Once) object detection algorithm. The model is trained on a dataset containing various stages of tomato ripeness and size classes, allowing it to accurately identify and categorize tomatoes in different conditions.

## Dataset

The dataset used in this project is based on the Laboro Tomato Dataset, which includes annotations for:

- **Tomato Size**: 
  - Normal
  - Cherry

- **Ripeness Stages**:
  - Fully Ripened
  - Half Ripened
  - Green

Each tomato is categorized based on these attributes, allowing the model to distinguish between different tomato types and ripeness stages.

## Project Structure

The project is organized as follows:
```
Tomato_Detection_Counting_Tracking_yolo/
├── yolov5/                   # YOLOv5 framework
├── data.yaml                 # Configuration file for the dataset
├── data_conversion.py        # Script for converting annotations to YOLO format
├── images/
│   ├── train/                # Training images
│   └── test/                 # Test images
├── labels/
│   ├── train/                # YOLO format annotations for training data
│   └── test/                 # YOLO format annotations for test data
├── laboro_tomato/            # Original dataset
├── testing.py                # Script for testing the model on custom images
├── testing_images/           # Folder containing images for testing
└── README.md                 # This README file
```

## Training the Model

To train the model, use the following command:

```bash
python yolov5/train.py --img 640 --batch 16 --epochs 50 --data /path/to/data.yaml --weights yolov5s.pt --cache
```

This command trains the YOLOv5 model using the specified dataset, batch size, and number of epochs. The trained model weights will be saved in the `runs/train/exp/weights/` directory.

## Testing the Model

To test the model on custom images, use the following command:

```bash
python yolov5/detect.py --weights /path/to/weights/best.pt --source /path/to/testing_images/ --save-txt --save-conf --project /path/to/save_results/
```

This command runs the YOLOv5 model on the specified images and saves the detection results in the `save_results` directory.

## Results

The model was trained for 50 epochs, achieving the following metrics:

- **Precision (P)**: 83.8%
- **Recall (R)**: 74.2%
- **mAP@50**: 84.0%
- **mAP@50-95**: 69.1%

These results demonstrate the model's effectiveness in detecting and categorizing tomatoes across different ripeness stages and sizes.

## GitHub Repository

This project is hosted on GitHub. You can clone the repository using the following command:

```bash
git clone https://github.com/NadaAbbas444/Tomato_Detection_Counting_Tracking_yolo.git
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

This project leverages the YOLOv5 implementation by [Ultralytics](https://github.com/ultralytics/yolov5) and the Laboro Tomato Dataset. Special thanks to the contributors of these resources for their invaluable tools and data.

## How to Contribute

We welcome contributions to this project! If you have suggestions, improvements, or would like to add new features, feel free to fork the repository and submit a pull request. Please make sure your code adheres to our contribution guidelines.

## Contact

If you have any questions, suggestions, or issues, please feel free to contact me at [your-email@example.com](mailto:your-email@example.com).

Thank you for using and contributing to this project!
