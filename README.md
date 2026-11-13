
# Real-Time Traffic Sign Detection

A computer vision project that detects traffic signs in images, videos, and live webcam streams using YOLOv8 and the Ultralytics framework.

The model was trained for 150 epochs on the Roboflow Road Signs dataset using Google Colab with GPU acceleration. The project includes custom image inference, batch evaluation, video processing, and MP4 output conversion.

## Overview

The system uses YOLOv8 to identify traffic signs and locate them within an image or video frame. Detected signs are displayed with bounding boxes, class labels, and confidence scores.

The trained model can be reused for predictions without repeating the training process.

## Features

- Traffic sign detection using YOLOv8.
- Training on the Roboflow Road Signs dataset.
- 150 training epochs with 640 × 640 image resolution.
- Automatic saving of the best model weights.
- Google Drive integration for model and dataset storage.
- Batch evaluation on dataset images.
- Custom image upload and detection.
- Video processing with frame-by-frame detection.
- MP4 conversion for processed videos.
- Live webcam detection.
- Bounding boxes, class labels, and confidence scores.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLOv8 | Object detection |
| Ultralytics | Model training and inference |
| Roboflow | Dataset management |
| OpenCV | Image and video processing |
| PyTorch | Deep learning framework |
| NumPy | Numerical operations |
| Google Colab | Model training environment |
| Google Drive | Cloud storage |

## Model Configuration

| Parameter | Value |
|---|---|
| Model | YOLOv8s |
| Training Epochs | 150 |
| Image Size | 640 × 640 |
| Dataset | Roboflow Road Signs |
| GPU | NVIDIA T4 |
| Confidence Threshold (Image) | 0.25 |
| Confidence Threshold (Webcam) | 0.40 |
| Best Weights | `best.pt` |

## Workflow

1. Download the traffic sign dataset from Roboflow.
2. Prepare the dataset configuration using `data.yaml`.
3. Load the YOLOv8s model.
4. Train the model for 150 epochs.
5. Evaluate model performance on the available dataset.
6. Save the best-performing weights.
7. Load the trained model for image, video, or webcam inference.
8. Process detection results and save output videos in MP4 format.

## Project Structure

```text
TrafficSignDetection/
│
├── TrafficSignDetection.ipynb
│
├── road-signs-2/
│   └── data.yaml
│
├── best.pt
│
└── runs/
    └── detect/
        └── train/
            └── weights/
                ├── best.pt
                └── last.pt
```

Note: Training outputs and model files are saved according to the configured Google Drive paths.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/sameermustaqeem878-source/TrafficSignDetection.git
```

### 2. Install Dependencies

```bash
pip install ultralytics roboflow opencv-python
```

### 3. Open the Notebook

Open `TrafficSignDetection.ipynb` in Google Colab.

### 4. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 5. Load the Trained Model

```python
from ultralytics import YOLO

model = YOLO("best.pt")
```

### 6. Detect Traffic Signs in an Image

```python
results = model.predict(
    source="test_image.jpg",
    conf=0.25,
    save=True
)
```

### 7. Detect Traffic Signs in a Video

```python
results = model.predict(
    source="input_video.mp4",
    conf=0.25,
    save=True
)
```

### 8. Run Webcam Detection

```python
results = model.predict(
    source=0,
    conf=0.4,
    show=True
)
```

## Applications

- Intelligent transportation systems.
- Driver assistance systems.
- Traffic monitoring.
- Road safety applications.
- Computer vision research.

## Author

**Muhammad Sameer Mustaqeem**

[GitHub Profile](https://github.com/sameermustaqeem878-source)
