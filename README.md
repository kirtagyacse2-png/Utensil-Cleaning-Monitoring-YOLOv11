# Utensil Cleaning Monitoring using YOLOv11

A complete Python project that uses YOLOv11 segmentation model to monitor utensil cleanliness in real-time. The system can detect whether utensils are clean or dirty using computer vision.

## Features

- **YOLOv11 Segmentation**: Uses Ultralytics YOLOv11 with segmentation capabilities
- **Automatic Weight Download**: Downloads model weights with corruption checking and fallback URLs
- **Synthetic Dataset Generation**: Creates 75 synthetic images with clean/dirty utensil classes
- **Real-time Inference**: Supports both image files and webcam feed
- **Easy Setup**: Fully automated setup and training pipeline

## Project Structure

```
Utensil Cleaning Monitoring using YOLOv11/
├── main.py                 # Main orchestration script
├── requirements.txt        # Python dependencies
├── data.yaml              # Dataset configuration
├── scripts/
│   ├── download_weights.py # Weight download with corruption check
│   ├── generate_dataset.py # Synthetic dataset generation
│   ├── train_model.py     # Model training script
│   └── inference.py       # Inference and monitoring
├── data/
│   ├── train/
│   │   ├── images/        # Training images
│   │   └── labels/        # YOLO format labels
│   └── val/
│       ├── images/        # Validation images
│       └── labels/        # YOLO format labels
└── models/                # Model weights and checkpoints
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
python main.py --full-pipeline
```

This will:
- Set up the environment
- Download YOLOv11 weights
- Generate synthetic dataset
- Train the model for 5 epochs
- Start webcam monitoring

### Alternative Commands

```bash
# Setup only (download weights, generate dataset)
python main.py --setup

# Train model only
python main.py --train --epochs 10

# Run webcam inference
python main.py --webcam

# Analyze single image
python main.py --image path/to/your/image.jpg
```

## Detailed Usage

### Setup Phase

The setup phase ensures all prerequisites are available:

1. **Weight Download**: Downloads `yolov11n-seg.pt` from Ultralytics with fallback URLs
2. **Dataset Generation**: Creates 75 synthetic images (60 train, 15 val) with:
   - Clean utensils (silver appearance)
   - Dirty utensils (dull with random stains)
   - YOLO format labels for segmentation

### Training Phase

- Uses YOLOv11 segmentation model
- Trains on synthetic dataset
- Saves best model to `models/utensil_cleaning_monitor/weights/best.pt`
- Configurable epochs (default: 5 for demo)

### Inference Phase

- **Image Mode**: Analyze single images and display results
- **Webcam Mode**: Real-time monitoring with live status display
- **Status Output**: Prints "CLEAN", "NOT CLEAN", or "NO UTENSILS DETECTED"
- **Visualization**: Shows segmentation masks and bounding boxes

## Requirements

- Python 3.8+
- Webcam (for real-time monitoring)
- Internet connection (for initial weight download)

## Dependencies

- ultralytics>=8.0.0
- opencv-python>=4.8.0
- numpy>=1.24.0
- pillow>=10.0.0
- torch>=2.0.0
- torchvision>=0.15.0
- requests>=2.31.0
- tqdm>=4.65.0

## How It Works

1. **Synthetic Dataset**: Creates realistic utensil images with clean/dirty variations
2. **Model Training**: Fine-tunes YOLOv11 on the synthetic dataset
3. **Inference**: Uses trained model to segment and classify utensils
4. **Status Determination**:
   - If any dirty utensils detected → "NOT CLEAN"
   - If only clean utensils detected → "CLEAN"
   - If no utensils detected → "NO UTENSILS DETECTED"

## Customization

### Dataset Configuration

Edit `data.yaml` to modify:
- Class names
- Dataset paths
- Training parameters

### Model Parameters

Modify training parameters in `scripts/train_model.py`:
- Batch size
- Image size
- Learning rate
- Optimizer settings

### Inference Settings

Adjust confidence thresholds and visualization in `scripts/inference.py`

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in training
2. **Webcam not working**: Check camera permissions and index
3. **Model download fails**: Check internet connection and firewall
4. **Training slow**: Use CPU mode or reduce epochs

### Logs

All operations are logged with timestamps. Check console output for detailed information.

## License

This project is for educational and demonstration purposes.

## Contributing

Feel free to improve the synthetic dataset generation, add more utensil types, or enhance the inference visualization.
