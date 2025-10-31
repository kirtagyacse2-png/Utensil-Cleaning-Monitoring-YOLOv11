"""
Script to train YOLOv11 segmentation model for utensil cleaning monitoring.
"""

import os
import logging
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_weights_available(model_path="models/yolov11n-seg.pt"):
    """
    Ensure YOLOv11 weights are available, download if necessary.

    Args:
        model_path: Path to the model weights
    """
    if not os.path.exists(model_path):
        logger.info("Model weights not found, downloading...")
        from download_weights import download_yolov11_weights
        success = download_yolov11_weights(model_path)
        if not success:
            # For demo purposes, create a mock model that will work with the synthetic dataset
            logger.warning("Using mock model for demonstration purposes")
            logger.info("In production, real YOLOv11 weights would be required")
            # We'll proceed with training using YOLO's built-in initialization
            return False  # Indicate we don't have real weights
    else:
        logger.info(f"Using existing model weights: {model_path}")
    return True

def ensure_dataset_available(data_yaml="data.yaml"):
    """
    Ensure dataset is available, generate if necessary.

    Args:
        data_yaml: Path to the data configuration file
    """
    if not os.path.exists(data_yaml):
        logger.error(f"Data configuration file not found: {data_yaml}")
        raise FileNotFoundError(f"Data configuration file not found: {data_yaml}")

    # Check if dataset directories exist and have images
    train_images = "data/train/images"
    val_images = "data/val/images"

    if not os.path.exists(train_images) or len(os.listdir(train_images)) == 0:
        logger.info("Training dataset not found, generating...")
        from generate_dataset import generate_dataset
        generate_dataset()

    if not os.path.exists(val_images) or len(os.listdir(val_images)) == 0:
        logger.info("Validation dataset not found, generating...")
        from generate_dataset import generate_dataset
        generate_dataset()

def train_yolov11_model(data_yaml="data.yaml", model_path="models/yolov11n-seg.pt", epochs=10):
    """
    Train YOLOv11 segmentation model.

    Args:
        data_yaml: Path to data configuration file
        model_path: Path to pretrained model weights
        epochs: Number of training epochs
    """
    logger.info("Starting YOLOv11 training...")

    # Ensure prerequisites
    ensure_weights_available(model_path)
    ensure_dataset_available(data_yaml)

    # Load the model - use pretrained weights if available, otherwise initialize from scratch
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000:  # Check if it's a real weights file
        model = YOLO(model_path)
        logger.info("Loaded pretrained YOLOv11 model")
    else:
        # For demo purposes, use YOLOv8 as fallback since YOLOv11 may not be available
        logger.warning("YOLOv11 weights not available, using YOLOv8 segmentation model for demo")
        try:
            model = YOLO('yolov8n-seg.pt')  # Try YOLOv8 segmentation
            logger.info("Using YOLOv8 segmentation model as fallback")
        except Exception as e:
            logger.error(f"Could not load any segmentation model: {e}")
            raise RuntimeError("No suitable segmentation model available")

    # Training configuration
    training_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': 4,  # Small batch size for limited dataset
        'imgsz': 640,
        'save': True,
        'save_period': 5,
        'cache': False,  # Disable caching for synthetic data
        'device': 'cpu',  # Use CPU for compatibility
        'workers': 0,  # Disable multiprocessing for Windows compatibility
        'project': 'models',
        'name': 'utensil_cleaning_monitor',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'Adam',
        'lr0': 0.001,
        'weight_decay': 0.0005,
        'momentum': 0.937,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
    }

    # Start training
    logger.info(f"Training with configuration: {training_args}")
    results = model.train(**training_args)

    logger.info("Training completed!")
    logger.info(f"Best model saved at: {results.save_dir}/weights/best.pt")

    return results

if __name__ == "__main__":
    try:
        results = train_yolov11_model(epochs=5)  # Few epochs for demo
        logger.info("Training pipeline completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise