"""
Main orchestration script for Utensil Cleaning Monitoring using YOLOv11.
This script handles the complete pipeline from setup to inference.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UtensilCleaningPipeline:
    """
    Main pipeline class for utensil cleaning monitoring system.
    """

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.scripts_dir = self.project_root / "scripts"
        self.models_dir = self.project_root / "models"
        self.data_dir = self.project_root / "data"

    def setup_environment(self):
        """Set up the project environment and dependencies."""
        logger.info("Setting up project environment...")

        # Add scripts directory to Python path
        sys.path.insert(0, str(self.scripts_dir))

        # Ensure required directories exist
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

        logger.info("Environment setup completed")

    def download_weights(self):
        """Download YOLOv11 weights if not present."""
        logger.info("Checking for YOLOv11 weights...")
        from scripts.download_weights import download_yolov11_weights

        model_path = self.models_dir / "yolov11n-seg.pt"
        if not model_path.exists():
            success = download_yolov11_weights(str(model_path))
            if not success:
                raise RuntimeError("Failed to download YOLOv11 weights")
        else:
            logger.info("YOLOv11 weights already available")

    def generate_dataset(self):
        """Generate synthetic dataset if not present."""
        logger.info("Checking for dataset...")
        train_images = self.data_dir / "train" / "images"
        val_images = self.data_dir / "val" / "images"

        if not train_images.exists() or not val_images.exists() or \
           len(list(train_images.glob("*.jpg"))) == 0:

            logger.info("Generating synthetic dataset...")
            from scripts.generate_dataset import generate_dataset
            generate_dataset(str(self.data_dir))
        else:
            logger.info("Dataset already available")

    def train_model(self, epochs=5):
        """Train the YOLOv11 model."""
        logger.info(f"Starting model training for {epochs} epochs...")
        from scripts.train_model import train_yolov11_model

        data_yaml = self.project_root / "data.yaml"
        model_path = self.models_dir / "yolov11n-seg.pt"

        results = train_yolov11_model(
            data_yaml=str(data_yaml),
            model_path=str(model_path),
            epochs=epochs
        )

        logger.info("Model training completed")
        return results

    def run_inference(self, image_path=None, use_webcam=False):
        """Run inference on image or webcam."""
        logger.info("Starting inference...")
        from scripts.inference import UtensilCleaningMonitor

        # Use the best trained model if available, otherwise base model
        best_model = self.models_dir / "utensil_cleaning_monitor" / "weights" / "best.pt"
        if best_model.exists():
            model_path = str(best_model)
        else:
            model_path = str(self.models_dir / "yolov11n-seg.pt")

        monitor = UtensilCleaningMonitor(model_path)

        if image_path:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            results = monitor.predict_image(image_path)
            logger.info(f"Inference result: {results['status']}")
            return results

        elif use_webcam:
            monitor.predict_webcam()
            return None

        else:
            raise ValueError("Must specify either image_path or use_webcam=True")

def main():
    """Main entry point for the utensil cleaning monitoring system."""
    parser = argparse.ArgumentParser(description="Utensil Cleaning Monitoring using YOLOv11")
    parser.add_argument('--setup', action='store_true', help='Set up the project (download weights, generate dataset)')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--image', type=str, help='Path to image for inference')
    parser.add_argument('--webcam', action='store_true', help='Run webcam inference')
    parser.add_argument('--full-pipeline', action='store_true', help='Run complete pipeline: setup -> train -> webcam')

    args = parser.parse_args()

    pipeline = UtensilCleaningPipeline()

    try:
        if args.full_pipeline:
            logger.info("Running complete pipeline...")
            pipeline.setup_environment()
            pipeline.download_weights()
            pipeline.generate_dataset()
            pipeline.train_model(args.epochs)
            pipeline.run_inference(use_webcam=True)

        elif args.setup:
            pipeline.setup_environment()
            pipeline.download_weights()
            pipeline.generate_dataset()

        elif args.train:
            pipeline.setup_environment()
            pipeline.train_model(args.epochs)

        elif args.image:
            pipeline.setup_environment()
            results = pipeline.run_inference(image_path=args.image)
            print(f"Utensil Status: {results['status']}")

        elif args.webcam:
            pipeline.setup_environment()
            pipeline.run_inference(use_webcam=True)

        else:
            print("Utensil Cleaning Monitoring using YOLOv11")
            print("Use --help for available options")
            print("\nQuick start:")
            print("  python main.py --full-pipeline    # Complete setup, train, and webcam demo")
            print("  python main.py --setup           # Setup environment and data")
            print("  python main.py --train           # Train model only")
            print("  python main.py --webcam          # Run webcam inference")
            print("  python main.py --image path/to/image.jpg  # Analyze single image")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
