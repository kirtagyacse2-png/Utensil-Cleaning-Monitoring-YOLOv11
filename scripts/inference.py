"""
Inference script for utensil cleaning monitoring using YOLOv11.
Supports both image files and webcam feed.
"""

import os
import cv2
import numpy as np
import logging
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UtensilCleaningMonitor:
    """
    Class for monitoring utensil cleanliness using YOLOv11 segmentation.
    """

    def __init__(self, model_path="models/utensil_cleaning_monitor/weights/best.pt"):
        """
        Initialize the monitor with trained model.
        Args:
            model_path: Path to the trained model weights
        """
        self.model_path = model_path
        self.model = None
        self.class_names = ['clean', 'dirty']
        self.load_model()

    def load_model(self):
        """Load the YOLOv11 model."""
        if not os.path.exists(self.model_path):
            logger.warning(f"Trained model not found at {self.model_path}, using base model")
            base_model = "models/yolov11n-seg.pt"
            if not os.path.exists(base_model):
                from download_weights import download_yolov11_weights
                download_yolov11_weights(base_model)
            self.model = YOLO(base_model)
        else:
            self.model = YOLO(self.model_path)
            logger.info(f"Loaded trained model from {self.model_path}")

    def predict_image(self, image_path):
        """
        Run prediction on a single image.
        Args:
            image_path: Path to the input image
        Returns:
            dict: Prediction results with status and visualization
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Run inference
        results = self.model(image, conf=0.5)

        # Process results
        processed_results = self.process_results(results, image)

        return processed_results

    def predict_webcam(self, camera_index=0):
        """
        Run real-time prediction using webcam feed.
        Args:
            camera_index: Camera device index (default 0)
        """
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")

        logger.info("Starting webcam monitoring. Press 'q' to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break

                # Run inference
                results = self.model(frame, conf=0.5)

                # Process and display results
                processed_results = self.process_results(results, frame)

                # Display the frame
                cv2.imshow('Utensil Cleaning Monitor', processed_results['annotated_image'])

                # Print status to console
                status = processed_results['status']
                logger.info(f"Status: {status}")

                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def process_results(self, results, original_image):
        """
        Process YOLO results and determine cleaning status.
        Args:
            results: YOLO prediction results
            original_image: Original image array
        Returns:
            dict: Processed results with status and annotated image
        """
        annotated_image = original_image.copy()

        if len(results) > 0 and len(results[0].boxes) > 0:
            # Get the first result (assuming single image)
            result = results[0]

            # Draw segmentation masks and boxes
            annotated_image = result.plot()

            # Analyze detections
            boxes = result.boxes
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()

            # Count clean vs dirty utensils
            clean_count = np.sum(class_ids == 0)  # 0 = clean
            dirty_count = np.sum(class_ids == 1)  # 1 = dirty

            # Determine overall status
            if dirty_count > 0:
                status = "NOT CLEAN"
                status_color = (0, 0, 255)  # Red
            elif clean_count > 0:
                status = "CLEAN"
                status_color = (0, 255, 0)  # Green
            else:
                status = "NO UTENSILS DETECTED"
                status_color = (255, 255, 255)  # White

            # Add status text to image
            cv2.putText(annotated_image, f"Status: {status}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            # Add counts
            cv2.putText(annotated_image, f"Clean: {clean_count}, Dirty: {dirty_count}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        else:
            status = "NO UTENSILS DETECTED"
            status_color = (255, 255, 255)  # White
            cv2.putText(annotated_image, f"Status: {status}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        return {
            'status': status,
            'annotated_image': annotated_image,
            'results': results
        }

def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Utensil Cleaning Monitor using YOLOv11")
    parser.add_argument('--image', type=str, help='Path to input image file')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for real-time monitoring')
    parser.add_argument('--model', type=str, default='models/utensil_cleaning_monitor/weights/best.pt',
                       help='Path to trained model weights')

    args = parser.parse_args()

    # Initialize monitor
    monitor = UtensilCleaningMonitor(args.model)

    if args.image:
        # Process single image
        try:
            results = monitor.predict_image(args.image)
            logger.info(f"Image analysis result: {results['status']}")

            # Display result
            cv2.imshow('Utensil Cleaning Monitor', results['annotated_image'])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")

    elif args.webcam:
        # Start webcam monitoring
        try:
            monitor.predict_webcam()
        except Exception as e:
            logger.error(f"Error with webcam: {str(e)}")

    else:
        logger.error("Please specify either --image or --webcam")

if __name__ == "__main__":
    main()
