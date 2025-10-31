"""
Script to generate a dummy dataset with synthetic images for utensil cleaning monitoring.
Creates 50-100 images with two classes: clean and dirty utensils.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset configuration
NUM_IMAGES = 75  # 75 images total (37-38 per class)
IMAGE_SIZE = (640, 480)
CLASSES = ['clean', 'dirty']
BACKGROUND_COLORS = [(240, 240, 240), (200, 200, 200), (220, 220, 220)]  # Light backgrounds

def create_synthetic_utensil(image, class_type, bbox):
    """
    Draw a synthetic utensil on the image.
    Args:
        image: PIL Image object
        class_type: 'clean' or 'dirty'
        bbox: tuple (x1, y1, x2, y2) for bounding box
    """
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    width = x2 - x1
    height = y2 - y1

    # Draw a simple utensil shape (fork-like)
    if class_type == 'clean':
        # Clean utensil - bright and shiny
        utensil_color = (220, 220, 220)  # Silver color
        shadow_color = (200, 200, 200)
    else:
        # Dirty utensil - with stains
        utensil_color = (180, 180, 180)  # Dull color
        shadow_color = (150, 150, 150)

    # Draw main body
    draw.rectangle([x1+10, y1+10, x2-10, y2-10], fill=utensil_color, outline=(100, 100, 100))

    # Draw handle
    draw.rectangle([x1+5, center_y-5, x1+15, center_y+5], fill=utensil_color, outline=(100, 100, 100))

    # Add some details
    if class_type == 'dirty':
        # Add random stains for dirty utensils
        for _ in range(random.randint(3, 8)):
            stain_x = random.randint(x1+15, x2-15)
            stain_y = random.randint(y1+15, y2-15)
            stain_size = random.randint(2, 6)
            draw.ellipse([stain_x, stain_y, stain_x+stain_size, stain_y+stain_size],
                        fill=(139, 69, 19), outline=(101, 67, 33))

    return image

def generate_yolo_label(bbox, class_id, image_size):
    """
    Convert bounding box to YOLO segmentation format.
    For segmentation, we need polygon points instead of just bbox.
    Args:
        bbox: tuple (x1, y1, x2, y2)
        class_id: int, class index
        image_size: tuple (width, height)
    Returns:
        str: YOLO segmentation format label line
    """
    x1, y1, x2, y2 = bbox
    img_w, img_h = image_size

    # Create a simple rectangular polygon from the bounding box
    # Normalize coordinates to 0-1 range
    x1_norm = x1 / img_w
    y1_norm = y1 / img_h
    x2_norm = x2 / img_w
    y2_norm = y2 / img_h

    # Create polygon points (rectangle): x1,y1 -> x2,y1 -> x2,y2 -> x1,y2 -> x1,y1
    polygon_points = [
        x1_norm, y1_norm,  # top-left
        x2_norm, y1_norm,  # top-right
        x2_norm, y2_norm,  # bottom-right
        x1_norm, y2_norm,  # bottom-left
        x1_norm, y1_norm   # back to top-left to close polygon
    ]

    # Format as space-separated values
    points_str = " ".join(f"{p:.6f}" for p in polygon_points)

    return f"{class_id} {points_str}"

def generate_dataset(data_dir="data"):
    """
    Generate synthetic dataset for utensil cleaning monitoring.
    Args:
        data_dir: Base directory for dataset
    """
    train_images_dir = os.path.join(data_dir, "train", "images")
    train_labels_dir = os.path.join(data_dir, "train", "labels")
    val_images_dir = os.path.join(data_dir, "val", "images")
    val_labels_dir = os.path.join(data_dir, "val", "labels")

    # Ensure directories exist
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)

    logger.info(f"Generating {NUM_IMAGES} synthetic images...")

    for i in range(NUM_IMAGES):
        # Randomly assign class
        class_type = random.choice(CLASSES)
        class_id = CLASSES.index(class_type)

        # Create base image with random background
        bg_color = random.choice(BACKGROUND_COLORS)
        image = Image.new('RGB', IMAGE_SIZE, bg_color)
        draw = ImageDraw.Draw(image)

        # Add some random noise/texture to background
        for _ in range(50):
            x = random.randint(0, IMAGE_SIZE[0]-5)
            y = random.randint(0, IMAGE_SIZE[1]-5)
            draw.point((x, y), fill=(bg_color[0] + random.randint(-20, 20),
                                    bg_color[1] + random.randint(-20, 20),
                                    bg_color[2] + random.randint(-20, 20)))

        # Generate random bounding box for utensil
        min_size = 80
        max_size = 150
        utensil_w = random.randint(min_size, max_size)
        utensil_h = random.randint(min_size, max_size)

        x1 = random.randint(50, IMAGE_SIZE[0] - utensil_w - 50)
        y1 = random.randint(50, IMAGE_SIZE[1] - utensil_h - 50)
        x2 = x1 + utensil_w
        y2 = y1 + utensil_h

        bbox = (x1, y1, x2, y2)

        # Draw the utensil
        image = create_synthetic_utensil(image, class_type, bbox)

        # Determine if this goes to train or val (80/20 split)
        is_train = i < int(NUM_IMAGES * 0.8)
        if is_train:
            img_dir = train_images_dir
            label_dir = train_labels_dir
            split = "train"
        else:
            img_dir = val_images_dir
            label_dir = val_labels_dir
            split = "val"

        # Save image
        img_filename = f"{i:04d}.jpg"
        img_path = os.path.join(img_dir, img_filename)
        image.save(img_path)

        # Save YOLO label
        label_filename = f"{i:04d}.txt"
        label_path = os.path.join(label_dir, label_filename)
        yolo_label = generate_yolo_label(bbox, class_id, IMAGE_SIZE)
        with open(label_path, 'w') as f:
            f.write(yolo_label + '\n')

        logger.info(f"Generated {split} image {i+1}/{NUM_IMAGES}: {class_type} utensil")

    logger.info("Dataset generation completed!")
    logger.info(f"Train images: {len(os.listdir(train_images_dir))}")
    logger.info(f"Val images: {len(os.listdir(val_images_dir))}")

if __name__ == "__main__":
    generate_dataset()
