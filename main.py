import os
import random
from pathlib import Path
import supervision as sv
from PIL import Image
import torch
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Initialize model
print(f"Loading RF-DETR model on {device.upper()}...")
model = RFDETRBase(device=device)

# Try to optimize for inference (may not work with all PyTorch versions)
try:
    model.optimize_for_inference()
    print("Model loaded and optimized!")
except RuntimeError as e:
    print("Model loaded (optimization skipped - not critical)")
    print(f"  Note: JIT tracing not compatible with this model output")

# Find all raw extracted images
base_path = Path("data/train/samples")
raw_images = list(base_path.glob("*/extract_images/raw/*.jpg"))

if not raw_images:
    print("No extracted images found! Please run extract_data.py first.")
    exit(1)

print(f"Found {len(raw_images)} extracted images")

# Select random images for testing (5 images)
num_test_images = min(5, len(raw_images))
test_images = random.sample(raw_images, num_test_images)

print(f"Testing on {num_test_images} random images...")

# Create output folder for inference results
output_folder = Path("inference_results")
output_folder.mkdir(exist_ok=True)

# Run inference on each test image
for i, image_path in enumerate(test_images, 1):
    print(f"\n[{i}/{num_test_images}] Processing: {image_path.name}")
    print(f"  Source: {image_path.parent.parent.parent.name}")

    # Load image
    image = Image.open(image_path)

    # Run inference
    detections = model.predict(image, threshold=0.05)

    print(f"  Detected {len(detections)} objects:")

    # Create labels with class names and confidence scores
    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    # Print detections
    for label in labels:
        print(f"    - {label}")

    # Annotate image
    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    # annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

    # Save annotated image
    output_path = output_folder / f"inference_{i}_{image_path.stem}.jpg"
    annotated_image.save(output_path)
    print(f"  Saved to: {output_path}")

print(f"\n✓ Inference complete! Results saved to '{output_folder}/' folder")
print(f"  Processed {num_test_images} images")