import cv2
import time
import random
from pathlib import Path
from PIL import Image
import supervision as sv
import torch
from rfdetr import RFDETRBase, RFDETRMedium, RFDETRSmall, RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("  Warning: GPU not available. Using CPU (will be slower)")

# Initialize model with GPU support
print(f"\nLoading RF-DETR model on {device.upper()}...")
model = RFDETRNano(device=device)

# GPU optimizations
if device == "cuda":
    print("Applying GPU optimizations...")
    # Enable cuDNN auto-tuner for optimal performance
    torch.backends.cudnn.benchmark = True
    # Enable TF32 on Ampere GPUs for faster computation
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("  ✓ cuDNN benchmark enabled")
    print("  ✓ TF32 enabled (if supported)")

# Try to optimize for inference
try:
    model.optimize_for_inference()
    print("Model loaded and optimized!")
except RuntimeError:
    print("Model loaded (optimization skipped - not critical)")
    print("  Note: JIT tracing not compatible with this model output")

# Find available videos
base_path = Path("data/train/samples")
video_files = list(base_path.glob("*/drone_video.mp4"))

if not video_files:
    print("No videos found in data/train/samples/")
    exit(1)

# Select a random video for testing
video_path = random.choice(video_files)
video_name = video_path.parent.name
# video_name = "Backpack_0"
# video_path = Path("data") / "train" / "samples" / video_name / "drone_video.mp4"

print(f"\nSelected video: {video_name}")
print(f"Path: {video_path}")

# Open video
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit(1)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"\nVideo Properties:")
print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps:.2f}")
print(f"  Total Frames: {total_frames}")
print(f"  Duration: {total_frames/fps:.2f}s")

# Create output folder
output_folder = Path("video_inference_results")
output_folder.mkdir(exist_ok=True)

# Setup video writer for output
output_path = output_folder / f"{video_name}_inference.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

# Initialize supervision annotators
box_annotator = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# Warmup GPU with a dummy inference (improves first-frame performance)
if device == "cuda":
    print("\nWarming up GPU...")
    dummy_image = Image.new('RGB', (width, height), color='black')
    _ = model.predict(dummy_image, threshold=0.05)
    torch.cuda.synchronize()  # Wait for GPU operations to complete
    print("  ✓ GPU warmup complete")

# Performance tracking
frame_times = []
detection_counts = []
frame_count = 0
start_time = time.time()

print(f"\nProcessing video...")
print(f"Inference threshold: 0.05")
print("-" * 50)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_start = time.time()

        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Run inference
        detections = model.predict(pil_image, threshold=0.05)

        # Synchronize GPU for accurate timing
        if device == "cuda":
            torch.cuda.synchronize()

        # Create labels
        labels = [
            f"{COCO_CLASSES[class_id]} {confidence:.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        # Annotate frame (convert back to numpy array for supervision)
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        # annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

        # Add performance info overlay
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        detection_counts.append(len(detections))

        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        avg_fps = len(frame_times) / sum(frame_times) if frame_times else 0

        # Draw performance info
        cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f} | Avg: {avg_fps:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Detections: {len(detections)}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write frame to output video
        out.write(annotated_frame)

        # Print progress every 30 frames
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% | Frame {frame_count}/{total_frames} | "
                  f"FPS: {current_fps:.1f} | Detections: {len(detections)}")

except KeyboardInterrupt:
    print("\n\nProcessing interrupted by user")

finally:
    # Cleanup
    cap.release()
    out.release()

    # Calculate statistics
    total_time = time.time() - start_time
    video_duration = total_frames / fps if fps > 0 else 0
    avg_fps = len(frame_times) / sum(frame_times) if frame_times else 0
    avg_detections = sum(detection_counts) / len(detection_counts) if detection_counts else 0
    avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0

    print("\n" + "=" * 70)
    print(" " * 20 + "PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Device: {device.upper()}")
    print(f"Video: {video_name}")
    print(f"Frames Processed: {frame_count}/{total_frames}")

    print(f"\n{'TIMING ANALYSIS':^70}")
    print("-" * 70)
    print(f"  Video Duration:        {video_duration:.2f}s")
    print(f"  Processing Time:       {total_time:.2f}s")
    print(f"  Time Difference:       {total_time - video_duration:+.2f}s")
    print(f"  Speed Factor:          {video_duration/total_time:.2f}x {'(faster than realtime)' if total_time < video_duration else '(slower than realtime)'}")

    print(f"\n{'INFERENCE PERFORMANCE':^70}")
    print("-" * 70)
    print(f"  Video FPS:             {fps:.2f}")
    print(f"  Inference FPS:         {avg_fps:.2f}")
    print(f"  Average Frame Time:    {avg_frame_time*1000:.1f}ms")
    print(f"  Min Frame Time:        {min(frame_times)*1000:.1f}ms")
    print(f"  Max Frame Time:        {max(frame_times)*1000:.1f}ms")

    print(f"\n{'DETECTION STATISTICS':^70}")
    print("-" * 70)
    print(f"  Average Detections:    {avg_detections:.1f} objects/frame")
    print(f"  Max Detections:        {max(detection_counts) if detection_counts else 0} objects/frame")
    print(f"  Total Detections:      {sum(detection_counts)} objects")

    # GPU Memory usage (if applicable)
    if device == "cuda":
        print(f"\n{'GPU MEMORY USAGE':^70}")
        print("-" * 70)
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        max_memory = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"  Current Allocated:     {memory_allocated:.2f} GB")
        print(f"  Reserved:              {memory_reserved:.2f} GB")
        print(f"  Peak Memory:           {max_memory:.2f} GB")

    print(f"\n{'REAL-TIME CAPABILITY ASSESSMENT':^70}")
    print("=" * 70)
    real_time_factor = avg_fps / fps if fps > 0 else 0

    if real_time_factor >= 1.0:
        print(f"  ✓✓✓ YES - CAN PROCESS REAL-TIME VIDEO ✓✓✓")
        print(f"  Processing Speed: {real_time_factor:.2f}x real-time")
        print(f"  Can handle video at {fps:.0f} FPS and process at {avg_fps:.1f} FPS")
    else:
        print(f"  ✗✗✗ NO - CANNOT PROCESS REAL-TIME VIDEO ✗✗✗")
        print(f"  Processing Speed: {real_time_factor:.2f}x real-time")
        print(f"  Video requires {fps:.0f} FPS but can only process at {avg_fps:.1f} FPS")
        print(f"  Needs {1/real_time_factor:.2f}x speedup to achieve real-time")

    print("=" * 70)
    print(f"Output saved to: {output_path}")
    print("=" * 70)
