import json
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def extract_frames_with_bboxes(video_path, bboxes, output_folder):
    """
    Extract frames from video and save both original and bbox-annotated versions.
    Uses sequential reading for much better performance.

    Args:
        video_path: Path to the video file
        bboxes: List of bbox dictionaries with frame, x1, y1, x2, y2
        output_folder: Path to save extracted frames
    """
    # Create output subfolders
    raw_folder = os.path.join(output_folder, "raw")
    bbox_folder = os.path.join(output_folder, "bbox")
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(bbox_folder, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0

    # Get all unique frames that need to be extracted
    frame_to_bbox = {}
    for bbox in bboxes:
        frame_num = bbox['frame']
        if frame_num not in frame_to_bbox:
            frame_to_bbox[frame_num] = []
        frame_to_bbox[frame_num].append(bbox)

    # Get sorted list of frames to extract
    target_frames = sorted(frame_to_bbox.keys())
    if not target_frames:
        cap.release()
        return 0

    # Sequential reading (much faster than seeking)
    current_frame = 0
    target_idx = 0
    extracted_count = 0

    while target_idx < len(target_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Check if this is a frame we need
        if current_frame == target_frames[target_idx]:
            # Save original frame to raw folder
            original_path = os.path.join(raw_folder, f"frame_{current_frame}.jpg")
            cv2.imwrite(original_path, frame)

            # Create a copy for drawing bounding boxes
            frame_with_bbox = frame.copy()

            # Draw all bounding boxes for this frame
            for bbox in frame_to_bbox[current_frame]:
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                # Draw rectangle (green color, thickness 2)
                cv2.rectangle(frame_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Save frame with bounding box to bbox folder
            bbox_path = os.path.join(bbox_folder, f"frame_{current_frame}.jpg")
            cv2.imwrite(bbox_path, frame_with_bbox)

            extracted_count += 1
            target_idx += 1

        current_frame += 1

        # Early exit if we've extracted all frames
        if target_idx >= len(target_frames):
            break

    cap.release()
    return extracted_count


def process_single_video(args):
    """
    Process a single video (for multiprocessing).

    Args:
        args: Tuple of (sample_folder, video_annotations)
    """
    sample_folder, video_annotations = args
    video_id = sample_folder.name

    # Find the video file
    video_file = sample_folder / "drone_video.mp4"
    if not video_file.exists():
        return f"Warning: Video file not found at {video_file}"

    # Create extract_images folder
    extract_folder = sample_folder / "extract_images"

    # Get all bboxes for this video
    all_bboxes = []
    for annotation in video_annotations:
        if 'bboxes' in annotation:
            all_bboxes.extend(annotation['bboxes'])

    if not all_bboxes:
        return f"Warning: No bboxes found for {video_id}"

    # Extract frames with bounding boxes
    extracted_count = extract_frames_with_bboxes(video_file, all_bboxes, extract_folder)
    return f"Processed {video_id}: {extracted_count} frames extracted"


def main():
    # Define paths
    base_path = Path("data/train")
    samples_path = base_path / "samples"
    annotations_path = base_path / "annotations" / "annotations.json"

    # Load annotations
    print("Loading annotations...")
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Create a dictionary for quick lookup: video_id -> annotations
    video_annotations = {}
    for item in annotations:
        video_id = item['video_id']
        video_annotations[video_id] = item['annotations']

    print(f"Found {len(video_annotations)} videos in annotations")

    # Process each sample folder
    sample_folders = sorted([f for f in samples_path.iterdir() if f.is_dir()])
    print(f"Found {len(sample_folders)} sample folders")

    # Prepare tasks for multiprocessing
    tasks = []
    for sample_folder in sample_folders:
        video_id = sample_folder.name
        if video_id in video_annotations:
            tasks.append((sample_folder, video_annotations[video_id]))

    print(f"Processing {len(tasks)} videos with multiprocessing...")

    # Use multiprocessing to process videos in parallel
    num_workers = min(cpu_count(), len(tasks))  # Don't use more workers than tasks
    print(f"Using {num_workers} workers")

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_video, tasks),
                           total=len(tasks),
                           desc="Processing videos"))

    # Print results
    print("\nProcessing complete!")
    for result in results:
        if result.startswith("Warning"):
            print(result)


if __name__ == "__main__":
    main()
