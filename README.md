# ZALO AI - Object Detection Project

## Data Structure

```
data/
└── train/
    ├── samples/
    │   ├── Backpack_0/
    │   │   ├── drone_video.mp4          # Original drone video
    │   │   ├── object_images/           # Reference images of objects
    │   │   │   ├── img_1.jpg
    │   │   │   ├── img_2.jpg
    │   │   │   └── img_3.jpg
    │   │   └── extract_images/          # Extracted frames (generated)
    │   │       ├── raw/                 # Original frames without annotations
    │   │       │   ├── frame_3483.jpg
    │   │       │   ├── frame_3484.jpg
    │   │       │   └── ...
    │   │       └── bbox/                # Frames with bounding boxes drawn
    │   │           ├── frame_3483.jpg
    │   │           ├── frame_3484.jpg
    │   │           └── ...
    │   ├── MobilePhone_1/
    │   │   ├── drone_video.mp4
    │   │   ├── object_images/
    │   │   └── extract_images/
    │   ├── WaterBottle_1/
    │   ├── Lifering_0/
    │   ├── Person1_0/
    │   └── ...
    └── annotations/
        └── annotations.json             # Bounding box annotations for all videos
```

## Annotations Format

The `annotations.json` file contains a list of video annotations with the following structure:

```json
[
  {
    "video_id": "Backpack_0",
    "annotations": [
      {
        "bboxes": [
          {
            "frame": 3483,
            "x1": 321,
            "y1": 0,
            "x2": 381,
            "y2": 12
          },
          {
            "frame": 3484,
            "x1": 302,
            "y1": 0,
            "x2": 387,
            "y2": 21
          }
        ]
      }
    ]
  }
]
```

### Annotation Fields:
- `video_id`: Name of the sample folder (matches folder name in `samples/`)
- `annotations`: List of annotation objects
- `bboxes`: List of bounding box annotations for specific frames
  - `frame`: Frame number in the video
  - `x1, y1`: Top-left corner coordinates
  - `x2, y2`: Bottom-right corner coordinates

## Scripts

### `extract_data.py`
Extracts frames from videos based on annotations and creates two versions:
- Raw frames (original)
- Annotated frames (with bounding boxes drawn)

**Features:**
- Sequential reading for fast frame extraction
- Multiprocessing support for parallel video processing
- Automatically creates `extract_images/raw/` and `extract_images/bbox/` folders

**Usage:**
```bash
python extract_data.py
```

### `main.py`
Runs RF-DETR object detection inference on random extracted images.

**Features:**
- Randomly selects 5 images from extracted frames
- Runs inference with RF-DETR model
- Saves annotated results to `inference_results/` folder
- Displays detection results with confidence scores

**Usage:**
```bash
python main.py
```

## Dependencies

```bash
pip install opencv-python tqdm pillow supervision torch rfdetr
```

## New UAV Target Pipeline (`uav_object_finder/`)

This repository now bundles an end-to-end, modular pipeline that follows the proposal → embedding → similarity → tracking stack described in the project brief. The implementation lives under `uav_object_finder/` and keeps using the same `data/train/samples/<SampleName>/object_images` references you have already prepared.

### Key components
- **Proposals:** YOLO/DETR backends when available, with a contour-based fallback so the pipeline always produces candidate boxes.
- **Embedding:** DINO/CLIP-style ViT features if PyTorch + TorchVision weights are present; otherwise a lightweight color-moment embedder keeps things running without GPU dependencies.
- **Similarity + Memory:** Reference gallery builder with augmentations, adaptive background calibration, and a FIFO diversity-aware memory bank.
- **Tracking:** ByteTrack-style assignment with Kalman motion, simple Hungarian association (SciPy if installed, greedy otherwise), and an OSTrack-inspired template fallback for short gaps.
- **Post:** Median smoothing plus presence gating to emit `{bbox, conf, present}` per frame while updating the memory bank online.

### Running the new pipeline

```bash
# Example: track the Backpack_0 target using its default references
python -m uav_object_finder.run \
  --video data/train/samples/Backpack_0/drone_video.mp4 \
  --sample Backpack_0 \
  --output video_inference_results/Backpack_0_uav.json
```

Use `--refs /custom/path/*.jpg` if you want to point at an arbitrary reference set, and `--config` to swap in a custom YAML that overrides detector thresholds, memory sizes, etc. Outputs follow the expected `{bbox, conf, present}` schema and can be post-processed into submission files just like the legacy scripts.

### Extra dependencies
- **Required:** `numpy`, `Pillow`, `opencv-python` (for video/proposals), `PyYAML`.
- **Optional but recommended:** `torch`, `torchvision`, `ultralytics`, `scipy`. The code auto-detects these packages and downgrades gracefully when they are missing.
