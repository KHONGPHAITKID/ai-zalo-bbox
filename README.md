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
