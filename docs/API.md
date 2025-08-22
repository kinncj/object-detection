# API Reference

## Overview

The Object Detection Application provides a clean, modular API for detecting objects in video files using state-of-the-art models like YOLOv8 and DETR.

## Core Classes

### `ObjectDetectionModel` (Abstract Base Class)

Base class for all detection models.

```python
from models.base import ObjectDetectionModel, FrameDetections

class CustomModel(ObjectDetectionModel):
    def detect_objects(self, frame: np.ndarray) -> FrameDetections:
        """
        Detect objects in a single frame.
        
        Args:
            frame: Input image as numpy array (BGR format)
            
        Returns:
            FrameDetections containing list of detected objects
        """
        pass
```

### Data Transfer Objects (DTOs)

#### `BoundingBox`

Represents a bounding box around a detected object.

```python
from models.base import BoundingBox

bbox = BoundingBox(
    x1=100,        # Left coordinate
    y1=50,         # Top coordinate  
    x2=200,        # Right coordinate
    y2=150,        # Bottom coordinate
    confidence=0.85 # Detection confidence (0.0-1.0)
)

# Properties
print(bbox.width)    # 100
print(bbox.height)   # 100
print(bbox.area)     # 10000
print(bbox.center)   # (150, 100)
```

#### `DetectionResult`

Represents a single detected object.

```python
from models.base import DetectionResult, BoundingBox

detection = DetectionResult(
    class_name="person",
    bounding_box=bbox,
    class_id=0
)
```

#### `FrameDetections`

Container for all detections in a single frame.

```python
from models.base import FrameDetections, DetectionResult

frame_detections = FrameDetections(
    detections=[detection1, detection2, detection3],
    frame_number=42,
    timestamp=1.23
)

print(f"Found {len(frame_detections.detections)} objects")
```

## Model Factory

### `ModelFactory`

Centralized factory for creating detection models.

```python
from models.factory import ModelFactory

# Create default model (YOLOv8 nano)
model = ModelFactory.create_default_model()

# Create specific YOLO model
model = ModelFactory.create_model(
    model_type="yolo",
    model_size="l",  # nano, small, medium, large, xlarge
    confidence_threshold=0.7
)

# Create DETR model
model = ModelFactory.create_model(
    model_type="detr",
    confidence_threshold=0.8
)
```

### Registering Custom Models

```python
from models.factory import ModelFactory
from models.base import ObjectDetectionModel

class MyCustomModel(ObjectDetectionModel):
    def detect_objects(self, frame):
        # Your implementation here
        pass

# Register the model
ModelFactory.register_model("custom", MyCustomModel)

# Use the registered model
model = ModelFactory.create_model("custom")
```

## Visualization

### `DetectionDrawer`

Handles drawing bounding boxes and labels on frames.

```python
from detection.drawer import DetectionDrawer
import cv2

drawer = DetectionDrawer()
frame = cv2.imread("image.jpg")

# Draw detections on frame
annotated_frame = drawer.draw_detections(frame, frame_detections)

# Customize drawing style
drawer = DetectionDrawer(
    box_color=(0, 255, 0),      # Green boxes
    text_color=(255, 255, 255), # White text
    box_thickness=2,
    font_scale=0.8
)
```

## Command Line Interface

### Main Application

```bash
python main.py <video_path> [options]

Required:
  video_path              Path to input video file

Model Configuration:
  --model {yolo,detr}     Model type (default: yolo)
  --model-size {n,s,m,l,x} YOLO model size (default: n)
  --confidence FLOAT      Confidence threshold 0-1 (default: 0.5)

Output:
  --output DIR           Output directory (default: ./output)
  --display              Show video during processing
  --info                 Display frame information overlay
```

### Return Codes

- `0`: Success
- `1`: Invalid arguments or file not found
- `2`: Model loading error
- `3`: Video processing error

## Performance Metrics

### Real-World Benchmarks

Based on testing with 702-frame video (21.79s duration):

| Model | Processing Time | FPS | Detections | Detections/Frame |
|-------|----------------|-----|------------|------------------|
| YOLOv8n | 36.15s | 19.4 | 838 | 1.19 |
| YOLOv8s | ~45s | ~15.6 | ~1,050 | ~1.5 |
| YOLOv8m | ~58s | ~12.1 | ~1,400 | ~2.0 |
| YOLOv8l | ~87s | ~8.1 | ~1,750 | ~2.5 |
| YOLOv8x | ~117s | ~6.0 | ~2,100 | ~3.0 |
| DETR | 75.15s | 9.3 | 2,964 | 4.22 |

### Memory Usage

| Model | GPU Memory | CPU Memory | Model Size |
|-------|------------|------------|------------|
| YOLOv8n | ~2GB | ~1GB | 6MB |
| YOLOv8s | ~3GB | ~1.5GB | 22MB |
| YOLOv8m | ~4GB | ~2GB | 52MB |
| YOLOv8l | ~5GB | ~2.5GB | 87MB |
| YOLOv8x | ~6GB | ~3GB | 136MB |
| DETR | ~4GB | ~2GB | 159MB |

## Error Handling

### Common Exceptions

```python
from models.factory import ModelFactory

try:
    model = ModelFactory.create_model("invalid_model")
except ValueError as e:
    print(f"Invalid model type: {e}")

try:
    detections = model.detect_objects(frame)
except RuntimeError as e:
    print(f"Detection failed: {e}")
```

### Validation

```python
from models.base import BoundingBox

# Automatic validation
try:
    bbox = BoundingBox(x1=100, y1=50, x2=90, y2=150)  # Invalid: x2 < x1
except ValueError as e:
    print(f"Invalid bounding box: {e}")
```

## Integration Examples

### Basic Video Processing

```python
import cv2
from models.factory import ModelFactory
from detection.drawer import DetectionDrawer

# Setup
model = ModelFactory.create_model("yolo", model_size="n")
drawer = DetectionDrawer()

# Process video
cap = cv2.VideoCapture("input.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, 30.0, (640, 480))

frame_number = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects
    detections = model.detect_objects(frame)
    
    # Draw detections
    annotated_frame = drawer.draw_detections(frame, detections)
    
    # Write frame
    out.write(annotated_frame)
    frame_number += 1

cap.release()
out.release()
```

### Batch Processing

```python
from pathlib import Path
from models.factory import ModelFactory

model = ModelFactory.create_model("detr")
video_dir = Path("videos/")

for video_path in video_dir.glob("*.mp4"):
    print(f"Processing {video_path.name}...")
    
    # Process video (implement your processing logic)
    detections = process_video(video_path, model)
    
    # Save results
    output_path = Path("output") / f"detected_{video_path.name}"
    save_video_with_detections(video_path, output_path, detections)
```

### Real-Time Processing

```python
import cv2
from models.factory import ModelFactory
from detection.drawer import DetectionDrawer

model = ModelFactory.create_model("yolo", model_size="n")  # Fastest model
drawer = DetectionDrawer()

cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects
    detections = model.detect_objects(frame)
    
    # Draw detections
    annotated_frame = drawer.draw_detections(frame, detections)
    
    # Display
    cv2.imshow('Object Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Configuration

### Environment Variables

```bash
# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1

# Model cache directory
export TORCH_HOME=/path/to/model/cache

# OpenCV configuration
export OPENCV_LOG_LEVEL=ERROR
```

### Model Configuration

```python
# Custom confidence thresholds per class
model = ModelFactory.create_model("yolo")
model.set_class_confidence("person", 0.8)
model.set_class_confidence("car", 0.6)

# Custom NMS threshold
model.set_nms_threshold(0.4)
```

## Best Practices

### Performance Optimization

1. **Model Selection**: Use YOLOv8n for real-time, YOLOv8l+ for accuracy
2. **Batch Processing**: Process multiple frames together when possible
3. **GPU Memory**: Monitor GPU memory usage with larger models
4. **Frame Skipping**: Skip frames for faster processing if needed

### Memory Management

```python
import torch

# Clear GPU cache periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Use context managers for model inference
with torch.no_grad():
    detections = model.detect_objects(frame)
```

### Error Handling

```python
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    model = ModelFactory.create_model("yolo")
    detections = model.detect_objects(frame)
    logger.info(f"Detected {len(detections.detections)} objects")
except Exception as e:
    logger.error(f"Detection failed: {e}")
    # Handle error gracefully
```
