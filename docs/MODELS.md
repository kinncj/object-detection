# Model Documentation

This document provides detailed information about the supported object detection models, their capabilities, and configuration options.

## 🤖 Supported Models

### YOLOv8 (Recommended, Default)

**Overview**: Ultralytics YOLOv8 is a state-of-the-art object detection model that provides excellent balance between speed and accuracy.

**Available Sizes**:
- `n` (nano): Fastest, smallest model - ideal for real-time applications
- `s` (small): Good balance of speed and accuracy 
- `m` (medium): Higher accuracy with moderate speed
- `l` (large): High accuracy for demanding applications
- `x` (extra-large): Maximum accuracy, slower inference

**Performance Characteristics**:
| Size | Parameters | Model Size | Speed (CPU) | Speed (GPU) | mAP50 |
|------|------------|------------|-------------|-------------|-------|
| n    | 3.2M       | 6.2 MB     | ~45 FPS     | ~200 FPS    | 50.7  |
| s    | 11.2M      | 21.5 MB    | ~35 FPS     | ~150 FPS    | 57.8  |
| m    | 25.9M      | 49.7 MB    | ~25 FPS     | ~120 FPS    | 63.9  |
| l    | 43.7M      | 83.7 MB    | ~20 FPS     | ~100 FPS    | 67.6  |
| x    | 68.2M      | 130.5 MB   | ~15 FPS     | ~80 FPS     | 69.4  |

**Configuration**:
```python
# Default configuration
model = ModelFactory.create_model('yolo', 'n')

# Custom configuration
model = ModelFactory.create_model(
    model_type='yolo',
    model_size='s',
    confidence_threshold=0.6
)
```

**Detected Classes** (COCO dataset subset):
- Person
- Cell phone
- Laptop
- TV/Monitor
- Keyboard
- Mouse
- Clock

### DETR (Detection Transformer)

**Overview**: Facebook's DETR (DEtection TRansformer) uses a transformer architecture for object detection, providing high-quality detections with end-to-end learning.

**Model Specifications**:
- **Backbone**: ResNet-50
- **Architecture**: Transformer encoder-decoder
- **Training Dataset**: COCO 2017
- **Input Resolution**: Flexible (automatically resized)
- **Parameters**: ~41M
- **Model Size**: ~159 MB

**Performance Characteristics**:
- **Accuracy**: High precision, especially for complex scenes
- **Speed**: Slower than YOLO models (~5-10 FPS on CPU)
- **Memory**: Higher memory usage due to transformer architecture
- **Strengths**: Excellent for detecting small objects and complex compositions

**Configuration**:
```python
# Default configuration
model = ModelFactory.create_model('detr')

# Custom confidence threshold
model = ModelFactory.create_model(
    model_type='detr',
    confidence_threshold=0.95
)
```

**Detected Classes** (Same subset as YOLOv8):
- Person
- Cell phone  
- Laptop
- TV/Monitor
- Keyboard
- Mouse
- Clock

## 🎯 Model Selection Guide

### When to Use YOLOv8

**Best for**:
- Real-time video processing
- Resource-constrained environments
- Batch processing of multiple videos
- Applications requiring fast inference
- Mobile or edge deployment

**Recommended Sizes**:
- **YOLOv8n**: Real-time applications, live video streams
- **YOLOv8s**: General-purpose detection with good performance
- **YOLOv8m**: High-quality results for important applications
- **YOLOv8l/x**: Maximum accuracy for critical applications

### When to Use DETR

**Best for**:
- High-accuracy requirements
- Complex scene analysis
- Research and experimentation
- Applications where inference time is not critical
- Fine-grained object detection

**Trade-offs**:
- Higher computational requirements
- Slower inference speed
- Better handling of occlusions
- More stable detection in challenging conditions

## ⚙️ Configuration Options

### Confidence Thresholds

**YOLOv8 Default**: 0.5
- Lower values (0.3-0.4): More detections, higher recall, more false positives
- Higher values (0.6-0.8): Fewer detections, higher precision, lower recall

**DETR Default**: 0.9
- Lower values (0.7-0.8): More detections, but potentially noisier
- Higher values (0.95+): Very conservative, only high-confidence detections

### Model Loading

Models are automatically downloaded on first use:

**YOLOv8**:
- Downloaded from Ultralytics hub
- Cached locally in `~/.ultralytics/`
- Automatic version management

**DETR**:
- Downloaded from Hugging Face hub
- Cached locally in `~/.cache/huggingface/`
- Automatic tokenizer and model syncing

## 🚀 Performance Optimization

### GPU Acceleration

Both models support GPU acceleration:

```python
# Automatic GPU detection (recommended)
model = ModelFactory.create_model('yolo', 'n')

# Check GPU availability
import torch
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name()}")
```

### Memory Optimization

**For YOLOv8**:
- Use smaller model sizes for memory-constrained environments
- Enable mixed precision training if fine-tuning
- Process videos in batches if memory allows

**For DETR**:
- Reduce input image resolution if needed
- Use CPU offloading for very large videos
- Consider using DETR-DC5 variant for better memory efficiency

### Batch Processing

```python
# Process multiple frames efficiently
results = []
for frame in frames:
    detections = model.detect_objects(frame)
    results.append(detections)
```

## 📊 Model Comparison

| Aspect | YOLOv8n | YOLOv8s | YOLOv8m | DETR |
|--------|---------|---------|---------|------|
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Accuracy** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Memory** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Setup** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Versatility** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🔧 Advanced Configuration

### Custom Model Loading

```python
# Custom YOLOv8 model path
from models.yolo_model import YOLOv8Model
model = YOLOv8Model(model_path="path/to/custom/model.pt")

# Custom DETR model
from models.detr_model import DETRModel
model = DETRModel(model_name="custom/detr-model")
```

### Model Fine-tuning

While the application uses pre-trained models by default, you can fine-tune models for specific use cases:

**YOLOv8 Fine-tuning**:
```bash
# Train on custom dataset
yolo train model=yolov8n.pt data=custom_dataset.yaml epochs=100
```

**DETR Fine-tuning**:
```python
# Use Hugging Face training scripts
from transformers import Trainer, TrainingArguments
# ... training configuration
```

## 🎨 Class Mappings

Both models use a unified class mapping system for consistency:

```python
CLASS_MAPPING = {
    0: "person",
    62: "tv", 
    63: "laptop",
    64: "mouse",
    66: "keyboard", 
    67: "cell phone",
    74: "clock"
}
```

This ensures that detection results are compatible regardless of the underlying model architecture.

## 🐛 Troubleshooting

### Common Issues

**Model Download Failures**:
- Check internet connection
- Verify disk space availability
- Clear model cache if corrupted

**GPU Memory Issues**:
- Reduce batch size
- Use smaller model variants
- Enable gradient checkpointing

**Slow Performance**:
- Verify GPU availability
- Check CPU usage and background processes
- Consider using lower resolution inputs

**Accuracy Issues**:
- Adjust confidence thresholds
- Try different model sizes
- Verify input video quality

For more troubleshooting tips, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
