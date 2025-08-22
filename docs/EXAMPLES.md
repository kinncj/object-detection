# Usage Examples and Tutorials

This document provides comprehensive examples and tutorials for using the Object Detection application effectively.

## 🎯 Basic Usage Examples

### Example 1: Quick Start
```bash
# Process a video with default settings (YOLOv8 nano)
python main.py path/to/video.mp4

# View the output
# Output will be saved to ./output/detected_video.mp4
```

### Example 2: Model Selection
```bash
# Use different YOLO model sizes
python main.py video.mp4 --model yolo --model-size n  # Fastest (default)
python main.py video.mp4 --model yolo --model-size s  # Balanced
python main.py video.mp4 --model yolo --model-size l  # High accuracy

# Use DETR model for comprehensive detection
python main.py video.mp4 --model detr --confidence 0.8
```

### Example 3: Custom Output and Display
```bash
# Custom output directory
python main.py video.mp4 --output ./my_results/

# Real-time display with info overlay
python main.py video.mp4 --display --info

# Combine options
python main.py video.mp4 --model yolo --model-size s --confidence 0.7 --output ./results/ --display
```

## 🔧 Python API Examples

### Example 1: Basic Object Detection

```python
import cv2
from models.factory import ModelFactory
from detection.drawer import DetectionDrawer

# Create model and drawer
model = ModelFactory.create_model("yolo", model_size="n")
drawer = DetectionDrawer()

# Load and process single image
image = cv2.imread("sample.jpg")
detections = model.detect_objects(image)

# Draw detections
annotated_image = drawer.draw_detections(image, detections)

# Save result
cv2.imwrite("result.jpg", annotated_image)

# Print detection summary
print(f"Found {len(detections.detections)} objects:")
for detection in detections.detections:
    print(f"- {detection.class_name}: {detection.bounding_box.confidence:.2f}")
```

### Example 2: Video Processing

```python
import cv2
from models.factory import ModelFactory
from detection.drawer import DetectionDrawer

def process_video(input_path, output_path, model_type="yolo", model_size="n"):
    """Process a video file with object detection."""
    
    # Setup
    model = ModelFactory.create_model(model_type, model_size=model_size)
    drawer = DetectionDrawer()
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            detections = model.detect_objects(frame)
            total_detections += len(detections.detections)
            
            # Draw detections
            annotated_frame = drawer.draw_detections(frame, detections)
            
            # Write frame
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:  # Progress every 30 frames
                print(f"Processed {frame_count} frames...")
                
    finally:
        cap.release()
        out.release()
    
    print(f"Processing complete!")
    print(f"Total frames: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per frame: {total_detections/frame_count:.2f}")

# Usage
process_video("input.mp4", "output.mp4", model_type="yolo", model_size="s")
```

### Example 3: Batch Processing

```python
from pathlib import Path
from models.factory import ModelFactory
import cv2

def batch_process_videos(input_dir, output_dir, model_type="yolo"):
    """Process all videos in a directory."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.glob(f"*{ext}"))
    
    print(f"Found {len(video_files)} video files")
    
    model = ModelFactory.create_model(model_type)
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\\nProcessing {i}/{len(video_files)}: {video_file.name}")
        
        output_file = output_path / f"detected_{video_file.name}"
        
        # Process with the function from Example 2
        process_video(str(video_file), str(output_file), model_type)

# Usage
batch_process_videos("./input_videos/", "./output_videos/", "yolo")
```

### Example 4: Real-time Webcam Detection

```python
import cv2
from models.factory import ModelFactory
from detection.drawer import DetectionDrawer
import time

def realtime_detection(model_type="yolo", model_size="n", camera_id=0):
    """Real-time object detection from webcam."""
    
    # Setup
    model = ModelFactory.create_model(model_type, model_size=model_size)
    drawer = DetectionDrawer()
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Performance tracking
    fps_counter = 0
    start_time = time.time()
    
    print("Starting real-time detection. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            frame_start = time.time()
            detections = model.detect_objects(frame)
            inference_time = time.time() - frame_start
            
            # Draw detections
            annotated_frame = drawer.draw_detections(frame, detections)
            
            # Add performance info
            fps_counter += 1
            elapsed = time.time() - start_time
            current_fps = fps_counter / elapsed if elapsed > 0 else 0
            
            # Draw performance text
            cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Inference: {inference_time*1000:.1f}ms", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Detections: {len(detections.detections)}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('Real-time Object Detection', annotated_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
    print(f"\\nSession completed. Average FPS: {current_fps:.1f}")

# Usage
realtime_detection(model_type="yolo", model_size="n")  # Fastest for real-time
```

### Example 5: Custom Model Implementation

```python
from models.base import ObjectDetectionModel, FrameDetections, DetectionResult, BoundingBox
from models.factory import ModelFactory
import numpy as np

class CustomDetectionModel(ObjectDetectionModel):
    """Example custom model implementation."""
    
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        # Initialize your custom model here
        
    def detect_objects(self, frame: np.ndarray) -> FrameDetections:
        """Implement your custom detection logic."""
        
        # Example: Simple edge-based "detection"
        # In practice, this would be your actual model inference
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours as mock "detections"
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            # Filter small contours
            area = cv2.contourArea(contour)
            if area < 1000:  # Minimum area threshold
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Create detection
            bbox = BoundingBox(
                x1=float(x),
                y1=float(y),
                x2=float(x + w),
                y2=float(y + h),
                confidence=0.8  # Mock confidence
            )
            
            detection = DetectionResult(
                class_name="custom_object",
                bounding_box=bbox,
                class_id=999  # Custom class ID
            )
            
            detections.append(detection)
        
        return FrameDetections(detections=detections)

# Register and use the custom model
ModelFactory.register_model("custom", CustomDetectionModel)

# Now you can use it like any other model
model = ModelFactory.create_model("custom", confidence_threshold=0.7)
detections = model.detect_objects(frame)
```

## 🎛️ Advanced Configuration Examples

### Example 1: Performance Optimization

```python
import torch
from models.factory import ModelFactory

# Enable GPU acceleration
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Use first GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Create model with optimized settings
model = ModelFactory.create_model(
    model_type="yolo",
    model_size="n",  # Use nano for maximum speed
    confidence_threshold=0.7  # Higher threshold for fewer false positives
)

# Memory optimization
with torch.no_grad():  # Disable gradients for inference
    detections = model.detect_objects(frame)

# Clear GPU cache periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### Example 2: Multi-Model Comparison

```python
from models.factory import ModelFactory
import time
import cv2

def compare_models(image_path):
    """Compare different models on the same image."""
    
    image = cv2.imread(image_path)
    models = [
        ("YOLOv8n", "yolo", {"model_size": "n"}),
        ("YOLOv8s", "yolo", {"model_size": "s"}),
        ("DETR", "detr", {})
    ]
    
    results = {}
    
    for name, model_type, kwargs in models:
        print(f"\\nTesting {name}...")
        
        # Create model
        model = ModelFactory.create_model(model_type, **kwargs)
        
        # Time inference
        start_time = time.time()
        detections = model.detect_objects(image)
        inference_time = time.time() - start_time
        
        results[name] = {
            "inference_time": inference_time,
            "detection_count": len(detections.detections),
            "detections": detections
        }
        
        print(f"  Inference time: {inference_time:.3f}s")
        print(f"  Detections: {len(detections.detections)}")
    
    # Print comparison
    print("\\n" + "="*50)
    print("Model Comparison Results:")
    print("="*50)
    
    for name, result in results.items():
        print(f"{name:12s}: {result['inference_time']:6.3f}s, {result['detection_count']:3d} detections")

# Usage
compare_models("test_image.jpg")
```

## 🔍 Troubleshooting Examples

### Example 1: Debugging Detection Issues

```python
from models.factory import ModelFactory
from detection.drawer import DetectionDrawer
import cv2

def debug_detection(image_path, model_type="yolo", model_size="n"):
    """Debug detection issues with detailed output."""
    
    print(f"Debugging detection for: {image_path}")
    print(f"Model: {model_type}, Size: {model_size}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("ERROR: Could not load image!")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Create model
    try:
        model = ModelFactory.create_model(model_type, model_size=model_size)
        print("✅ Model created successfully")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return
    
    # Run detection
    try:
        detections = model.detect_objects(image)
        print(f"✅ Detection completed: {len(detections.detections)} objects found")
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        return
    
    # Detailed detection info
    if len(detections.detections) > 0:
        print("\\nDetection details:")
        for i, detection in enumerate(detections.detections):
            bbox = detection.bounding_box
            print(f"  {i+1}. {detection.class_name}")
            print(f"     Confidence: {bbox.confidence:.3f}")
            print(f"     Box: ({bbox.x1:.1f}, {bbox.y1:.1f}) -> ({bbox.x2:.1f}, {bbox.y2:.1f})")
            print(f"     Size: {bbox.width:.1f} x {bbox.height:.1f}")
    else:
        print("\\n⚠️  No detections found")
        print("Try:")
        print("  - Lower confidence threshold")
        print("  - Different model (try DETR vs YOLO)")
        print("  - Check image quality/content")

# Usage
debug_detection("problematic_image.jpg", "yolo", "n")
```

### Example 2: Memory Usage Monitoring

```python
import psutil
import torch
from models.factory import ModelFactory
import cv2

def monitor_memory_usage(video_path, model_type="yolo"):
    """Monitor memory usage during video processing."""
    
    # Setup
    model = ModelFactory.create_model(model_type)
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    
    print("Frame | CPU RAM | GPU RAM | Detections")
    print("-" * 40)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            detections = model.detect_objects(frame)
            
            # Monitor memory
            cpu_percent = psutil.virtual_memory().percent
            gpu_memory = "N/A"
            
            if torch.cuda.is_available():
                gpu_memory = f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB"
            
            frame_count += 1
            
            if frame_count % 10 == 0:  # Print every 10 frames
                print(f"{frame_count:5d} | {cpu_percent:6.1f}% | {gpu_memory:>7s} | {len(detections.detections):10d}")
            
            # Clear GPU cache if memory is high
            if torch.cuda.is_available() and torch.cuda.memory_allocated() > 2 * 1024**3:  # 2GB
                torch.cuda.empty_cache()
                
    finally:
        cap.release()

# Usage
monitor_memory_usage("large_video.mp4", "yolo")
```

## 📝 Testing Examples

### Example 1: Unit Testing Custom Models

```python
import unittest
import numpy as np
from models.base import BoundingBox, DetectionResult, FrameDetections

class TestCustomModel(unittest.TestCase):
    """Unit tests for custom model implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
    def test_bounding_box_validation(self):
        """Test bounding box validation."""
        
        # Valid bounding box
        valid_bbox = BoundingBox(10, 10, 100, 100, 0.8)
        self.assertEqual(valid_bbox.width, 90)
        self.assertEqual(valid_bbox.height, 90)
        
        # Invalid bounding box (x2 < x1)
        with self.assertRaises(ValueError):
            BoundingBox(100, 10, 10, 100, 0.8)
            
        # Invalid confidence
        with self.assertRaises(ValueError):
            BoundingBox(10, 10, 100, 100, 1.5)  # > 1.0
    
    def test_model_factory(self):
        """Test model factory functionality."""
        from models.factory import ModelFactory
        
        # Test valid model creation
        model = ModelFactory.create_model("yolo", model_size="n")
        self.assertIsNotNone(model)
        
        # Test invalid model type
        with self.assertRaises(ValueError):
            ModelFactory.create_model("invalid_model")
    
    def test_detection_output_format(self):
        """Test that detection output follows expected format."""
        from models.factory import ModelFactory
        
        model = ModelFactory.create_model("yolo", model_size="n")
        detections = model.detect_objects(self.test_frame)
        
        # Check return type
        self.assertIsInstance(detections, FrameDetections)
        self.assertIsInstance(detections.detections, list)
        
        # Check individual detections if any
        for detection in detections.detections:
            self.assertIsInstance(detection, DetectionResult)
            self.assertIsInstance(detection.bounding_box, BoundingBox)
            self.assertIsInstance(detection.class_name, str)

if __name__ == "__main__":
    unittest.main()
```

### Example 2: Performance Benchmarking

```python
import time
import cv2
from models.factory import ModelFactory

def benchmark_models(test_video_path, iterations=10):
    """Benchmark different models for performance comparison."""
    
    models_to_test = [
        ("YOLOv8n", "yolo", {"model_size": "n"}),
        ("YOLOv8s", "yolo", {"model_size": "s"}),
        ("DETR", "detr", {})
    ]
    
    # Load test frame
    cap = cv2.VideoCapture(test_video_path)
    ret, test_frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not load test frame!")
        return
    
    print(f"Benchmarking {len(models_to_test)} models with {iterations} iterations each\\n")
    
    results = {}
    
    for model_name, model_type, kwargs in models_to_test:
        print(f"Testing {model_name}...")
        
        # Create model
        model = ModelFactory.create_model(model_type, **kwargs)
        
        # Warm up
        for _ in range(3):
            model.detect_objects(test_frame)
        
        # Benchmark
        times = []
        detection_counts = []
        
        for i in range(iterations):
            start_time = time.time()
            detections = model.detect_objects(test_frame)
            end_time = time.time()
            
            times.append(end_time - start_time)
            detection_counts.append(len(detections.detections))
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        avg_detections = sum(detection_counts) / len(detection_counts)
        
        results[model_name] = {
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "avg_detections": avg_detections,
            "fps": 1.0 / avg_time
        }
        
        print(f"  Average time: {avg_time:.3f}s ({1.0/avg_time:.1f} FPS)")
        print(f"  Min/Max time: {min_time:.3f}s / {max_time:.3f}s")
        print(f"  Avg detections: {avg_detections:.1f}\\n")
    
    # Print summary table
    print("="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"{'Model':<12} {'Avg Time':<10} {'FPS':<8} {'Detections':<12}")
    print("-"*70)
    
    for model_name, result in results.items():
        print(f"{model_name:<12} {result['avg_time']:<10.3f} {result['fps']:<8.1f} {result['avg_detections']:<12.1f}")

# Usage
benchmark_models("test_video.mp4", iterations=20)
```

## 🎓 Learning Path

### Beginner Level
1. Start with [Basic Usage Examples](#basic-usage-examples)
2. Try the [Quick Start](#example-1-quick-start) example
3. Experiment with different models using [Model Selection](#example-2-model-selection)

### Intermediate Level
1. Learn the [Python API Examples](#python-api-examples)
2. Try [Video Processing](#example-2-video-processing)
3. Implement [Real-time Detection](#example-4-real-time-webcam-detection)

### Advanced Level
1. Create [Custom Model Implementation](#example-5-custom-model-implementation)
2. Use [Advanced Configuration](#advanced-configuration-examples)
3. Implement [Performance Monitoring](#example-2-memory-usage-monitoring)

## 📚 Additional Resources

- **Code Examples**: All examples are available in the `examples/` directory
- **API Reference**: See [API.md](API.md) for detailed class documentation
- **Model Details**: Check [MODELS.md](MODELS.md) for model specifications
- **Troubleshooting**: Visit [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues

## 🤝 Community Examples

Share your own examples by submitting a pull request! We welcome:
- Novel use cases
- Performance optimizations  
- Integration examples
- Creative applications

See [DEVELOPMENT.md](DEVELOPMENT.md) for contribution guidelines.
