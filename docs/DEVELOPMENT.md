# Development Guide

## Overview

This guide covers setting up the development environment, understanding the codebase architecture, and contributing to the Object Detection Application.

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+
- Git
- Conda or Miniconda
- CUDA toolkit (optional, for GPU acceleration)

### Environment Setup

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd object-detection
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Activate Environment**:
   ```bash
   conda activate object-detection
   ```

3. **Verify Installation**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import ultralytics; print('YOLO: OK')"
   python -c "import transformers; print('DETR: OK')"
   ```

### Development Dependencies

For development, install additional tools:

```bash
pip install -r requirements-dev.txt
# Or manually:
pip install pytest black flake8 mypy pre-commit jupyter
```

## 🏗️ Architecture Overview

### Project Structure

```
object-detection/
├── models/                 # Core model implementations
│   ├── __init__.py
│   ├── base.py            # Abstract base classes and DTOs
│   ├── yolo_model.py      # YOLOv8 implementation
│   ├── detr_model.py      # DETR implementation
│   └── factory.py         # Model factory pattern
├── detection/             # Detection utilities
│   ├── __init__.py
│   └── drawer.py          # Visualization and annotation
├── processor/             # Video processing
│   ├── __init__.py
│   └── frame_processor.py # Frame-by-frame processing
├── config/                # Configuration management
│   ├── __init__.py
│   └── config.py          # Application settings
├── tests/                 # Test suite
│   ├── __init__.py
│   ├── test_architecture.py
│   ├── test_simple.py
│   ├── test_yolo_integration.py
│   └── test_video.mp4
├── docs/                  # Documentation
│   ├── API.md
│   ├── MODELS.md
│   ├── ARCHITECTURE.md
│   └── DEVELOPMENT.md (this file)
├── main.py               # Main application entry point
├── requirements.txt      # Production dependencies
├── setup.sh             # Environment setup script
└── README.md            # Project overview
```

### Design Patterns

1. **Factory Pattern**: `ModelFactory` for creating models
2. **Strategy Pattern**: `ObjectDetectionModel` interface with different implementations
3. **Data Transfer Objects**: `BoundingBox`, `DetectionResult`, `FrameDetections`
4. **Command Pattern**: CLI interface in `main.py`

### Core Components

#### 1. Model Abstraction (`models/base.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class BoundingBox:
    """Represents a bounding box with validation."""
    x1: float
    y1: float 
    x2: float
    y2: float
    confidence: float
    
    def __post_init__(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Invalid bounding box coordinates")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")

class ObjectDetectionModel(ABC):
    """Abstract base class for all detection models."""
    
    @abstractmethod
    def detect_objects(self, frame: np.ndarray) -> FrameDetections:
        """Detect objects in a single frame."""
        pass
```

#### 2. Model Factory (`models/factory.py`)

```python
class ModelFactory:
    """Centralized model creation with registry pattern."""
    
    _models = {
        "yolo": YOLOv8Model,
        "detr": DETRModel
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> ObjectDetectionModel:
        """Create model instance with configuration."""
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return cls._models[model_type](**kwargs)
    
    @classmethod
    def register_model(cls, name: str, model_class: type):
        """Register new model type."""
        cls._models[name] = model_class
```

#### 3. YOLO Implementation (`models/yolo_model.py`)

```python
from ultralytics import YOLO
from .base import ObjectDetectionModel, FrameDetections, DetectionResult, BoundingBox

class YOLOv8Model(ObjectDetectionModel):
    """YOLOv8 model implementation using Ultralytics."""
    
    def __init__(self, model_size: str = "n", confidence_threshold: float = 0.5):
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(f"yolov8{model_size}.pt")
    
    def detect_objects(self, frame: np.ndarray) -> FrameDetections:
        """Detect objects using YOLOv8."""
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        detections = []
        for result in results:
            for box in result.boxes:
                # Convert to our DTO format
                detection = self._convert_detection(box)
                detections.append(detection)
        
        return FrameDetections(detections=detections)
```

## 🧪 Testing Strategy

### Test Structure

```
tests/
├── test_architecture.py    # Core architecture tests
├── test_simple.py         # Basic functionality tests  
├── test_yolo_integration.py # YOLO model integration tests
└── test_video.mp4         # Sample test video
```

### Running Tests

```bash
# Run all tests
cd tests/
python test_architecture.py
python test_simple.py
python test_yolo_integration.py

# Run with pytest (if available)
python -m pytest tests/ -v

# Test specific functionality
python test_simple.py test_video.mp4
```

### Test Examples

#### Architecture Tests

```python
# tests/test_architecture.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_bounding_box_validation():
    """Test BoundingBox validation."""
    from models.base import BoundingBox
    
    # Valid bounding box
    bbox = BoundingBox(10, 10, 20, 20, 0.8)
    assert bbox.width == 10
    assert bbox.height == 10
    
    # Invalid coordinates should raise error
    try:
        BoundingBox(20, 20, 10, 10, 0.8)  # x2 < x1
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_model_factory():
    """Test ModelFactory functionality."""
    from models.factory import ModelFactory
    
    # Test model creation
    model = ModelFactory.create_model("yolo", model_size="n")
    assert model is not None
    
    # Test invalid model type
    try:
        ModelFactory.create_model("invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
```

#### Integration Tests

```python
# tests/test_yolo_integration.py
def test_yolo_detection():
    """Test YOLO model detection on sample frame."""
    import cv2
    import numpy as np
    from models.factory import ModelFactory
    
    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create model and run detection
    model = ModelFactory.create_model("yolo", model_size="n")
    detections = model.detect_objects(frame)
    
    # Verify response structure
    assert hasattr(detections, 'detections')
    assert isinstance(detections.detections, list)
```

### Performance Testing

```python
import time
import cv2
from models.factory import ModelFactory

def benchmark_model(model_type: str, video_path: str):
    """Benchmark model performance."""
    model = ModelFactory.create_model(model_type)
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    total_time = 0
    total_detections = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        detections = model.detect_objects(frame)
        inference_time = time.time() - start_time
        
        frame_count += 1
        total_time += inference_time
        total_detections += len(detections.detections)
    
    cap.release()
    
    # Calculate metrics
    fps = frame_count / total_time
    avg_detections = total_detections / frame_count
    
    print(f"Model: {model_type}")
    print(f"FPS: {fps:.1f}")
    print(f"Avg detections/frame: {avg_detections:.2f}")
    print(f"Total detections: {total_detections}")
```

## 🔧 Adding New Models

### Step 1: Implement Model Class

```python
# models/my_custom_model.py
from .base import ObjectDetectionModel, FrameDetections, DetectionResult, BoundingBox
import numpy as np

class MyCustomModel(ObjectDetectionModel):
    """Custom model implementation."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        # Initialize your model here
        
    def detect_objects(self, frame: np.ndarray) -> FrameDetections:
        """Implement your detection logic."""
        detections = []
        
        # Your detection logic here
        # For each detected object, create DetectionResult
        
        return FrameDetections(detections=detections)
```

### Step 2: Register Model

```python
# models/factory.py (add to existing factory)
from .my_custom_model import MyCustomModel

class ModelFactory:
    _models = {
        "yolo": YOLOv8Model,
        "detr": DETRModel,
        "custom": MyCustomModel  # Add your model
    }
```

### Step 3: Add Tests

```python
# tests/test_custom_model.py
def test_custom_model():
    """Test custom model implementation."""
    from models.factory import ModelFactory
    import numpy as np
    
    model = ModelFactory.create_model("custom")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    detections = model.detect_objects(frame)
    assert isinstance(detections.detections, list)
```

### Step 4: Update Documentation

Update the following files:
- `docs/MODELS.md`: Add model specifications and performance
- `docs/API.md`: Add API examples
- `README.md`: Update model comparison table

## 🎨 Code Style

### Python Style Guide

Follow PEP 8 with these tools:

```bash
# Code formatting
black .

# Linting
flake8 .

# Type checking
mypy models/ detection/ processor/
```

### Pre-commit Hooks

Set up pre-commit hooks for automatic formatting:

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

Example `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.950
    hooks:
      - id: mypy
```

### Documentation Standards

- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Add type hints to all public functions
- **Comments**: Explain complex logic, not obvious code
- **README Updates**: Keep README.md updated with new features

Example docstring format:

```python
def detect_objects(self, frame: np.ndarray) -> FrameDetections:
    """Detect objects in a video frame.
    
    Args:
        frame: Input image as BGR numpy array with shape (H, W, 3)
        
    Returns:
        FrameDetections containing list of detected objects with bounding boxes
        
    Raises:
        ValueError: If frame is not a valid image array
        RuntimeError: If model inference fails
        
    Example:
        >>> model = ModelFactory.create_model("yolo")
        >>> frame = cv2.imread("image.jpg")
        >>> detections = model.detect_objects(frame)
        >>> print(f"Found {len(detections.detections)} objects")
    """
```

## 🐛 Debugging

### Common Issues

1. **Import Errors**:
   ```bash
   # Check Python path
   python -c "import sys; print(sys.path)"
   
   # Add project root to path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Model Loading Issues**:
   ```python
   # Check model cache
   import torch
   print(torch.hub.get_dir())
   
   # Clear cache if needed
   torch.hub._get_cache_dir()
   ```

3. **GPU Issues**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA devices: {torch.cuda.device_count()}")
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# In your code
logger = logging.getLogger(__name__)
logger.debug("Debug message")
```

### Performance Profiling

```python
import cProfile
import pstats

def profile_detection():
    # Your detection code here
    pass

# Profile the function
cProfile.run('profile_detection()', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(20)
```

## 📊 Performance Monitoring

### Metrics Collection

```python
import time
import psutil
import torch
from dataclasses import dataclass
from typing import List

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    inference_time: float
    fps: float
    cpu_usage: float
    gpu_memory_mb: float
    detection_count: int

def collect_metrics(model, frame) -> PerformanceMetrics:
    """Collect performance metrics during inference."""
    start_time = time.time()
    
    # Run detection
    detections = model.detect_objects(frame)
    
    # Calculate metrics
    inference_time = time.time() - start_time
    fps = 1.0 / inference_time if inference_time > 0 else 0
    cpu_usage = psutil.cpu_percent()
    
    gpu_memory_mb = 0
    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.memory_allocated() / 1024**2
    
    return PerformanceMetrics(
        inference_time=inference_time,
        fps=fps,
        cpu_usage=cpu_usage,
        gpu_memory_mb=gpu_memory_mb,
        detection_count=len(detections.detections)
    )
```

## 🚀 Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Expose port for web service (if applicable)
EXPOSE 8000

# Default command
CMD ["python", "main.py", "--help"]
```

### Production Considerations

1. **Model Caching**: Pre-download models in container
2. **Memory Management**: Monitor GPU memory usage
3. **Batch Processing**: Process multiple videos efficiently
4. **Error Handling**: Graceful failure handling
5. **Logging**: Structured logging for monitoring
6. **Health Checks**: Endpoint for service health

## 🤝 Contributing

### Contribution Workflow

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Implement** your changes with tests
4. **Test** thoroughly: Run all tests and add new ones
5. **Document**: Update relevant documentation
6. **Commit**: Use clear, descriptive commit messages
7. **Push**: Push to your feature branch
8. **Submit**: Create a Pull Request

### Commit Message Format

```
type(scope): brief description

Longer description if needed.

Closes #issue-number
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(models): add support for YOLOv9 architecture

- Implement YOLOv9Model class with new architecture
- Add performance benchmarks and tests
- Update model factory and documentation

Closes #123
```

### Review Process

All contributions go through code review:

1. **Automated Checks**: CI/CD pipeline runs tests and linting
2. **Manual Review**: Code review by maintainers
3. **Testing**: Verify functionality on different systems
4. **Documentation**: Ensure docs are updated
5. **Merge**: Approved PRs are merged to main

## 📚 Resources

### Learning Resources

- **YOLO Papers**: [YOLOv8 Documentation](https://docs.ultralytics.com/)
- **DETR Paper**: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- **Computer Vision**: [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
- **PyTorch**: [PyTorch Tutorials](https://pytorch.org/tutorials/)

### Development Tools

- **VS Code Extensions**: Python, Pylance, GitLens
- **Git GUI**: GitKraken, SourceTree
- **Debugging**: pdb, ipdb, VS Code debugger
- **Profiling**: cProfile, line_profiler, py-spy

### Community

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: GitHub Discussions for questions
- **Discord/Slack**: Real-time community chat (if available)

---

Happy coding! 🎯 Feel free to reach out if you have questions or need help with development.
