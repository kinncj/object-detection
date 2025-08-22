# Object Detection Application

A modern, high-performance object detection application supporting multiple state-of-the-art models including YOLOv8 and DETR. This application processes video files to detect and annotate objects with bounding boxes, providing real-time visualization and comprehensive analytics.

## 🚀 Features

- **Multiple Model Support**: YOLOv8 (nano to extra-large) and DETR models
- **Real-time Processing**: Efficient video frame processing with live visualization
- **Flexible Output**: Generate annotated videos with customizable overlays
- **Production Ready**: Clean architecture with proper error handling and logging
- **Easy to Use**: Simple CLI interface with sensible defaults

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for best performance)
- 4GB+ RAM
- OpenCV-compatible video formats (MP4, AVI, MOV, etc.)

## 🛠️ Installation

### Quick Setup (Recommended)

```bash
# Clone the repository
git clone <repository_url>
cd object-detection

# Run automated setup script
chmod +x setup.sh
./setup.sh

# Activate the environment  
conda activate object-detection

# Test the installation
python main.py tests/test_video.mp4 --display
```

### Manual Setup

#### Option 1: Using Conda (Recommended)

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate object-detection
```

#### Option 2: Using pip

```bash
# Create virtual environment
python -m venv object-detection-env
source object-detection-env/bin/activate  # On Windows: object-detection-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

### Basic Usage

```bash
# Detect objects in a video using YOLOv8 nano (fastest)
python main.py your_video.mp4

# Use a larger, more accurate model
python main.py your_video.mp4 --model yolo --model-size l

# Use DETR model with custom confidence threshold
python main.py your_video.mp4 --model detr --confidence 0.8

# Display video during processing
python main.py your_video.mp4 --display --info
```

### Advanced Examples

```bash
# High-accuracy detection with custom output directory
python main.py video.mp4 --model yolo --model-size x --confidence 0.6 --output ./results/

# Real-time display with frame information overlay
python main.py video.mp4 --display --info --output ./live_demo/

# Process with specific model and output folder structure
python main.py /Users/kinncj/Movies/iPhoneDeskView.mp4 --model yolo --model-size n --output ./output/yolo_results
python main.py /Users/kinncj/Movies/iPhoneDeskView.mp4 --model detr --output ./output/detr_results
```

## 📊 Real-World Performance

Testing with actual video data (`iPhoneDeskView.mp4` - 702 frames, 21.79s duration):

### YOLOv8 Nano Results
- **Processing Time**: 36.15s (19.4 FPS)
- **Total Detections**: 838 objects
- **Average per Frame**: 1.19 detections
- **Frame Processing**: 0.051s/frame
- **Objects Detected**: person, cell phone, laptop, clock
- **Use Case**: Perfect for real-time applications

### DETR Results  
- **Processing Time**: 75.15s (9.3 FPS)
- **Total Detections**: 2,964 objects
- **Average per Frame**: 4.22 detections
- **Frame Processing**: 0.107s/frame
- **Objects Detected**: More comprehensive detection across all classes
- **Use Case**: Best for detailed analysis and research

## 📊 Model Comparison

| Model | Size | Speed (FPS) | Accuracy | Detections/Frame | Use Case |
|-------|------|-------------|----------|------------------|----------|
| YOLOv8n | ~6MB | 19.4 FPS | ⭐⭐⭐ | 1.19 avg | Real-time, mobile |
| YOLOv8s | ~22MB | ~15 FPS | ⭐⭐⭐⭐ | ~1.5 avg | Balanced performance |
| YOLOv8m | ~52MB | ~12 FPS | ⭐⭐⭐⭐⭐ | ~2.0 avg | High accuracy |
| YOLOv8l | ~87MB | ~8 FPS | ⭐⭐⭐⭐⭐ | ~2.5 avg | Production accuracy |
| YOLOv8x | ~136MB | ~6 FPS | ⭐⭐⭐⭐⭐ | ~3.0 avg | Maximum accuracy |
| DETR | ~159MB | 9.3 FPS | ⭐⭐⭐⭐ | 4.22 avg | Research, comprehensive |

*Performance data based on 702-frame test video with mixed content (person, electronics, furniture)*

## 🎮 Command Line Options

```
python main.py <video_path> [options]

Required Arguments:
  video_path              Path to the input video file

Model Configuration:
  --model {yolo,detr}     Object detection model (default: yolo)
  --model-size {n,s,m,l,x} Model size for YOLO (default: n)
  --confidence FLOAT      Confidence threshold 0-1 (default: 0.5)

Output Configuration:
  --output DIR           Output directory (default: ./output)
  --display              Display video during processing
  --info                 Show frame info overlay
```

## 🏗️ Architecture

The application follows a clean, modular architecture:

```
object-detection/
├── models/                 # Model implementations
│   ├── base.py            # Abstract base classes and DTOs
│   ├── detr_model.py      # DETR implementation
│   ├── yolo_model.py      # YOLOv8 implementation
│   └── factory.py         # Model factory
├── detection/             # Detection utilities
│   └── drawer.py          # Visualization and drawing
├── config/                # Configuration
│   └── config.py          # Settings and constants
├── docs/                  # Documentation
└── tests/                 # Test suite
```

## 🔧 Configuration

### Supported Object Classes

The application detects the following 80 COCO dataset classes:
- **People**: person
- **Electronics**: laptop, cell phone, tv, keyboard, mouse
- **Furniture**: chair, couch, bed, dining table
- **Transportation**: car, motorcycle, airplane, bus, train, truck, boat
- **Animals**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Sports**: tennis racket, frisbee, skis, snowboard, sports ball
- **Kitchen**: bottle, wine glass, cup, fork, knife, spoon, bowl
- **Food**: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
- **Household**: clock, vase, scissors, teddy bear, hair drier, toothbrush
- And many more...

### Output Structure

Processed videos are saved with clear organization:

```
output/
├── yolo_results/
│   └── detected_[original_filename].mp4
├── detr_results/
│   └── detected_[original_filename].mp4
└── [custom_folder]/
    └── detected_[original_filename].mp4
```

### Environment Variables

```bash
# Optional: Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Optional: Set torch device
export TORCH_DEVICE=cuda:0
```

## 📈 Performance Tips

1. **GPU Acceleration**: Ensure CUDA is properly installed for GPU acceleration
2. **Model Selection**: Use YOLOv8n for real-time applications, YOLOv8l+ for accuracy
3. **Batch Processing**: Process multiple videos sequentially for efficiency
4. **Memory Management**: Close display windows when processing large videos

## 🧪 Testing

The application includes a comprehensive test suite to ensure reliability:

```bash
# Run all tests from the tests/ directory
cd tests/
python test_architecture.py    # Test core architecture and DTOs
python test_simple.py         # Basic functionality test
python test_yolo_integration.py  # YOLO model integration test

# Test with sample video
python test_simple.py test_video.mp4

# Run from project root (alternative)
python -m pytest tests/ -v
```

### Test Coverage
- **Architecture Tests**: Validate DTOs, model factory, and base classes
- **Integration Tests**: End-to-end model processing
- **Performance Tests**: Real video processing validation
- **Sample Data**: Included test video for validation

## 📚 Documentation

- [API Documentation](docs/API.md)
- [Model Details](docs/MODELS.md)
- [Architecture Guide](docs/ARCHITECTURE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [Hugging Face](https://huggingface.co/) for DETR model and transformers
- [OpenCV](https://opencv.org/) for computer vision utilities

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/kinncj/object-detection/issues)
- **Documentation**: [Project Wiki](https://github.com/kinncj/object-detection/wiki)
- **Email**: kinncj@gmail.com

---

⭐ **Star this repository if you found it helpful!**
