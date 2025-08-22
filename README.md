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

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/kinncj/object-detection.git
cd object-detection

# Create and activate conda environment
conda env create -f environment.yml
conda activate object-detection
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/kinncj/object-detection.git
cd object-detection

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
```

## 📊 Model Comparison

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| YOLOv8n | ~6MB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Real-time, mobile |
| YOLOv8s | ~22MB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Balanced performance |
| YOLOv8m | ~52MB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | High accuracy |
| YOLOv8l | ~87MB | ⚡⚡ | ⭐⭐⭐⭐⭐ | Production accuracy |
| YOLOv8x | ~136MB | ⚡ | ⭐⭐⭐⭐⭐ | Maximum accuracy |
| DETR | ~159MB | ⚡ | ⭐⭐⭐⭐ | Research, transformers |

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

The application detects the following object classes:
- Person
- Laptop
- Cell phone
- TV/Monitor
- Keyboard
- Mouse
- Clock

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

```bash
# Run the test suite
python -m pytest tests/

# Test specific model
python test_simple.py
```

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
