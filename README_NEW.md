# Object Detection Application

A robust, extensible object detection application supporting multiple state-of-the-art models including YOLOv8 and DETR for real-time video analysis.

## ✨ Features

- 🤖 **Multiple Models**: YOLOv8 (nano to extra-large) and Facebook DETR
- 🎬 **Video Processing**: Complete video analysis with audio preservation
- ⚡ **Real-time Detection**: Optimized for speed and accuracy
- 🎨 **Rich Visualization**: Customizable bounding boxes and labels
- 📊 **Detailed Analytics**: Processing statistics and performance metrics
- 🔧 **Easy Configuration**: Command-line interface with sensible defaults
- 📁 **Batch Processing**: Process multiple videos efficiently
- 🏗️ **Extensible Architecture**: Clean, modular design for easy extension

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ 
- 4GB+ RAM (8GB+ recommended for larger models)
- Optional: NVIDIA GPU with CUDA support for acceleration

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kinncj/object-detection.git
   cd object-detection
   ```

2. **Set up environment** (recommended):
   ```bash
   # Using conda (recommended)
   conda env create -f environment.yml
   conda activate object-detection

   # Or using pip
   pip install -r requirements.txt
   ```

3. **Run your first detection**:
   ```bash
   python main_new.py path/to/your/video.mp4
   ```

## 📖 Usage

### Basic Examples

```bash
# Use default YOLOv8 nano model
python main_new.py video.mp4

# Use YOLOv8 small model with custom output directory
python main_new.py video.mp4 --model yolo --model-size s --output ./results

# Use DETR model with higher confidence threshold
python main_new.py video.mp4 --model detr --confidence 0.95

# Real-time display with frame saving
python main_new.py video.mp4 --display --save-images --open-result
```

### Advanced Usage

```bash
# High-accuracy processing with YOLOv8 large
python main_new.py video.mp4 \
    --model yolo \
    --model-size l \
    --confidence 0.7 \
    --output ./high_quality_results \
    --frame-rate 2 \
    --display

# Custom visualization options
python main_new.py video.mp4 \
    --show-confidence \
    --show-class-id \
    --save-images \
    --open-result
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `video_path` | Input video file path | Required |
| `--model` | Model type: `yolo`, `detr` | `yolo` |
| `--model-size` | YOLO size: `n`, `s`, `m`, `l`, `x` | `n` |
| `--confidence` | Detection confidence threshold | `0.5` (YOLO), `0.9` (DETR) |
| `--output` | Output directory | `./output` |
| `--frame-rate` | Frame extraction rate (ms) | `1` |
| `--display` | Show real-time video | `False` |
| `--save-images` | Save individual frames | `False` |
| `--open-result` | Auto-open result video | `False` |
| `--show-confidence` | Show confidence scores | `True` |
| `--show-class-id` | Show class IDs | `False` |

## 🤖 Supported Models

### YOLOv8 (Default, Recommended)
- **Sizes**: nano (n), small (s), medium (m), large (l), extra-large (x)
- **Speed**: Excellent (45-200 FPS on GPU)
- **Accuracy**: Very good to excellent
- **Use case**: Real-time applications, general-purpose detection

### DETR (Detection Transformer)
- **Architecture**: ResNet-50 + Transformer
- **Speed**: Good (5-15 FPS on GPU)  
- **Accuracy**: Excellent
- **Use case**: High-accuracy requirements, research

For detailed model comparisons, see [docs/MODELS.md](docs/MODELS.md).

## 🎯 Detected Objects

The application detects the following object classes:
- 👤 **Person**
- 📱 **Cell phone**
- 💻 **Laptop**
- 📺 **TV/Monitor** 
- ⌨️ **Keyboard**
- 🖱️ **Mouse**
- 🕐 **Clock**

## 📁 Project Structure

```
object-detection/
├── main_new.py              # Main application entry point
├── models/                  # Object detection models
│   ├── __init__.py         # Models package
│   ├── base.py             # Base classes and data structures
│   ├── detr_model.py       # DETR model implementation
│   ├── yolo_model.py       # YOLOv8 model implementation
│   └── factory.py          # Model factory pattern
├── detection/               # Detection utilities
│   └── drawer.py           # Visualization and drawing
├── processor/               # Video processing
│   └── frame_processor.py  # Frame extraction and processing
├── config/                  # Configuration
│   └── config.py           # Application configuration
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md     # System architecture
│   ├── MODELS.md           # Model documentation
│   └── ...                 # Additional documentation
├── tests/                  # Test suite
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment
└── README.md              # This file
```

## 🛠️ Development

### Code Style

This project follows PEP 8 coding standards:

```bash
# Format code
black .

# Check style
flake8 .

# Type checking
mypy .
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=models --cov=detection --cov=processor
```

### Adding New Models

1. Implement the `ObjectDetectionModel` interface
2. Add your model to the `models/` package
3. Register in `ModelFactory.SUPPORTED_MODELS`
4. Add tests and documentation

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed guidelines.

## ⚡ Performance

### Benchmarks (1080p video, RTX 3080)

| Model | FPS | Memory | Accuracy |
|-------|-----|--------|----------|
| YOLOv8n | 180 | 2GB | Good |
| YOLOv8s | 150 | 3GB | Very Good |
| YOLOv8m | 120 | 4GB | Excellent |
| YOLOv8l | 100 | 5GB | Excellent |
| YOLOv8x | 80 | 6GB | Outstanding |
| DETR | 15 | 8GB | Outstanding |

### Optimization Tips

- Use GPU acceleration when available
- Choose appropriate model size for your use case
- Process videos in batches for better throughput
- Adjust confidence thresholds based on requirements

## 🔧 Configuration

### Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
export OMP_NUM_THREADS=4       # CPU thread limit
```

### Custom Configuration

```python
# config/config.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.5
OUTPUT_FORMAT = "mp4"
```

## 📊 Output

The application generates:

1. **Processed Video**: Video with detection overlays
2. **Audio-preserved Video**: Original audio + processed video
3. **Individual Frames** (optional): Annotated frame images
4. **Statistics**: Processing metrics and detection counts

Example output structure:
```
output/
├── detected_video.mp4           # Processed video
├── detected_video_with_audio.mp4 # With original audio
├── frame_0000.png              # Individual frames (if enabled)
├── frame_0001.png
└── ...
```

## 🐛 Troubleshooting

### Common Issues

**"Could not open video"**:
- Check file path and permissions
- Verify video format is supported (mp4, avi, mov, etc.)

**Slow performance**:
- Enable GPU acceleration
- Use smaller model sizes (YOLOv8n)
- Reduce frame extraction rate

**Out of memory errors**:
- Use CPU instead of GPU
- Process shorter video segments
- Reduce model size

For more solutions, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## 📚 Documentation

- [**Architecture Guide**](docs/ARCHITECTURE.md) - System design and patterns
- [**Model Documentation**](docs/MODELS.md) - Detailed model information
- [**API Reference**](docs/API.md) - Class and function documentation
- [**Examples**](docs/EXAMPLES.md) - Usage examples and tutorials
- [**Development Guide**](docs/DEVELOPMENT.md) - Contributing guidelines

## 🤝 Contributing

We welcome contributions! Please see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for:

- Development setup
- Coding standards
- Testing requirements
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [Facebook Research](https://github.com/facebookresearch/detr) for DETR
- [Hugging Face](https://huggingface.co/) for model hosting and transformers library
- [OpenCV](https://opencv.org/) for computer vision utilities

## 📞 Support

- 📖 Check the [documentation](docs/)
- 🐛 Report issues on [GitHub Issues](https://github.com/kinncj/object-detection/issues)
- 💬 Ask questions in [Discussions](https://github.com/kinncj/object-detection/discussions)
- 📧 Contact: kinncj@gmail.com

---

**Made with ❤️ by the Object Detection Team**
