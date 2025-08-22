#!/bin/bash

# Object Detection Environment Setup Script
# This script sets up the development environment using conda

set -e  # Exit on any error

echo "🚀 Setting up Object Detection Environment with Conda..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ conda is not installed. Please install it first:"
    echo "   Option 1 - Miniconda (recommended): https://docs.conda.io/en/latest/miniconda.html"
    echo "   Option 2 - Anaconda: https://www.anaconda.com/products/distribution"
    echo "   Option 3 - Mamba (faster): https://mamba.readthedocs.io/"
    exit 1
fi

# Initialize conda for the current shell if needed
if ! conda info &> /dev/null; then
    echo "� Initializing conda for your shell..."
    conda init "$(basename "$SHELL")"
    echo "⚠️  Please restart your terminal and run this script again."
    exit 0
fi

ENV_NAME="object-detection"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "🔄 Environment '${ENV_NAME}' already exists. Updating..."
    conda env update -n $ENV_NAME -f environment.yml
else
    echo "🏗️  Creating new conda environment '${ENV_NAME}'..."
    conda env create -f environment.yml
fi

echo "🔄 Activating environment..."
# Note: This won't work in the script context, but we'll tell the user
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install critical dependencies that work better via conda
echo "📦 Installing critical dependencies via conda for best compatibility..."
conda install -n $ENV_NAME -c conda-forge decorator=4.4.2 moviepy=1.0.3 -y

# Verify installation
echo "✅ Verifying installation..."
python -c "
try:
    import torch
    import transformers
    import cv2
    import ultralytics
    from moviepy.editor import VideoFileClip
    print('✅ All core dependencies imported successfully')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print('✅ moviepy.editor import successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🎯 To activate the environment:"
echo "   conda activate $ENV_NAME"
echo ""
echo "🚀 Quick start commands:"
echo "   # Test the installation"
echo "   python main.py tests/test_video.mp4 --display"
echo ""
echo "   # Basic detection with YOLOv8 nano (fastest)"
echo "   python main.py path/to/video.mp4"
echo ""
echo "   # High-accuracy detection with YOLOv8 large"
echo "   python main.py path/to/video.mp4 --model yolo --model-size l"
echo ""
echo "   # Comprehensive detection with DETR"
echo "   python main.py path/to/video.mp4 --model detr --confidence 0.8"
echo ""
echo "   # Real-time display with info overlay"
echo "   python main.py path/to/video.mp4 --display --info"
echo ""
echo "🧪 Run tests:"
echo "   python -m pytest tests/ -v    # Run all tests"
echo "   python -m pytest tests/test_base.py tests/test_config.py tests/test_drawer.py tests/test_factory.py -v    # Core tests"
echo ""
echo "📊 Performance expectations (based on 702-frame test video):"
echo "   YOLOv8n: ~19 FPS, 1.2 detections/frame (real-time)"
echo "   DETR:    ~9 FPS, 4.2 detections/frame (comprehensive)"
echo ""
echo "💡 Tips:"
echo "   - Use YOLOv8n for real-time applications"
echo "   - Use DETR for detailed analysis and research"
echo "   - GPU acceleration detected: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "⚙️  Environment management:"
echo "   conda env update -f environment.yml     # Update environment"
echo "   conda env export > environment.yml      # Export current environment"
