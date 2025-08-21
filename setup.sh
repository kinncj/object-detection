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

# Verify installation
echo "✅ Verifying installation..."
python -c "
try:
    import torch
    import transformers
    import cv2
    import ultralytics
    print('✅ All core dependencies imported successfully')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "   conda activate $ENV_NAME"
echo ""
echo "To run the object detection:"
echo "   python main.py path/to/video.mp4 --model detr    # Use DETR model"
echo "   python main.py path/to/video.mp4 --model yolo    # Use YOLOv8 model"
echo ""
echo "To run tests:"
echo "   pytest"
echo ""
echo "To update the environment:"
echo "   conda env update -f environment.yml"
echo ""
echo "To export current environment:"
echo "   conda env export > environment.yml"
