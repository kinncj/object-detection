# Object Detection in Video

## Overview

This project implements an object detection system supporting both DETR (DEtection TRansformer) and YOLOv8 models. It processes video frames to detect specific restricted classes of objects and draws bounding boxes around them.

## Features

- **Multiple Model Support**: Choose between DETR (Facebook) and YOLOv8 (Ultralytics) models
- **Restricted Object Detection**: Detects person, cell phone, laptop, TV, keyboard, mouse, and clock
- **Video Processing**: Processes video frames with customizable frame rates
- **Visualization**: Draws colored bounding boxes around detected objects
- **Output Options**: Save processed frames and/or compiled videos

## Quick Start

### 1. Install Conda (if not already installed)

Choose one of these options:
- **Miniconda (recommended)**: [Download here](https://docs.conda.io/en/latest/miniconda.html)
- **Anaconda**: [Download here](https://www.anaconda.com/products/distribution)
- **Mamba (faster alternative)**: [Install guide](https://mamba.readthedocs.io/)

### 2. Set up the environment

```bash
# Clone the repository
git clone <your-repo-url>
cd object-detection

# Run the setup script
./setup.sh
```

### 3. Activate the environment

```bash
conda activate object-detection
```

### 4. Run object detection

```bash
# Using DETR model (default)
python main.py path/to/your/video.mp4

# Using YOLOv8 nano model (fastest)
python main.py path/to/your/video.mp4 --model yolo --model_size n

# Using YOLOv8 medium model (balanced speed/accuracy)
python main.py path/to/your/video.mp4 --model yolo --model_size m

# Using YOLOv8 extra large model (most accurate)
python main.py path/to/your/video.mp4 --model yolo --model_size x

# With additional options
python main.py path/to/your/video.mp4 \
  --model yolo \
  --model_size s \
  --frame_rate 2 \
  --display_video true \
  --store_video_path output/ \
  --store_image_path frames/
```

## Manual Setup (Alternative)

If you prefer to set up manually:

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate object-detection
```

## Command Line Options

The main script supports the following arguments:

```bash
python main.py <video_path> [OPTIONS]

Required:
  video_path                    Path to the input video file

Optional:
  --model {detr,yolo}           Model to use (default: detr)
  --model_size {n,s,m,l,x}      YOLOv8 model size (default: n, ignored for DETR)
                                n=nano, s=small, m=medium, l=large, x=extra large
  --frame_rate INT              Frame extraction rate per ms (default: 1)
  --display_video BOOL          Display video after processing (default: False)
  --store_video_path STR        Path to save processed video
  --store_image_path STR        Path to save processed frames
```

## Development Commands

Use the provided Makefile for common development tasks:

```bash
make help               # Show all available commands
make test               # Run tests
make format             # Format code with black
make lint               # Run linting
make clean              # Clean up temporary files
make demo               # Run demo with test video

# Run with different models
make run-detr VIDEO_PATH=video.mp4              # DETR model
make run-yolo VIDEO_PATH=video.mp4              # YOLOv8n (nano)
make run-yolo VIDEO_PATH=video.mp4 MODEL_SIZE=s # YOLOv8s (small)
make run-yolo-medium VIDEO_PATH=video.mp4       # YOLOv8m (medium)
```

## Models Supported

### DETR (Default)
- **Source**: Facebook Research
- **Model**: `facebook/detr-resnet-50`
- **Strengths**: High accuracy, research-grade
- **Use case**: When accuracy is more important than speed

### YOLOv8
- **Source**: Ultralytics (via Hugging Face)
- **Models Available**: 
  - YOLOv8n (nano): Fastest, 6.2M parameters
  - YOLOv8s (small): 11.2M parameters  
  - YOLOv8m (medium): 25.9M parameters
  - YOLOv8l (large): 43.7M parameters
  - YOLOv8x (extra large): 68.2M parameters, most accurate
- **Strengths**: Fast inference, real-time capable, multiple size options
- **Use case**: When speed is important, or when you need to balance speed vs accuracy

## Usage

### Running the Script

To run the object detection on a video file, execute the script from the command line:

```bash
python main.py <path_to_video_file> --frame_rate=<frame_rate> --display_video=<display_video> --store_image_path=<image_path> --store_video_path=<video_path>
```

#### Arguments

- `<path_to_video_file>`: The path to the input video file.
- `--frame_rate`: (Optional) The rate at which frames are extracted from the video. The default value is 1 frame per second.

### Example

```bash
python main.py input_video.mp4
python main.py input_video.mp4 --frame_rate=1
python main.py input_video.mp4 --frame_rate=1 --display_video=True
python main.py input_video.mp4 --frame_rate=1 --store_image_path=/tmp/ai_files
python main.py input_video.mp4 --frame_rate=1 --store_video_path=/tmp/ai_files
python main.py input_video.mp4 --frame_rate=1 --store_image_path=/tmp/ai_files --display_video=True
python main.py ~/Movies/video_for_ai2.mp4 --frame_rate=1 --store_video_path=/tmp/ai_files  --display_video=True --store_image_path=/tmp/ai_files
python -m unittest discover -s tests
```

This command will process `input_video.mp4`, extracting 2 frames per second. The output frames with detected objects will be saved in the temporary directory `/tmp/ai_files`.

### Output

Processed frames will be saved in the specified temporary directory with filenames in the format `detected_frame_<index>_<timestamp>.png`, where `<index>` is the frame index and `<timestamp>` indicates when the frame was processed.

## Code Structure

- `detection_model.py`: Contains the implementation of the DETR model and other related classes for object detection.
- `frame_processor.py`: Handles the extraction and processing of video frames.
- `detection_drawer.py`: Manages the drawing of bounding boxes and labels on the video frames.
- `main.py`: The entry point of the application.

## Author

Kinn Coelho Juliao <kinncj@gmail.com>

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

