# Object Detection in Video

## Overview

This project implements an object detection system using the DETR (DEtection TRansformer) model from Hugging Face's Transformers library. It processes video frames to detect specific restricted classes of objects and draws bounding boxes around them.

## Features

- Detects specific restricted objects: person, cell phone, laptop, TV, keyboard, and mouse.
- Draws bounding boxes around detected objects with different colors.
- Saves processed frames with detections to a temporary directory.

## Requirements

To run this project, you'll need the following Python packages:

- `torch`
- `transformers`
- `Pillow`
- `opencv-python`
- `numpy`

## Setup Instructions

### Installing FFmpeg

If you are using macOS, you can install FFmpeg using Homebrew. Open your terminal and run the following command:

```bash
brew install ffmpeg
```

### Creating a Conda Environment

1. **Create a new Conda environment:**

   Open your terminal (or Anaconda Prompt) and run the following command:

   ```bash
   conda create --name video_analyzer python=3.11
   ```

   Replace `video_analyzer` with your preferred environment name if needed.

2. **Activate the Conda environment:**

   ```bash
   conda activate video_analyzer
   ```

3. **Install required packages:**

   Once the environment is activated, install the required packages using the `requirements.txt` file provided in the project:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Script

To run the object detection on a video file, execute the script from the command line:

```bash
python main.py <path_to_video_file> --frame_rate <frame_rate>
```

#### Arguments

- `<path_to_video_file>`: The path to the input video file.
- `--frame_rate`: (Optional) The rate at which frames are extracted from the video. The default value is 1 frame per second.

### Example

```bash
python main.py input_video.mp4 --frame_rate 2
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

### Summary of Changes
- Added a section for installing FFmpeg with the Homebrew command at the beginning of the setup instructions. This ensures users have all necessary dependencies for video processing.