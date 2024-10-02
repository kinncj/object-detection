# Copyright (c) 2024 Kinn Coelho Juliao <kinncj@gmail.com>
# All rights reserved.
#
# This software is licensed under the terms of the MIT License.
# See the LICENSE file in the project root for license terms.
import argparse
import os
import platform
import subprocess

from detection.model import DETRModel
from detection.drawer import DetectionDrawer
from processor.frame_processor import FrameProcessor

def main(video_path, frame_rate, display_video, image_path, store_video_path):
    """
    Main function to perform object detection on a video.

    Args:
        video_path (str): Path to the input video file.
        frame_rate (int): Frame extraction rate.
        display_video (bool): Whether to display the video.
        image_path (str): Path to save the images
        store_video_path (str): Path to save the video.
    """
    model = DETRModel()
    drawer = DetectionDrawer()
    processor = FrameProcessor(model, drawer)

    fps, frames, audio = processor.extract_video_fragments(video_path, frame_rate)
    for idx, frame in enumerate(frames):
        processor.process_frame(frame, idx, display_video, image_path)

    output_video_path, output_video_and_audio_path = processor.compile_video(frames, store_video_path, fps, audio)

    if display_video and output_video_path:
        _open_video(output_video_path, output_video_and_audio_path)

def _open_video(output_video_path, output_video_and_audio_path):
    print(f"Opening video: {output_video_path} and {output_video_and_audio_path}")
    # Detect the operating system
    if platform.system() == 'Windows':
        print("Windows")
        os.startfile(output_video_path)  # Windows
        os.startfile(output_video_and_audio_path)  # Windows
    elif platform.system() == 'Darwin':  # macOS
        print("macOS")
        subprocess.run(['open', output_video_path])
        subprocess.run(['open', output_video_and_audio_path])
    else:  # Assume Linux or other Unix-like OS
        print("Linux")
        subprocess.run(['xdg-open', output_video_path])
        subprocess.run(['xdg-open', output_video_and_audio_path])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection in Video")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--frame_rate", type=int, default=1, help="Frame extraction rate per ms.")
    parser.add_argument("--display_video", type=bool, default=False, help="Display Video.")
    parser.add_argument("--store_video_path", type=str, default=None, help="Save Final Video.")
    parser.add_argument("--store_image_path", type=str, default=None, help="Save Images.")
    args = parser.parse_args()

    main(args.video_path, args.frame_rate, args.display_video,  args.store_image_path, args.store_video_path)