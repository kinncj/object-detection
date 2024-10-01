# Copyright (c) 2024 Kinn Coelho Juliao <kinncj@gmail.com>
# All rights reserved.
#
# This software is licensed under the terms of the MIT License.
# See the LICENSE file in the project root for license terms.
import argparse
from detection.model import DETRModel
from detection.drawer import DetectionDrawer
from processor.frame_processor import FrameProcessor

def main(video_path, frame_rate):
    """
    Main function to perform object detection on a video.

    Args:
        video_path (str): Path to the input video file.
        frame_rate (int): Frame extraction rate.
    """
    model = DETRModel()
    drawer = DetectionDrawer()
    processor = FrameProcessor(model, drawer)

    frames = processor.extract_frames(video_path, frame_rate)
    for idx, frame in enumerate(frames):
        processor.process_frame(frame, idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection in Video")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--frame_rate", type=int, default=1, help="Frame extraction rate.")
    args = parser.parse_args()

    main(args.video_path, args.frame_rate)