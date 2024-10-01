# Copyright (c) 2024 Kinn Coelho Juliao <kinncj@gmail.com>
# All rights reserved.
#
# This software is licensed under the terms of the MIT License.
# See the LICENSE file in the project root for license terms.
import cv2
import os
import datetime
from config import TEMP_DIR

class FrameProcessor:
    def __init__(self, model, drawer):
        """
        Initializes the FrameProcessor with a model and a drawer.

        Args:
            model: The object detection model to use.
            drawer: The drawer to use for drawing detections on frames.
        """
        self.model = model
        self.drawer = drawer

    def extract_frames(self, video_path, frame_rate=1):
        """
        Extracts frames from a video at a specified frame rate.

        Args:
            video_path (str): Path to the input video file.
            frame_rate (int): Frame extraction rate.

        Returns:
            list: A list of extracted frames.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Could not open video")

        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps / frame_rate)

        for frame_count in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % interval == 0:
                frames.append(frame)

        cap.release()
        return frames

    def process_frame(self, frame, frame_idx):
        """
        Processes a single frame by performing object detection and drawing the detections.

        Args:
            frame: The frame to process.
            frame_idx (int): The index of the frame.
        """
        labels, boxes = self.model.analyze_frame(frame)
        frame_with_detections = self.drawer.draw_detections(frame, labels, boxes, self.model.id2label)
        self._save_frame(frame_with_detections, frame_idx)

    def _save_frame(self, frame, frame_idx, dir_path = TEMP_DIR):
        """
        Saves a processed frame to the temporary directory.

        Args:
            frame: The frame to save.
            frame_idx (int): The index of the frame.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(dir_path, f"detected_frame_{frame_idx}_{timestamp}.png")
        cv2.imwrite(output_path, frame)