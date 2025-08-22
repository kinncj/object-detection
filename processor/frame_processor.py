"""
Frame processor for object detection video processing.

This module provides the FrameProcessor class for processing video frames
with object detection models and drawing detection results.
"""
# Copyright (c) 2024 Kinn Coelho Juliao <kinncj@gmail.com>
# All rights reserved.
#
# This software is licensed under the terms of the MIT License.
# See the LICENSE file in the project root for license terms.

from typing import List, Tuple, Optional
import cv2
import os
import datetime
import numpy as np
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip

from models.base import ObjectDetectionModel, FrameDetections
from detection.drawer import DetectionDrawer


class FrameProcessor:
    """
    Processes video frames using object detection models.
    
    This class handles video frame extraction, object detection,
    and drawing detection results on frames.
    """
    
    def __init__(self, model: ObjectDetectionModel, drawer: DetectionDrawer):
        """
        Initialize the FrameProcessor.

        Args:
            model (ObjectDetectionModel): The object detection model to use
            drawer (DetectionDrawer): The drawer for visualizing detections
        """
        self.model = model
        self.drawer = drawer

    def extract_video_fragments(
        self, 
        video_path: str, 
        frame_rate: int = 1
    ) -> Tuple[float, List[np.ndarray], AudioSegment]:
        """
        Extract frames from a video at a specified frame rate.

        Args:
            video_path (str): Path to the input video file
            frame_rate (int): Frame extraction rate in milliseconds

        Returns:
            Tuple[float, List[np.ndarray], AudioSegment]: 
                (fps, frames, audio) extracted from video

        Raises:
            IOError: If the video file cannot be opened
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        try:
            audio = AudioSegment.from_file(video_path)
        except Exception as e:
            print(f"Warning: Could not extract audio: {e}")
            audio = None

        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:  # Safety check for invalid fps
            fps = 30.0  # Default to 30 fps
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Extracting frames every {frame_rate} milliseconds")

        for current_frame in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            current_time_ms = current_frame * (1000 / fps)
            if current_time_ms % frame_rate < (1000 / fps):
                frames.append(frame)

        cap.release()
        return fps, frames, audio

    def process_frame(
        self, 
        frame: np.ndarray, 
        frame_idx: int, 
        display_video: bool = False, 
        image_path: Optional[str] = None
    ) -> FrameDetections:
        """
        Process a single frame by performing object detection and drawing results.

        Args:
            frame (np.ndarray): The frame to process
            frame_idx (int): The index of the frame
            display_video (bool): Whether to display the video with detections
            image_path (str, optional): Path to save the processed images

        Returns:
            FrameDetections: The detection results for this frame
        """
        # Perform object detection
        detections = self.model.detect_objects(frame)
        detections.frame_id = frame_idx
        
        # Draw detections on frame
        frame_with_detections = self.drawer.draw_detections(frame, detections)
        
        # Optionally draw frame info
        frame_with_detections = self.drawer.draw_frame_info(
            frame_with_detections, detections, "top-left"
        )
        
        # Save and display frame if requested
        self._save_frame(frame_with_detections, frame_idx, image_path)
        self._display_frame(frame_with_detections, display_video)
        
        return detections

    def _save_frame(self, frame, frame_idx, image_path):
        """
        Saves a processed frame to the specified directory.

        Args:
            frame: The frame to save.
            frame_idx (int): The index of the frame.
            image_path (str): Path to save the images.
        """
        if image_path is not None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(image_path, f"detected_frame_{frame_idx}_{timestamp}.png")
            cv2.imwrite(output_path, frame)

    def _display_frame(self, frame, display_video):
        """
        Displays a frame with detections if specified.

        Args:
            frame: The frame to display.
            display_video (bool): Whether to display the video with detections.
        """
        if display_video:
            cv2.imshow("ANALYZING FRAMES...", frame)
            cv2.waitKey(1)

    """
    Compiles a list of frames into a video and saves it to the specified directory.
    
    Args:
        frames (list): A list of frames to compile into a video.
        store_video_path (str): Path to save the video.
        fps (int): Frames per second for the video.
        audio (AudioSegment): Audio to add to the video.
    """
    def compile_video(self, frames, store_video_path, fps=20, audio=None):
        """
        Compiles a list of frames into a video and adds audio if provided.

        Args:
            frames (list): A list of frames to compile into a video.
            store_video_path (str): Path to save the video.
            fps (int): Frames per second for the video.
            audio (AudioSegment): Audio to add to the video.
        """
        output_video_path = None
        output_video_and_audio_path = None
        if store_video_path is not None and len(frames) > 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_video_path = os.path.join(store_video_path, f"detected_frames_{timestamp}.mp4")

            # Get the dimensions from the first frame
            h, w, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

            # Write each frame into the video file
            for frame in frames:
                video_writer.write(frame)

            video_writer.release()

            # Add audio to the video if provided
            if audio is not None:
                # Save audio to a temporary file
                temp_audio_path = os.path.join(store_video_path, f"temp_audio_{timestamp}.mp3")
                output_video_and_audio_path = os.path.join(store_video_path, f"detected_frames_with_audio{timestamp}.mp4")
                audio.export(temp_audio_path, format="mp3")

                # Merge the video and audio using moviepy
                video_clip = VideoFileClip(output_video_path)
                audio_clip = AudioFileClip(temp_audio_path)
                video_with_audio = video_clip.set_audio(audio_clip)

                # Write the final video with audio
                video_with_audio.write_videofile(output_video_and_audio_path, audio_codec="aac", fps=fps)

                # Clean up temporary files
                os.remove(temp_audio_path)
                print(f"Video with audio saved to: {output_video_and_audio_path}")
            else:
                print("No audio provided. Video saved without audio.")

            cv2.destroyAllWindows()
        return output_video_path, output_video_and_audio_path