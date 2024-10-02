# Copyright (c) 2024 Kinn Coelho Juliao <kinncj@gmail.com>
# All rights reserved.
#
# This software is licensed under the terms of the MIT License.
# See the LICENSE file in the project root for license terms.
"""
FrameProcessor class for processing video frames with object detection and drawing detections.

Attributes:
    model: The object detection model to use.
    drawer: The drawer to use for drawing detections on frames.
"""

import cv2
import os
import datetime
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip

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

    def extract_video_fragments(self, video_path, frame_rate=1):
        """
        Extracts frames from a video at a specified frame rate.

        Args:
            video_path (str): Path to the input video file.
            frame_rate (int): Frame extraction rate in milliseconds.

        Returns:
            list: A list of extracted frames.

        Raises:
            IOError: If the video file cannot be opened.
        """
        cap = cv2.VideoCapture(video_path)
        audio = AudioSegment.from_file(video_path)
        if not cap.isOpened():
            raise IOError("Could not open video")

        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Extracting frames every {frame_rate} milliseconds")

        for frame_count in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            current_time_ms = frame_count * (1000 / fps)
            if current_time_ms % frame_rate < (1000 / fps):
                frames.append(frame)

        cap.release()
        return fps, frames, audio

    def process_frame(self, frame, frame_idx, display_video=False, image_path=None):
        """
        Processes a single frame by performing object detection and drawing the detections.

        Args:
            frame: The frame to process.
            frame_idx (int): The index of the frame.
            display_video (bool): Whether to display the video with detections.
            image_path (str): Path to save the images.
        """
        labels, boxes = self.model.analyze_frame(frame)
        frame_with_detections = self.drawer.draw_detections(frame, labels, boxes, self.model.id2label)
        self._save_frame(frame_with_detections, frame_idx, image_path)
        self._display_frame(frame_with_detections, display_video)

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