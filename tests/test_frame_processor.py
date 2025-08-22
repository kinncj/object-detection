#!/usr/bin/env python3
"""
Unit tests for the frame processor module.

Tests video processing, frame extraction, and detection integration.
"""

import unittest
import numpy as np
import os
from unittest.mock import MagicMock, patch, mock_open
from typing import List

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from processor.frame_processor import FrameProcessor
from models.base import ObjectDetectionModel, DetectionResult, BoundingBox, FrameDetections
from detection.drawer import DetectionDrawer


class TestFrameProcessor(unittest.TestCase):
    """Test cases for FrameProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = MagicMock(spec=ObjectDetectionModel)
        self.mock_model.detect_objects = MagicMock()  # Correct method name
        self.mock_drawer = MagicMock(spec=DetectionDrawer)
        self.processor = FrameProcessor(self.mock_model, self.mock_drawer)
        self.test_video_path = "test_video.mp4"

    def test_processor_initialization(self):
        """Test FrameProcessor initialization."""
        self.assertEqual(self.processor.model, self.mock_model)
        self.assertEqual(self.processor.drawer, self.mock_drawer)

    @patch('cv2.VideoCapture')
    @patch('pydub.AudioSegment.from_file')
    def test_extract_video_fragments_success(self, mock_audio, mock_video_capture):
        """Test successful video frame extraction."""
        # Mock video capture
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 30.0,  # cv2.CAP_PROP_FPS
            7: 100    # cv2.CAP_PROP_FRAME_COUNT
        }.get(prop, 0)
        
        # Mock frame reading
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, test_frame)] * 100 + [(False, None)]
        
        # Mock audio
        mock_audio_segment = MagicMock()
        mock_audio.return_value = mock_audio_segment
        
        # Test extraction
        fps, frames, audio = self.processor.extract_video_fragments(
            self.test_video_path, frame_rate=1000
        )
        
        self.assertEqual(fps, 30.0)
        self.assertIsInstance(frames, list)
        self.assertGreater(len(frames), 0)
        self.assertEqual(audio, mock_audio_segment)
        mock_cap.release.assert_called_once()

    @patch('cv2.VideoCapture')
    def test_extract_video_fragments_file_not_found(self, mock_video_capture):
        """Test video extraction with file that cannot be opened."""
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = False
        
        with self.assertRaises(IOError) as context:
            self.processor.extract_video_fragments(self.test_video_path)
        
        self.assertIn("Could not open video", str(context.exception))

    @patch('cv2.VideoCapture')
    @patch('pydub.AudioSegment.from_file')
    def test_extract_video_fragments_audio_failure(self, mock_audio, mock_video_capture):
        """Test video extraction when audio extraction fails."""
        # Mock successful video capture
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 30.0,  # cv2.CAP_PROP_FPS
            7: 10     # cv2.CAP_PROP_FRAME_COUNT
        }.get(prop, 0)
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, test_frame)] * 10 + [(False, None)]
        
        # Mock audio failure
        mock_audio.side_effect = Exception("Audio extraction failed")
        
        # Should handle audio failure gracefully
        fps, frames, audio = self.processor.extract_video_fragments(self.test_video_path)
        
        self.assertEqual(fps, 30.0)
        self.assertIsInstance(frames, list)
        self.assertIsNone(audio)

    def test_process_frame_basic(self):
        """Test basic frame processing."""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_idx = 0
        
        # Mock model analysis - use center coordinates and dimensions
        bbox = BoundingBox(x=0.25, y=0.25, width=0.15625, height=0.208333, confidence=0.95)  # roughly 100,100,200,200 in 640x480
        detection = DetectionResult(
            class_id=1,
            class_name="person",
            bbox=bbox,
            confidence=0.95,
            model_type="test"
        )
        mock_detections = FrameDetections(frame_id=frame_idx, detections=[detection])
        self.mock_model.detect_objects.return_value = mock_detections
        
        # Mock drawer
        drawn_frame = test_frame.copy()
        self.mock_drawer.draw_detections.return_value = drawn_frame
        
        # Process frame
        result = self.processor.process_frame(test_frame, frame_idx)
        
        # Verify calls and results
        self.mock_model.detect_objects.assert_called_once_with(test_frame)
        self.mock_drawer.draw_detections.assert_called_once_with(test_frame, mock_detections)
        self.assertEqual(result, mock_detections)

    @patch('cv2.imwrite')
    def test_process_frame_with_image_save(self, mock_imwrite):
        """Test frame processing with image saving."""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_idx = 5
        image_path = "/test/output/"
        
        # Mock model analysis - include required frame_id
        mock_detections = FrameDetections(frame_id=frame_idx, detections=[])
        self.mock_model.detect_objects.return_value = mock_detections
        
        # Mock drawer
        drawn_frame = test_frame.copy()
        self.mock_drawer.draw_detections.return_value = drawn_frame
        
        # Process frame with image saving
        result = self.processor.process_frame(
            test_frame, frame_idx, image_path=image_path
        )
        
        # Verify image was saved
        mock_imwrite.assert_called_once()
        call_args = mock_imwrite.call_args[0]
        self.assertIn("frame_5", call_args[0])  # Check filename contains frame index
        
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    def test_process_frame_with_display(self, mock_destroy, mock_waitkey, mock_imshow):
        """Test frame processing with video display."""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_idx = 0
        
        # Mock model analysis
        mock_detections = FrameDetections(frame_id=0, detections=[])
        self.mock_model.detect_objects.return_value = mock_detections
        
        # Mock drawer
        drawn_frame = test_frame.copy()
        self.mock_drawer.draw_detections.return_value = drawn_frame
        
        # Mock OpenCV display functions
        mock_waitkey.return_value = ord('q')  # Simulate 'q' key press
        
        # Process frame with display
        result = self.processor.process_frame(
            test_frame, frame_idx, display_video=True
        )
        
        # Verify display was called
        mock_imshow.assert_called_once()
        mock_waitkey.assert_called_once()

    def test_save_frame_path_creation(self):
        """Test frame saving with path creation."""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_idx = 10
        image_path = "/test/output/"
        
        with patch('cv2.imwrite') as mock_imwrite, \
             patch('os.makedirs') as mock_makedirs:
            
            self.processor._save_frame(test_frame, frame_idx, image_path)
            
            # Verify directory creation and file saving
            mock_makedirs.assert_called_once_with(image_path, exist_ok=True)
            mock_imwrite.assert_called_once()

    def test_display_frame_quit_on_q(self):
        """Test display frame quits on 'q' key."""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with patch('cv2.imshow') as mock_imshow, \
             patch('cv2.waitKey') as mock_waitkey, \
             patch('cv2.destroyAllWindows') as mock_destroy:
            
            mock_waitkey.return_value = ord('q')
            
            self.processor._display_frame(test_frame, display_video=True)
            
            mock_imshow.assert_called_once()
            mock_waitkey.assert_called_once()
            mock_destroy.assert_called_once()

    def test_display_frame_continue_on_other_key(self):
        """Test display frame continues on non-'q' key."""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with patch('cv2.imshow') as mock_imshow, \
             patch('cv2.waitKey') as mock_waitkey, \
             patch('cv2.destroyAllWindows') as mock_destroy:
            
            mock_waitkey.return_value = ord('c')  # Any key except 'q'
            
            self.processor._display_frame(test_frame, display_video=True)
            
            mock_imshow.assert_called_once()
            mock_waitkey.assert_called_once()
            mock_destroy.assert_not_called()  # Should not destroy on non-'q'

    def test_frame_rate_calculation(self):
        """Test frame rate calculation in extraction."""
        with patch('cv2.VideoCapture') as mock_video_capture, \
             patch('pydub.AudioSegment.from_file'):
            
            mock_cap = MagicMock()
            mock_video_capture.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            
            # Test different FPS values
            fps_values = [24.0, 30.0, 60.0]
            
            for fps in fps_values:
                mock_cap.get.side_effect = lambda prop: {
                    0: fps,  # cv2.CAP_PROP_FPS
                    7: 100   # cv2.CAP_PROP_FRAME_COUNT
                }.get(prop, 0)
                
                test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                mock_cap.read.side_effect = [(True, test_frame)] * 5 + [(False, None)]
                
                extracted_fps, frames, audio = self.processor.extract_video_fragments(
                    self.test_video_path, frame_rate=1000
                )
                
                self.assertEqual(extracted_fps, fps)

    def test_integration_with_real_detection_objects(self):
        """Test integration with actual detection objects."""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create real detection objects using correct constructor
        bbox = BoundingBox.from_xyxy(50, 50, 150, 150, 640, 480, 0.95)
        detection1 = DetectionResult(
            class_id=1,
            class_name="person",
            bbox=bbox,
            confidence=0.95,
            model_type="test"
        )
        
        bbox2 = BoundingBox.from_xyxy(200, 200, 300, 300, 640, 480, 0.87)
        detection2 = DetectionResult(
            class_id=73,
            class_name="laptop",
            bbox=bbox2,
            confidence=0.87,
            model_type="test"
        )
        
        frame_detections = FrameDetections(frame_id=0, detections=[detection1, detection2])
        
        # Mock model to return real detections
        self.mock_model.detect_objects.return_value = frame_detections
        self.mock_drawer.draw_detections.return_value = test_frame
        
        # Process frame
        result = self.processor.process_frame(test_frame, 0)
        
        # Verify the result contains the expected detections
        self.assertIsInstance(result, FrameDetections)
        self.assertEqual(len(result.detections), 2)
        self.assertEqual(result.detections[0].class_name, "person")
        self.assertEqual(result.detections[1].class_name, "laptop")


if __name__ == '__main__':
    unittest.main()
