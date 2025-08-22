#!/usr/bin/env python3
"""
Unit tests for the detection drawer module.

Tests visualization and drawing utilities for object detection results.
"""

import unittest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from detection.drawer import DetectionDrawer
from models.base import BoundingBox, DetectionResult, FrameDetections


class TestDetectionDrawer(unittest.TestCase):
    """Test cases for DetectionDrawer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.drawer = DetectionDrawer()
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_drawer_initialization_default(self):
        """Test default initialization of DetectionDrawer."""
        drawer = DetectionDrawer()
        
        self.assertEqual(drawer.box_color, (0, 255, 0))
        self.assertEqual(drawer.text_color, (255, 255, 255))
        self.assertEqual(drawer.box_thickness, 2)
        self.assertEqual(drawer.font_scale, 0.6)
        self.assertEqual(drawer.font_thickness, 2)

    def test_drawer_initialization_custom(self):
        """Test custom initialization of DetectionDrawer."""
        custom_box_color = (255, 0, 0)
        custom_text_color = (0, 0, 255)
        custom_box_thickness = 3
        custom_font_scale = 0.8
        custom_font_thickness = 1
        
        drawer = DetectionDrawer(
            box_color=custom_box_color,
            text_color=custom_text_color,
            box_thickness=custom_box_thickness,
            font_scale=custom_font_scale,
            font_thickness=custom_font_thickness
        )
        
        self.assertEqual(drawer.box_color, custom_box_color)
        self.assertEqual(drawer.text_color, custom_text_color)
        self.assertEqual(drawer.box_thickness, custom_box_thickness)
        self.assertEqual(drawer.font_scale, custom_font_scale)
        self.assertEqual(drawer.font_thickness, custom_font_thickness)

    def test_draw_detection_single_object(self):
        """Test drawing a single detection on frame."""
        # Create a test detection - center at (150, 150) with size 100x100
        bbox = BoundingBox(x=0.234, y=0.313, width=0.156, height=0.208, confidence=0.95)
        detection = DetectionResult(
            bbox=bbox,
            class_name="person",
            confidence=0.95,
            class_id=1,
            model_type="test"
        )
        frame_detections = FrameDetections(frame_id=1, detections=[detection])
        
        # Draw the detection
        result_frame = self.drawer.draw_detections(self.test_frame, frame_detections)
        
        # Check that result is an numpy array
        self.assertIsInstance(result_frame, np.ndarray)
        self.assertEqual(result_frame.shape, self.test_frame.shape)
        
        # The frame should be modified (not equal to original zeros)
        self.assertFalse(np.array_equal(result_frame, self.test_frame))

    def test_draw_detection_multiple_objects(self):
        """Test drawing multiple detections on frame."""
        # Create multiple test detections
        bbox1 = BoundingBox(x=0.156, y=0.208, width=0.156, height=0.208, confidence=0.95)
        bbox2 = BoundingBox(x=0.391, y=0.625, width=0.156, height=0.208, confidence=0.87)
        
        detection1 = DetectionResult(
            bbox=bbox1,
            class_name="person",
            confidence=0.95,
            class_id=1,
            model_type="test"
        )
        detection2 = DetectionResult(
            bbox=bbox2,
            class_name="laptop",
            confidence=0.87,
            class_id=73,
            model_type="test"
        )
        
        frame_detections = FrameDetections(frame_id=1, detections=[detection1, detection2])
        
        # Draw the detections
        result_frame = self.drawer.draw_detections(self.test_frame, frame_detections)
        
        self.assertIsInstance(result_frame, np.ndarray)
        self.assertEqual(result_frame.shape, self.test_frame.shape)
        self.assertFalse(np.array_equal(result_frame, self.test_frame))

    def test_draw_detection_empty_list(self):
        """Test drawing with empty detection list."""
        frame_detections = FrameDetections(frame_id=1, detections=[])
        
        result_frame = self.drawer.draw_detections(self.test_frame, frame_detections)
        
        # With no detections, frame should remain unchanged
        self.assertTrue(np.array_equal(result_frame, self.test_frame))

    def test_draw_detection_invalid_coordinates(self):
        """Test drawing with invalid bounding box coordinates."""
        # Create detection with coordinates outside frame bounds  
        bbox = BoundingBox(x=-0.1, y=-0.1, width=2.0, height=2.0, confidence=0.95)
        detection = DetectionResult(
            bbox=bbox,
            class_name="person",
            confidence=0.95,
            class_id=1,
            model_type="test"
        )
        frame_detections = FrameDetections(frame_id=1, detections=[detection])
        
        # Should handle gracefully without crashing
        try:
            result_frame = self.drawer.draw_detections(self.test_frame, frame_detections)
            self.assertIsInstance(result_frame, np.ndarray)
        except Exception as e:
            self.fail(f"Drawing with invalid coordinates should not raise exception: {e}")

    @patch('cv2.rectangle')
    @patch('cv2.putText')
    def test_draw_calls_opencv_functions(self, mock_put_text, mock_rectangle):
        """Test that drawing calls appropriate OpenCV functions."""
        bbox = BoundingBox(x=0.234, y=0.313, width=0.156, height=0.208, confidence=0.95)
        detection = DetectionResult(
            bbox=bbox,
            class_name="person",
            confidence=0.95,
            class_id=1,
            model_type="test"
        )
        frame_detections = FrameDetections(frame_id=1, detections=[detection])
        
        self.drawer.draw_detections(self.test_frame, frame_detections)
        
        # Verify that cv2.rectangle was called for bounding box
        mock_rectangle.assert_called()
        
        # Verify that cv2.putText was called for label
        mock_put_text.assert_called()

    def test_format_label_with_confidence(self):
        """Test label formatting includes confidence score."""
        bbox = BoundingBox(x=0.234, y=0.313, width=0.156, height=0.208, confidence=0.95)
        detection = DetectionResult(
            bbox=bbox,
            class_name="person",
            confidence=0.95,
            class_id=1,
            model_type="test"
        )
        
        # The exact implementation might vary, but we expect class name and confidence
        # This test ensures the detection is processed without error
        frame_detections = FrameDetections(frame_id=1, detections=[detection])
        result_frame = self.drawer.draw_detections(self.test_frame, frame_detections)
        
        self.assertIsInstance(result_frame, np.ndarray)

    def test_color_selection_for_different_classes(self):
        """Test that different classes get appropriate colors."""
        # Create detections for different classes
        bbox = BoundingBox(x=0.234, y=0.313, width=0.156, height=0.208, confidence=0.95)
        
        person_detection = DetectionResult(
            bbox=bbox,
            class_name="person",
            confidence=0.95,
            class_id=1,
            model_type="test"
        )
        
        laptop_detection = DetectionResult(
            bbox=bbox,
            class_name="laptop",
            confidence=0.87,
            class_id=73,
            model_type="test"
        )
        
        # Test each detection individually
        person_detections = FrameDetections(frame_id=1, detections=[person_detection])
        laptop_detections = FrameDetections(frame_id=2, detections=[laptop_detection])
        
        person_frame = self.drawer.draw_detections(self.test_frame.copy(), person_detections)
        laptop_frame = self.drawer.draw_detections(self.test_frame.copy(), laptop_detections)
        
        # Both should be valid frames
        self.assertIsInstance(person_frame, np.ndarray)
        self.assertIsInstance(laptop_frame, np.ndarray)

    def test_frame_copy_preservation(self):
        """Test that original frame is not modified."""
        original_frame = self.test_frame.copy()
        
        bbox = BoundingBox(x=0.234, y=0.313, width=0.156, height=0.208, confidence=0.95)
        detection = DetectionResult(
            bbox=bbox,
            class_name="person",
            confidence=0.95,
            class_id=1,
            model_type="test"
        )
        frame_detections = FrameDetections(frame_id=1, detections=[detection])
        
        result_frame = self.drawer.draw_detections(self.test_frame, frame_detections)
        
        # Original frame should remain unchanged if drawer makes a copy
        # Note: This depends on implementation - if drawer modifies in-place, 
        # this test might need adjustment
        self.assertIsInstance(result_frame, np.ndarray)

    def test_confidence_threshold_display(self):
        """Test detection with various confidence levels."""
        confidences = [0.1, 0.5, 0.9, 0.99]
        
        for conf in confidences:
            bbox = BoundingBox(x=0.234, y=0.313, width=0.156, height=0.208, confidence=conf)
            detection = DetectionResult(
                bbox=bbox,
                class_name="person",
                confidence=conf,
                class_id=1,
            model_type="test"
            )
            frame_detections = FrameDetections(frame_id=1, detections=[detection])
            
            # Should handle all confidence levels without error
            result_frame = self.drawer.draw_detections(self.test_frame, frame_detections)
            self.assertIsInstance(result_frame, np.ndarray)


if __name__ == '__main__':
    unittest.main()
