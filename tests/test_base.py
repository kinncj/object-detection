#!/usr/bin/env python3
"""
Unit tests for the models.base module.

Tests data structures, DTOs, and base classes for object detection.
"""

import unittest
import numpy as np
from typing import List

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.base import BoundingBox, DetectionResult, FrameDetections, ObjectDetectionModel


class TestBoundingBox(unittest.TestCase):
    """Test cases for BoundingBox data class."""

    def test_bounding_box_creation(self):
        """Test BoundingBox creation with valid coordinates."""
        bbox = BoundingBox(x=0.5, y=0.5, width=0.2, height=0.3, confidence=0.9)
        
        self.assertEqual(bbox.x, 0.5)
        self.assertEqual(bbox.y, 0.5)
        self.assertEqual(bbox.width, 0.2)
        self.assertEqual(bbox.height, 0.3)
        self.assertEqual(bbox.confidence, 0.9)

    def test_bounding_box_to_xyxy(self):
        """Test conversion to absolute coordinates."""
        bbox = BoundingBox(x=0.5, y=0.5, width=0.2, height=0.3)
        x1, y1, x2, y2 = bbox.to_xyxy(image_width=100, image_height=200)
        
        # Center at (50, 100), width=20, height=60
        # So x1=40, y1=70, x2=60, y2=130
        self.assertEqual(x1, 40)
        self.assertEqual(y1, 70)
        self.assertEqual(x2, 60)
        self.assertEqual(y2, 130)

    def test_bounding_box_from_xyxy(self):
        """Test creation from absolute coordinates."""
        bbox = BoundingBox.from_xyxy(
            x1=40, y1=70, x2=60, y2=130,
            image_width=100, image_height=200,
            confidence=0.8
        )
        
        self.assertAlmostEqual(bbox.x, 0.5, places=2)
        self.assertAlmostEqual(bbox.y, 0.5, places=2)
        self.assertAlmostEqual(bbox.width, 0.2, places=2)
        self.assertAlmostEqual(bbox.height, 0.3, places=2)
        self.assertEqual(bbox.confidence, 0.8)

    def test_bounding_box_edge_cases(self):
        """Test BoundingBox with edge cases."""
        # Zero area box
        bbox_zero = BoundingBox(x=0.5, y=0.5, width=0.0, height=0.0)
        self.assertEqual(bbox_zero.width, 0.0)
        self.assertEqual(bbox_zero.height, 0.0)
        
        # Full image box
        bbox_full = BoundingBox(x=0.5, y=0.5, width=1.0, height=1.0)
        self.assertEqual(bbox_full.width, 1.0)
        self.assertEqual(bbox_full.height, 1.0)

    def test_bounding_box_string_representation(self):
        """Test BoundingBox string representation."""
        bbox = BoundingBox(x=0.5, y=0.5, width=0.2, height=0.3, confidence=0.9)
        str_repr = str(bbox)
        
        self.assertIn("0.5", str_repr)
        self.assertIn("0.2", str_repr)
        self.assertIn("0.3", str_repr)
        self.assertIn("0.9", str_repr)

    def test_bounding_box_equality(self):
        """Test BoundingBox equality comparison."""
        bbox1 = BoundingBox(x=0.5, y=0.5, width=0.2, height=0.3, confidence=0.9)
        bbox2 = BoundingBox(x=0.5, y=0.5, width=0.2, height=0.3, confidence=0.9)
        bbox3 = BoundingBox(x=0.6, y=0.5, width=0.2, height=0.3, confidence=0.9)
        
        self.assertEqual(bbox1, bbox2)
        self.assertNotEqual(bbox1, bbox3)


class TestDetectionResult(unittest.TestCase):
    """Test cases for DetectionResult data class."""

    def setUp(self):
        """Set up test fixtures."""
        self.bbox = BoundingBox(x=0.5, y=0.5, width=0.2, height=0.3, confidence=0.9)

    def test_detection_result_creation(self):
        """Test DetectionResult creation."""
        detection = DetectionResult(
            class_id=1,
            class_name="person",
            bbox=self.bbox,
            confidence=0.95,
            model_type="YOLOv8"
        )
        
        self.assertEqual(detection.class_id, 1)
        self.assertEqual(detection.class_name, "person")
        self.assertEqual(detection.bbox, self.bbox)
        self.assertEqual(detection.confidence, 0.95)
        self.assertEqual(detection.model_type, "YOLOv8")

    def test_detection_result_confidence_validation(self):
        """Test DetectionResult confidence validation."""
        # Valid confidence values
        valid_confidences = [0.0, 0.5, 1.0]
        
        for conf in valid_confidences:
            detection = DetectionResult(
                class_id=1,
                class_name="test",
                bbox=self.bbox,
                confidence=conf,
                model_type="test"
            )
            self.assertEqual(detection.confidence, conf)
        
        # Invalid confidence values
        invalid_confidences = [-0.1, 1.1, 2.0]
        
        for conf in invalid_confidences:
            with self.assertRaises(ValueError):
                DetectionResult(
                    class_id=1,
                    class_name="test",
                    bbox=self.bbox,
                    confidence=conf,
                    model_type="test"
                )

    def test_detection_result_class_id_validation(self):
        """Test DetectionResult class ID validation."""
        # Valid class IDs
        valid_ids = [0, 1, 10, 100]
        
        for class_id in valid_ids:
            detection = DetectionResult(
                class_id=class_id,
                class_name="test",
                bbox=self.bbox,
                confidence=0.5,
                model_type="test"
            )
            self.assertEqual(detection.class_id, class_id)
        
        # Invalid class IDs
        invalid_ids = [-1, -10]
        
        for class_id in invalid_ids:
            with self.assertRaises(ValueError):
                DetectionResult(
                    class_id=class_id,
                    class_name="test",
                    bbox=self.bbox,
                    confidence=0.5,
                    model_type="test"
                )

    def test_detection_result_string_representation(self):
        """Test DetectionResult string representation."""
        detection = DetectionResult(
            class_id=1,
            class_name="person",
            bbox=self.bbox,
            confidence=0.95,
            model_type="YOLOv8"
        )
        
        str_repr = str(detection)
        self.assertIn("person", str_repr)
        self.assertIn("0.95", str_repr)
        self.assertIn("1", str_repr)

    def test_detection_result_equality(self):
        """Test DetectionResult equality comparison."""
        detection1 = DetectionResult(
            class_id=1,
            class_name="person",
            bbox=self.bbox,
            confidence=0.95,
            model_type="YOLOv8"
        )
        detection2 = DetectionResult(
            class_id=1,
            class_name="person",
            bbox=self.bbox,
            confidence=0.95,
            model_type="YOLOv8"
        )
        detection3 = DetectionResult(
            class_id=73,
            class_name="laptop",
            bbox=self.bbox,
            confidence=0.95,
            model_type="YOLOv8"
        )
        
        self.assertEqual(detection1, detection2)
        self.assertNotEqual(detection1, detection3)


class TestFrameDetections(unittest.TestCase):
    """Test cases for FrameDetections data class."""

    def setUp(self):
        """Set up test fixtures."""
        self.bbox1 = BoundingBox(x=0.3, y=0.3, width=0.2, height=0.3, confidence=0.9)
        self.bbox2 = BoundingBox(x=0.7, y=0.7, width=0.2, height=0.3, confidence=0.8)
        
        self.detection1 = DetectionResult(
            class_id=1,
            class_name="person",
            bbox=self.bbox1,
            confidence=0.95,
            model_type="YOLOv8"
        )
        self.detection2 = DetectionResult(
            class_id=73,
            class_name="laptop",
            bbox=self.bbox2,
            confidence=0.87,
            model_type="YOLOv8"
        )

    def test_frame_detections_creation_empty(self):
        """Test FrameDetections creation with empty list."""
        frame_detections = FrameDetections(frame_id=0, detections=[])
        
        self.assertEqual(frame_detections.frame_id, 0)
        self.assertEqual(len(frame_detections.detections), 0)
        self.assertEqual(frame_detections.detection_count, 0)
        self.assertIsInstance(frame_detections.detections, list)

    def test_frame_detections_creation_with_detections(self):
        """Test FrameDetections creation with detection list."""
        frame_detections = FrameDetections(
            frame_id=5,
            detections=[self.detection1, self.detection2],
            processing_time=0.05
        )
        
        self.assertEqual(frame_detections.frame_id, 5)
        self.assertEqual(len(frame_detections.detections), 2)
        self.assertEqual(frame_detections.detection_count, 2)
        self.assertEqual(frame_detections.processing_time, 0.05)
        self.assertEqual(frame_detections.detections[0], self.detection1)
        self.assertEqual(frame_detections.detections[1], self.detection2)

    def test_frame_detections_get_detections_by_class(self):
        """Test filtering detections by class."""
        frame_detections = FrameDetections(
            frame_id=0,
            detections=[self.detection1, self.detection2]
        )
        
        # Filter for person detections
        person_detections = frame_detections.get_detections_by_class("person")
        self.assertEqual(len(person_detections), 1)
        self.assertEqual(person_detections[0].class_name, "person")
        
        # Filter for laptop detections
        laptop_detections = frame_detections.get_detections_by_class("laptop")
        self.assertEqual(len(laptop_detections), 1)
        self.assertEqual(laptop_detections[0].class_name, "laptop")
        
        # Filter for non-existent class
        car_detections = frame_detections.get_detections_by_class("car")
        self.assertEqual(len(car_detections), 0)

    def test_frame_detections_model_info(self):
        """Test FrameDetections with model info."""
        model_info = {"model_name": "YOLOv8n", "inference_time": 0.05}
        frame_detections = FrameDetections(
            frame_id=0,
            detections=[self.detection1],
            processing_time=0.05,
            model_info=model_info
        )
        
        self.assertEqual(frame_detections.model_info, model_info)
        self.assertEqual(frame_detections.model_info["model_name"], "YOLOv8n")

    def test_frame_detections_string_representation(self):
        """Test FrameDetections string representation."""
        frame_detections = FrameDetections(
            frame_id=5,
            detections=[self.detection1, self.detection2]
        )
        
        str_repr = str(frame_detections)
        self.assertIn("5", str_repr)  # Should mention frame ID
        self.assertIn("2", str_repr)  # Should mention count


class TestObjectDetectionModelBase(unittest.TestCase):
    """Test cases for ObjectDetectionModel abstract base class."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that ObjectDetectionModel cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            ObjectDetectionModel()

    def test_concrete_implementation_requires_all_methods(self):
        """Test that concrete implementations must implement all abstract methods."""
        class IncompleteModel(ObjectDetectionModel):
            pass
        
        with self.assertRaises(TypeError):
            IncompleteModel()

    def test_concrete_implementation_with_all_methods(self):
        """Test that concrete implementations work with all methods implemented."""
        class CompleteModel(ObjectDetectionModel):
            def detect_objects(self, frame):
                return FrameDetections(frame_id=0, detections=[])
            
            def get_model_info(self):
                return {"model_name": "TestModel", "version": "1.0"}
            
            @property
            def model_name(self):
                return "TestModel"
            
            @property 
            def supported_classes(self):
                return {1: "person", 73: "laptop"}
        
        # Should not raise an error
        model = CompleteModel()
        self.assertIsInstance(model, ObjectDetectionModel)
        
        # Test methods
        test_frame = np.zeros((100, 100, 3))
        result = model.detect_objects(test_frame)
        self.assertIsInstance(result, FrameDetections)
        
        info = model.get_model_info()
        self.assertIsInstance(info, dict)
        
        self.assertEqual(model.model_name, "TestModel")
        self.assertIsInstance(model.supported_classes, dict)
        
        # Test is_class_supported method
        self.assertTrue(model.is_class_supported(1))
        self.assertTrue(model.is_class_supported(73))
        self.assertFalse(model.is_class_supported(999))


if __name__ == '__main__':
    unittest.main()
