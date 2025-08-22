#!/usr/bin/env python3
"""
Unit tests for the model factory module.

Tests model creation patterns and factory functionality.
"""

import unittest
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestModelFactory(unittest.TestCase):
    """Test cases for ModelFactory class."""

    def setUp(self):
        """Set up test fixtures."""
        # Import here to ensure patching works correctly
        from models.factory import ModelFactory
        self.factory = ModelFactory

    def test_supported_models_exist(self):
        """Test that factory has supported models defined."""
        self.assertIn('detr', self.factory.SUPPORTED_MODELS)
        self.assertIn('yolo', self.factory.SUPPORTED_MODELS)
        self.assertIn('yolov8', self.factory.SUPPORTED_MODELS)

    def test_create_model_yolo(self):
        """Test creating YOLO model."""
        mock_yolo = MagicMock()
        mock_instance = MagicMock()
        mock_yolo.return_value = mock_instance
        
        with patch.object(self.factory, 'SUPPORTED_MODELS', {'yolo': mock_yolo}):
            result = self.factory.create_model('yolo')
            
        mock_yolo.assert_called_once()
        self.assertEqual(result, mock_instance)

    def test_create_model_yolov8_alias(self):
        """Test creating YOLO model using yolov8 alias."""
        mock_yolo = MagicMock()
        mock_instance = MagicMock()
        mock_yolo.return_value = mock_instance
        
        with patch.object(self.factory, 'SUPPORTED_MODELS', {'yolov8': mock_yolo}):
            result = self.factory.create_model('yolov8')
            
        mock_yolo.assert_called_once()
        self.assertEqual(result, mock_instance)

    def test_create_model_detr(self):
        """Test creating DETR model."""
        mock_detr = MagicMock()
        mock_instance = MagicMock()
        mock_detr.return_value = mock_instance
        
        with patch.object(self.factory, 'SUPPORTED_MODELS', {'detr': mock_detr}):
            result = self.factory.create_model('detr')
            
        mock_detr.assert_called_once()
        self.assertEqual(result, mock_instance)

    def test_create_default_model(self):
        """Test creating model with default parameters."""
        mock_yolo = MagicMock()
        mock_instance = MagicMock()
        mock_yolo.return_value = mock_instance
        
        with patch.object(self.factory, 'SUPPORTED_MODELS', {'yolo': mock_yolo}):
            result = self.factory.create_model()
            
        mock_yolo.assert_called_once()
        self.assertEqual(result, mock_instance)

    def test_create_model_with_custom_parameters(self):
        """Test creating model with custom parameters."""
        mock_yolo = MagicMock()
        mock_instance = MagicMock()
        mock_yolo.return_value = mock_instance
        
        with patch.object(self.factory, 'SUPPORTED_MODELS', {'yolo': mock_yolo}):
            self.factory.create_model(
                'yolo',
                model_size='s',
                confidence_threshold=0.7
            )
            
        mock_yolo.assert_called_once_with(
            model_size='s',
            confidence_threshold=0.7
        )

    def test_detr_with_confidence_threshold(self):
        """Test creating DETR model with confidence threshold."""
        mock_detr = MagicMock()
        mock_instance = MagicMock()
        mock_detr.return_value = mock_instance
        
        with patch.object(self.factory, 'SUPPORTED_MODELS', {'detr': mock_detr}):
            self.factory.create_model('detr', confidence_threshold=0.8)
            
        mock_detr.assert_called_once_with(confidence_threshold=0.8)

    def test_create_model_invalid_type(self):
        """Test creating model with invalid type raises ValueError."""
        with self.assertRaises(ValueError):
            self.factory.create_model('invalid_model')

    def test_factory_creates_different_instances(self):
        """Test that factory creates different instances for different calls."""
        mock_yolo = MagicMock()
        mock_detr = MagicMock()
        mock_yolo_instance = MagicMock()
        mock_detr_instance = MagicMock()
        mock_yolo.return_value = mock_yolo_instance
        mock_detr.return_value = mock_detr_instance
        
        mock_models = {'yolo': mock_yolo, 'detr': mock_detr}
        
        with patch.object(self.factory, 'SUPPORTED_MODELS', mock_models):
            yolo_model = self.factory.create_model('yolo')
            detr_model = self.factory.create_model('detr')
            
        self.assertNotEqual(yolo_model, detr_model)
        mock_yolo.assert_called_once()
        mock_detr.assert_called_once()

    def test_model_info_consistency(self):
        """Test that model info is consistent with factory registration."""
        mock_detr = MagicMock()
        mock_instance = MagicMock()
        mock_detr.return_value = mock_instance
        
        with patch.object(self.factory, 'SUPPORTED_MODELS', {'detr': mock_detr}):
            model = self.factory.create_model('detr')
            
        # Test that the factory creates the expected type
        self.assertEqual(model, mock_instance)
        mock_detr.assert_called_once()


if __name__ == '__main__':
    unittest.main()
