#!/usr/bin/env python3
"""
Unit tests for the config module.

Tests configuration settings, device detection, and constants.
"""

import unittest
import torch
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import device, RESTRICTED_CLASSES, RESTRICTED_COLORS


class TestConfig(unittest.TestCase):
    """Test cases for configuration module."""

    def test_device_configuration(self):
        """Test that device is properly configured."""
        self.assertIsInstance(device, torch.device)
        # Device should be either CPU or MPS (or CUDA if available)
        self.assertIn(str(device), ['cpu', 'mps', 'mps:0', 'cuda', 'cuda:0'])

    def test_restricted_classes_structure(self):
        """Test that RESTRICTED_CLASSES has correct structure."""
        self.assertIsInstance(RESTRICTED_CLASSES, dict)
        self.assertGreater(len(RESTRICTED_CLASSES), 0)
        
        # Test that all keys are integers (class IDs)
        for class_id in RESTRICTED_CLASSES.keys():
            self.assertIsInstance(class_id, int)
            self.assertGreaterEqual(class_id, 0)
        
        # Test that all values are strings (class names)
        for class_name in RESTRICTED_CLASSES.values():
            self.assertIsInstance(class_name, str)
            self.assertGreater(len(class_name), 0)

    def test_restricted_colors_structure(self):
        """Test that RESTRICTED_COLORS has correct structure."""
        self.assertIsInstance(RESTRICTED_COLORS, dict)
        self.assertGreater(len(RESTRICTED_COLORS), 0)
        
        # Test that all keys are strings (class names)
        for class_name in RESTRICTED_COLORS.keys():
            self.assertIsInstance(class_name, str)
            self.assertGreater(len(class_name), 0)
        
        # Test that all values are color tuples (B, G, R)
        for color in RESTRICTED_COLORS.values():
            self.assertIsInstance(color, tuple)
            self.assertEqual(len(color), 3)
            for channel in color:
                self.assertIsInstance(channel, int)
                self.assertGreaterEqual(channel, 0)
                self.assertLessEqual(channel, 255)

    def test_class_color_consistency(self):
        """Test that classes and colors are consistent."""
        # Every class in RESTRICTED_CLASSES should have a corresponding color
        for class_name in RESTRICTED_CLASSES.values():
            self.assertIn(class_name, RESTRICTED_COLORS, 
                         f"Class '{class_name}' missing from RESTRICTED_COLORS")

    def test_specific_classes_present(self):
        """Test that expected classes are present."""
        expected_classes = {"person", "cell phone", "laptop", "tv", "keyboard", "mouse", "clock"}
        actual_classes = set(RESTRICTED_CLASSES.values())
        
        for expected_class in expected_classes:
            if expected_class != "something":  # Skip generic placeholder
                self.assertIn(expected_class, actual_classes, 
                             f"Expected class '{expected_class}' not found")

    @patch('torch.backends.mps.is_available')
    def test_device_selection_mps_available(self, mock_mps_available):
        """Test device selection when MPS is available."""
        mock_mps_available.return_value = True
        
        # Re-import to trigger device selection
        import importlib
        import config.config
        importlib.reload(config.config)
        
        # Should select MPS when available
        expected_device = torch.device("mps")
        self.assertEqual(config.config.device.type, expected_device.type)

    @patch('torch.backends.mps.is_available')
    def test_device_selection_mps_unavailable(self, mock_mps_available):
        """Test device selection when MPS is unavailable."""
        mock_mps_available.return_value = False
        
        # Re-import to trigger device selection
        import importlib
        import config.config
        importlib.reload(config.config)
        
        # Should fallback to CPU when MPS unavailable
        expected_device = torch.device("cpu")
        self.assertEqual(config.config.device.type, expected_device.type)

    def test_color_format_bgr(self):
        """Test that colors are in BGR format (OpenCV standard)."""
        # Test a few known colors to ensure BGR format
        test_colors = {
            "person": (0, 0, 255),      # Red in BGR
            "cell phone": (0, 255, 0),  # Green in BGR
            "laptop": (255, 255, 0),    # Yellow in BGR
        }
        
        for class_name, expected_color in test_colors.items():
            if class_name in RESTRICTED_COLORS:
                actual_color = RESTRICTED_COLORS[class_name]
                self.assertEqual(actual_color, expected_color,
                               f"Color for '{class_name}' should be {expected_color} (BGR)")


if __name__ == '__main__':
    unittest.main()
