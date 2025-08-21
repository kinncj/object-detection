#!/usr/bin/env python3
"""
Test script for the improved object detection architecture.

This script tests the new modular architecture with proper separation of concerns.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_model_factory():
    """Test the ModelFactory functionality."""
    print("🧪 Testing ModelFactory...")
    
    try:
        from models.factory import ModelFactory
        
        # Test getting supported models
        supported = ModelFactory.get_supported_models()
        print(f"   ✅ Supported models: {list(supported.keys())}")
        
        # Test creating default model
        model = ModelFactory.create_default_model()
        print(f"   ✅ Default model created: {model.model_name}")
        
        # Test model info
        info = model.get_model_info()
        print(f"   ✅ Model info: {info['model_type']} with {info['restricted_classes']} classes")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ModelFactory test failed: {e}")
        return False

def test_detection_data_structures():
    """Test the detection data structures."""
    print("🧪 Testing detection data structures...")
    
    try:
        from models.base import BoundingBox, DetectionResult, FrameDetections
        
        # Test BoundingBox
        bbox = BoundingBox(0.5, 0.5, 0.2, 0.3, 0.9)
        x1, y1, x2, y2 = bbox.to_xyxy(1920, 1080)
        print(f"   ✅ BoundingBox conversion: ({x1}, {y1}, {x2}, {y2})")
        
        # Test DetectionResult
        detection = DetectionResult(
            class_id=1,
            class_name="person",
            bbox=bbox,
            confidence=0.9,
            model_type="YOLOv8"
        )
        print(f"   ✅ DetectionResult: {detection.class_name} ({detection.confidence:.2f})")
        
        # Test FrameDetections
        frame_detections = FrameDetections(
            frame_id=0,
            detections=[detection],
            processing_time=0.05
        )
        print(f"   ✅ FrameDetections: {frame_detections.detection_count} detections")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data structures test failed: {e}")
        return False

def test_synthetic_detection():
    """Test detection on a synthetic image."""
    print("🧪 Testing synthetic detection...")
    
    try:
        from models.factory import ModelFactory
        
        # Create a synthetic image (blue background)
        synthetic_image = np.full((480, 640, 3), [255, 0, 0], dtype=np.uint8)  # Blue background
        
        # Create model
        model = ModelFactory.create_model('yolo', 'n')
        
        # Run detection
        start_time = time.time()
        detections = model.detect_objects(synthetic_image)
        processing_time = time.time() - start_time
        
        print(f"   ✅ Detection completed in {processing_time:.3f}s")
        print(f"   ✅ Found {detections.detection_count} detections")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Synthetic detection test failed: {e}")
        return False

def test_drawer():
    """Test the detection drawer."""
    print("🧪 Testing DetectionDrawer...")
    
    try:
        from detection.drawer import DetectionDrawer
        from models.base import BoundingBox, DetectionResult, FrameDetections
        
        # Create synthetic image and detection
        image = np.full((480, 640, 3), [128, 128, 128], dtype=np.uint8)  # Gray background
        
        bbox = BoundingBox(0.5, 0.5, 0.3, 0.4, 0.8)
        detection = DetectionResult(
            class_id=1,
            class_name="person",
            bbox=bbox,
            confidence=0.8,
            model_type="YOLOv8"
        )
        
        frame_detections = FrameDetections(
            frame_id=1,
            detections=[detection],
            processing_time=0.05
        )
        
        # Test drawing
        drawer = DetectionDrawer()
        result_image = drawer.draw_detections(image, frame_detections)
        
        print(f"   ✅ Detection drawing completed")
        print(f"   ✅ Result image shape: {result_image.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ DetectionDrawer test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting improved architecture tests...\n")
    
    tests = [
        test_model_factory,
        test_detection_data_structures,
        test_synthetic_detection,
        test_drawer,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Empty line between tests
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The improved architecture is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
