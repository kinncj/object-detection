#!/usr/bin/env python3
"""
Simple test script to verify YOLOv8 and DETR models work correctly.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.factory import ModelFactory

def test_model_creation():
    """Test that models can be created successfully."""
    print("🧪 Testing model creation...")
    
    try:
        # Test DETR model
        print("  Creating DETR model...")
        detr_model = ModelFactory.create_model("detr")
        print("  ✅ DETR model created successfully")
        
        if hasattr(detr_model, 'get_model_info'):
            info = detr_model.get_model_info()
            print(f"  📋 DETR Model Info: {info}")
        
        # Test YOLOv8 models
        for size in ['n', 's']:  # Test nano and small only to save time
            print(f"  Creating YOLOv8{size.upper()} model...")
            yolo_model = ModelFactory.create_model("yolo", model_size=size)
            print(f"  ✅ YOLOv8{size.upper()} model created successfully")
            
            if hasattr(yolo_model, 'get_model_info'):
                info = yolo_model.get_model_info()
                print(f"  📋 YOLOv8{size.upper()} Model Info: {info}")
        
        print("🎉 All models created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating models: {e}")
        return False

def test_dummy_inference():
    """Test inference with a dummy frame."""
    print("\n🧪 Testing inference with dummy data...")
    
    try:
        # Create a dummy frame (640x480 RGB)
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test DETR
        print("  Testing DETR inference...")
        detr_model = ModelFactory.create_model("detr")
        detections = detr_model.detect_objects(dummy_frame)
        print(f"  ✅ DETR inference completed. Found {len(detections.detections)} detections")
        
        # Test YOLOv8 nano (fastest for testing)
        print("  Testing YOLOv8n inference...")
        yolo_model = ModelFactory.create_model("yolo", model_size="n")
        detections = yolo_model.detect_objects(dummy_frame)
        print(f"  ✅ YOLOv8n inference completed. Found {len(detections.detections)} detections")
        
        print("🎉 All inference tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting YOLOv8 Integration Tests")
    print("=" * 50)
    
    success = True
    
    # Test model creation
    success &= test_model_creation()
    
    # Test inference
    success &= test_dummy_inference()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! YOLOv8 integration is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)
