#!/usr/bin/env python3

# Simple test without moviepy dependency

import cv2
from detection.model import create_model

def simple_video_test(video_path, model_type='yolo', model_size='n'):
    """Simple video test using only OpenCV"""
    print(f"Testing {model_type} model on video: {video_path}")
    
    # Create model
    model = create_model(model_type, model_size)
    print(f"Model created: {model.__class__.__name__}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    print(f"Video info: {frame_count} frames, {fps:.2f} FPS, {duration:.2f}s duration")
    
    # Process every 30th frame (roughly 1 frame per second for 30fps video)
    frame_skip = max(1, int(fps))
    processed_frames = 0
    total_detections = 0
    
    for i in range(0, frame_count, frame_skip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Analyze frame
        labels, boxes = model.analyze_frame(frame)
        detections = len(labels)
        total_detections += detections
        processed_frames += 1
        
        timestamp = i / fps
        print(f"Frame {i:4d} ({timestamp:6.2f}s): {detections} detections")
        
        if detections > 0:
            for j, (label, box) in enumerate(zip(labels, boxes)):
                class_name = model.id2label.get(label, f"Class {label}")
                print(f"  - {class_name}: {box}")
    
    cap.release()
    
    print(f"\nSummary:")
    print(f"Processed {processed_frames} frames")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per frame: {total_detections/max(1,processed_frames):.2f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python test_simple.py <video_path> [model_type] [model_size]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else 'yolo'
    model_size = sys.argv[3] if len(sys.argv) > 3 else 'n'
    
    simple_video_test(video_path, model_type, model_size)
