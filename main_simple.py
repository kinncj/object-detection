#!/usr/bin/env python3

# Simplified main app that works without moviepy dependencies

import argparse
import cv2
import os
from detection.model import create_model
from detection.drawer import DetectionDrawer

def simple_video_detection(video_path, model_type='yolo', model_size='n', output_dir=None):
    """
    Simple video detection without moviepy dependencies
    """
    print(f"🎬 Processing video: {video_path}")
    print(f"🤖 Using {model_type.upper()} model" + (f" (size: {model_size})" if model_type == 'yolo' else ""))
    
    # Create model
    if model_type.lower() == "yolo":
        model = create_model(model_type, model_size)
    else:
        model = create_model(model_type)
    
    # Print model information
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        print(f"📋 Model info: {info}")
    
    # Create drawer
    drawer = DetectionDrawer()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"❌ Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps
    
    print(f"📺 Video: {frame_count} frames, {fps:.2f} FPS, {width}x{height}, {duration:.2f}s")
    
    # Setup output video writer if output directory specified
    output_video_path = None
    video_writer = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_video_path = os.path.join(output_dir, f"detected_{os.path.basename(video_path)}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"💾 Output video will be saved to: {output_video_path}")
    
    # Process frames
    frame_idx = 0
    total_detections = 0
    
    print(f"🔍 Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze frame
        labels, boxes = model.analyze_frame(frame)
        detections = len(labels)
        total_detections += detections
        
        # Draw detections
        if detections > 0:
            frame_with_detections = drawer.draw_detections(frame, labels, boxes, model.id2label)
        else:
            frame_with_detections = frame.copy()
        
        # Save to output video
        if video_writer:
            video_writer.write(frame_with_detections)
        
        # Display progress
        if frame_idx % 30 == 0:  # Every 30 frames
            timestamp = frame_idx / fps
            print(f"  Frame {frame_idx:4d} ({timestamp:6.2f}s): {detections} detections")
        
        # Show frame (optional - press 'q' to quit)
        cv2.imshow('Object Detection', frame_with_detections)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_idx += 1
    
    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    # Summary
    avg_detections = total_detections / max(1, frame_idx)
    print(f"\n✅ Processing complete!")
    print(f"📊 Summary:")
    print(f"   - Processed {frame_idx} frames")
    print(f"   - Total detections: {total_detections}")
    print(f"   - Average detections per frame: {avg_detections:.2f}")
    if output_video_path:
        print(f"   - Output saved to: {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description='Object Detection on Video')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('--model', choices=['detr', 'yolo'], default='yolo', 
                       help='Model type to use (default: yolo)')
    parser.add_argument('--model_size', choices=['n', 's', 'm', 'l', 'x'], default='n',
                       help='YOLOv8 model size (default: n)')
    parser.add_argument('--output', '-o', help='Output directory for processed video')
    
    args = parser.parse_args()
    
    try:
        simple_video_detection(
            video_path=args.video_path,
            model_type=args.model,
            model_size=args.model_size,
            output_dir=args.output
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
