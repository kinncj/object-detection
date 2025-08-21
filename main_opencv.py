#!/usr/bin/env python3
"""
Simplified Object Detection Application (OpenCV only)

A streamlined version of the object detection application that uses only OpenCV
for video processing, avoiding moviepy dependencies.
"""
# Copyright (c) 2024 Kinn Coelho Juliao <kinncj@gmail.com>
# All rights reserved.
#
# This software is licensed under the terms of the MIT License.
# See the LICENSE file in the project root for license terms.

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np

from models.factory import ModelFactory
from detection.drawer import DetectionDrawer
from models.base import FrameDetections


def create_output_directory(output_path: str) -> None:
    """Create output directory if it doesn't exist."""
    Path(output_path).mkdir(parents=True, exist_ok=True)


def open_video(video_path: str) -> None:
    """Open video file with the default system application."""
    try:
        if platform.system() == "Darwin":  # macOS
            subprocess.run(["open", video_path], check=True)
        elif platform.system() == "Windows":
            subprocess.run(["start", video_path], shell=True, check=True)
        else:  # Linux
            subprocess.run(["xdg-open", video_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Could not open video automatically: {e}")
        print(f"Please open manually: {video_path}")


def extract_frames(video_path: str) -> tuple:
    """Extract frames from video using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📺 Video: {frame_count} frames, {fps:.2f} FPS, {width}x{height}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames, fps, width, height


def save_video(frames: List[np.ndarray], output_path: str, fps: float) -> str:
    """Save processed frames as video."""
    if not frames:
        return None
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    base_name = Path(output_path).stem
    output_file = os.path.join(output_path, f"detected_{base_name}.mp4")
    
    writer = cv2.VideoWriter(output_file, fourcc, fps, (w, h))
    
    for frame in frames:
        writer.write(frame)
    
    writer.release()
    return output_file


def main() -> int:
    """
    Main function to perform object detection on a video.
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Object Detection Application (OpenCV only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4                                    # Use default YOLOv8n
  %(prog)s video.mp4 --model yolo --model-size s        # Use YOLOv8s
  %(prog)s video.mp4 --model detr                       # Use DETR
  %(prog)s video.mp4 --output ./results --display       # Save and display
        """
    )
    
    # Required arguments
    parser.add_argument(
        "video_path",
        help="Path to the input video file"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        choices=["yolo", "yolov8", "detr"],
        default="yolo",
        help="Object detection model to use (default: yolo)"
    )
    
    parser.add_argument(
        "--model-size",
        choices=["n", "s", "m", "l", "x"],
        default="n",
        help="Model size for YOLO models (default: n)"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Confidence threshold for detections"
    )
    
    # Output configuration
    parser.add_argument(
        "--output",
        default="./output",
        help="Output directory (default: ./output)"
    )
    
    # Display options
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display video with detections in real-time"
    )
    
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save individual frames with detections"
    )
    
    parser.add_argument(
        "--open-result",
        action="store_true",
        help="Automatically open the result video"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    try:
        # Create output directory
        create_output_directory(args.output)
        
        # Create model
        print(f"🤖 Initializing {args.model.upper()} model...")
        model_kwargs = {}
        if args.confidence is not None:
            model_kwargs['confidence_threshold'] = args.confidence
        
        model = ModelFactory.create_model(
            model_type=args.model,
            model_size=args.model_size,
            **model_kwargs
        )
        
        # Print model information
        model_info = model.get_model_info()
        print(f"📋 Model: {model.model_name}")
        print(f"    Type: {model_info['model_type']}")
        if 'model_size' in model_info:
            print(f"    Size: {model_info['model_size']}")
        print(f"    Confidence: {model_info.get('confidence_threshold', 'default')}")
        print(f"    Classes: {model_info['restricted_classes']} restricted classes")
        
        # Create drawer
        drawer = DetectionDrawer()
        
        # Extract frames
        print(f"🎬 Processing video: {args.video_path}")
        frames, fps, width, height = extract_frames(args.video_path)
        print(f"📺 Extracted {len(frames)} frames")
        
        # Process frames
        processed_frames = []
        total_detections = 0
        total_processing_time = 0.0
        
        print("🔍 Processing frames...")
        for idx, frame in enumerate(frames):
            start_time = time.time()
            
            # Detect objects
            detections = model.detect_objects(frame)
            detections.frame_id = idx
            processing_time = time.time() - start_time
            
            total_detections += detections.detection_count
            total_processing_time += processing_time
            
            # Draw detections
            frame_with_detections = drawer.draw_detections(frame, detections)
            frame_with_detections = drawer.draw_frame_info(
                frame_with_detections, detections, "top-left"
            )
            
            processed_frames.append(frame_with_detections)
            
            # Save individual frame if requested
            if args.save_images:
                frame_path = os.path.join(args.output, f"frame_{idx:04d}.png")
                cv2.imwrite(frame_path, frame_with_detections)
            
            # Display frame if requested
            if args.display:
                cv2.imshow("Object Detection", frame_with_detections)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress update
            if idx % 30 == 0:
                timestamp = idx / fps if fps > 0 else 0
                print(f"  Frame {idx:4d} ({timestamp:6.2f}s): {detections.detection_count} detections")
        
        if args.display:
            cv2.destroyAllWindows()
        
        # Save output video
        print("🎞️  Saving output video...")
        output_video_path = save_video(processed_frames, args.output, fps)
        
        # Summary
        print("\n✅ Processing complete!")
        print(f"📊 Summary:")
        print(f"   - Processed {len(frames)} frames")
        print(f"   - Total detections: {total_detections}")
        print(f"   - Average detections per frame: {total_detections/len(frames):.2f}")
        print(f"   - Total processing time: {total_processing_time:.2f}s")
        print(f"   - Average time per frame: {total_processing_time/len(frames):.3f}s")
        
        if output_video_path:
            print(f"💾 Output saved to: {output_video_path}")
            
            # Open result if requested
            if args.open_result:
                print(f"🚀 Opening result video: {output_video_path}")
                open_video(output_video_path)
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
