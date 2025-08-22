#!/usr/bin/env python3
"""
Object Detection Application.

This application performs real-time object detection on video files using
state-of-the-art models like YOLOv8 and DETR. It processes video frames,
detects objects, and outputs annotated videos with bounding boxes.

Usage:
    python main.py <video_path> [options]

Example:
    python main.py video.mp4 --model yolo --model-size n --output ./results/
"""
# Copyright (c) 2024 Kinn Coelho Juliao <kinncj@gmail.com>
# All rights reserved.
#
# This software is licensed under the terms of the MIT License.
# See the LICENSE file in the project root for license terms.

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from models import ModelFactory
from detection.drawer import DetectionDrawer


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Object Detection Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4
  %(prog)s video.mp4 --model yolo --model-size s
  %(prog)s video.mp4 --model detr --confidence 0.8
  %(prog)s video.mp4 --output ./results/ --display
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
        choices=["yolo", "detr"],
        default="yolo",
        help="Object detection model to use (default: yolo)"
    )
    
    parser.add_argument(
        "--model-size", 
        choices=["n", "s", "m", "l", "x"],
        default="n",
        help="Model size for YOLO (n=nano, s=small, m=medium, l=large, x=extra-large, default: n)"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output", 
        default="./output",
        help="Output directory for processed video (default: ./output)"
    )
    
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display video during processing"
    )
    
    parser.add_argument(
        "--info",
        action="store_true", 
        help="Show frame info overlay"
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        SystemExit: If validation fails
    """
    # Check video file exists
    if not os.path.isfile(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found.", file=sys.stderr)
        sys.exit(1)
    
    # Check confidence threshold
    if not 0 <= args.confidence <= 1:
        print(f"Error: Confidence must be between 0 and 1, got {args.confidence}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)


def process_video(
    video_path: str,
    model_type: str,
    model_size: str,
    confidence: float,
    output_dir: str,
    display: bool = False,
    show_info: bool = False
) -> None:
    """
    Process video with object detection.
    
    Args:
        video_path: Path to input video
        model_type: Type of model ('yolo' or 'detr')
        model_size: Model size (for YOLO)
        confidence: Confidence threshold
        output_dir: Output directory
        display: Whether to display video during processing
        show_info: Whether to show frame info overlay
    """
    # Initialize model
    print(f"🚀 Initializing {model_type.upper()} model...")
    model = ModelFactory.create_model(
        model_type=model_type,
        model_size=model_size,
        confidence_threshold=confidence
    )
    
    # Print model info
    model_info = model.get_model_info()
    print(f"📋 Model: {model_info}")
    
    # Initialize drawer
    drawer = DetectionDrawer()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"📺 Video: {total_frames} frames, {fps:.2f} FPS, {width}x{height}, {duration:.2f}s")
    
    # Setup output video
    video_name = Path(video_path).stem
    output_path = os.path.join(output_dir, f"detected_{video_name}.mp4")
    print(f"💾 Output video will be saved to: {output_path}")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Processing statistics
    frame_count = 0
    total_detections = 0
    processing_times = []
    
    print("🔍 Processing frames...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Detect objects
            detections = model.detect_objects(frame)
            detections.frame_id = frame_count
            
            # Draw detections
            frame_with_detections = drawer.draw_detections(
                frame, 
                detections,
                show_confidence=True
            )
            
            # Optionally draw frame info
            if show_info:
                frame_with_detections = drawer.draw_frame_info(
                    frame_with_detections, 
                    detections,
                    "top-left"
                )
            
            # Write frame to output video
            out.write(frame_with_detections)
            
            # Update statistics
            frame_count += 1
            total_detections += detections.detection_count
            processing_times.append(detections.processing_time)
            
            # Print progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                avg_time = sum(processing_times[-30:]) / min(30, len(processing_times))
                print(f"  Frame {frame_count:4d} ({progress:5.1f}%): "
                      f"{detections.detection_count} detections, "
                      f"avg {avg_time:.3f}s/frame")
            
            # Display if requested
            if display:
                cv2.imshow('Object Detection', frame_with_detections)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()
    
    # Print final statistics
    avg_processing_time = sum(processing_times) / len(processing_times)
    avg_detections_per_frame = total_detections / frame_count
    
    print("\n✅ Processing complete!")
    print(f"📊 Summary:")
    print(f"   - Processed {frame_count} frames")
    print(f"   - Total detections: {total_detections}")
    print(f"   - Average detections per frame: {avg_detections_per_frame:.2f}")
    print(f"   - Average processing time: {avg_processing_time:.3f}s/frame")
    print(f"   - Output saved to: {output_path}")


def main() -> None:
    """Main application entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    validate_arguments(args)
    
    try:
        process_video(
            video_path=args.video_path,
            model_type=args.model,
            model_size=args.model_size,
            confidence=args.confidence,
            output_dir=args.output,
            display=args.display,
            show_info=args.info
        )
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
