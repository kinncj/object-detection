#!/usr/bin/env python3
"""
Object Detection Application

A command-line application for performing object detection on videos
using various state-of-the-art models including DETR and YOLOv8.
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
from pathlib import Path
from typing import Optional

from models.factory import ModelFactory
from detection.drawer import DetectionDrawer
from processor.frame_processor import FrameProcessor


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


def main() -> int:
    """
    Main function to perform object detection on a video.
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Object Detection Application",
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
        help="Model size for YOLO models: n(ano), s(mall), m(edium), l(arge), x(tra-large) (default: n)"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Confidence threshold for detections (default: 0.5 for YOLO, 0.9 for DETR)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output",
        default="./output",
        help="Output directory for processed video and images (default: ./output)"
    )
    
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=1,
        help="Frame extraction rate in milliseconds (default: 1)"
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
        help="Automatically open the result video when processing is complete"
    )
    
    # Visualization options
    parser.add_argument(
        "--show-confidence",
        action="store_true",
        default=True,
        help="Show confidence scores in detection labels"
    )
    
    parser.add_argument(
        "--show-class-id",
        action="store_true",
        help="Show class IDs in detection labels"
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
        
        # Create drawer and processor
        drawer = DetectionDrawer()
        processor = FrameProcessor(model, drawer)
        
        # Process video
        print(f"🎬 Processing video: {args.video_path}")
        
        try:
            fps, frames, audio = processor.extract_video_fragments(
                args.video_path, 
                args.frame_rate
            )
            print(f"📺 Extracted {len(frames)} frames at {fps:.2f} FPS")
            
            # Process each frame
            total_detections = 0
            total_processing_time = 0.0
            
            image_path = args.output if args.save_images else None
            
            print("🔍 Processing frames...")
            for idx, frame in enumerate(frames):
                detections = processor.process_frame(
                    frame, 
                    idx, 
                    args.display, 
                    image_path
                )
                
                total_detections += detections.detection_count
                total_processing_time += detections.processing_time
                
                # Progress update every 30 frames
                if idx % 30 == 0:
                    timestamp = idx / fps if fps > 0 else 0
                    print(f"  Frame {idx:4d} ({timestamp:6.2f}s): {detections.detection_count} detections")
            
            # Compile output video
            print("🎞️  Compiling output video...")
            output_video_path, output_with_audio_path = processor.compile_video(
                frames, args.output, fps, audio
            )
            
            # Summary
            print("\n✅ Processing complete!")
            print(f"📊 Summary:")
            print(f"   - Processed {len(frames)} frames")
            print(f"   - Total detections: {total_detections}")
            print(f"   - Average detections per frame: {total_detections/len(frames):.2f}")
            print(f"   - Total processing time: {total_processing_time:.2f}s")
            print(f"   - Average time per frame: {total_processing_time/len(frames):.3f}s")
            print(f"💾 Output saved to: {output_with_audio_path or output_video_path}")
            
            # Open result if requested
            if args.open_result and (output_with_audio_path or output_video_path):
                result_path = output_with_audio_path or output_video_path
                print(f"🚀 Opening result video: {result_path}")
                open_video(result_path)
                
        except Exception as e:
            print(f"Error processing video: {e}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
