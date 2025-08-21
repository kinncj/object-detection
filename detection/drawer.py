"""
Object detection visualization and drawing utilities.

This module provides classes for drawing detection results on images
and creating visual annotations for object detection outputs.
"""
# Copyright (c) 2024 Kinn Coelho Juliao <kinncj@gmail.com>
# All rights reserved.
#
# This software is licensed under the terms of the MIT License.
# See the LICENSE file in the project root for license terms.

from typing import List, Tuple, Optional
import cv2
import numpy as np

from models.base import DetectionResult, FrameDetections
from config.config import RESTRICTED_COLORS


class DetectionDrawer:
    """
    Utility class for drawing object detection results on images.
    
    This class provides methods to visualize bounding boxes, labels,
    and confidence scores on detected objects.
    """
    
    def __init__(
        self,
        box_color: Tuple[int, int, int] = (0, 255, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        box_thickness: int = 2,
        font_scale: float = 0.6,
        font_thickness: int = 2
    ):
        """
        Initialize the detection drawer.
        
        Args:
            box_color (Tuple[int, int, int]): BGR color for bounding boxes
            text_color (Tuple[int, int, int]): BGR color for text labels
            box_thickness (int): Thickness of bounding box lines
            font_scale (float): Scale factor for text size
            font_thickness (int): Thickness of text lines
        """
        self.box_color = box_color
        self.text_color = text_color
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: FrameDetections,
        show_confidence: bool = True,
        show_class_id: bool = False
    ) -> np.ndarray:
        """
        Draw all detections on an image.
        
        Args:
            image (np.ndarray): Input image to draw on
            detections (FrameDetections): Detection results to visualize
            show_confidence (bool): Whether to show confidence scores
            show_class_id (bool): Whether to show class IDs
            
        Returns:
            np.ndarray: Image with detections drawn
        """
        result_image = image.copy()
        h, w = image.shape[:2]
        
        for detection in detections.detections:
            result_image = self._draw_single_detection(
                result_image, detection, w, h, 
                show_confidence, show_class_id
            )
        
        return result_image
    
    def _draw_single_detection(
        self,
        image: np.ndarray,
        detection: DetectionResult,
        image_width: int,
        image_height: int,
        show_confidence: bool,
        show_class_id: bool
    ) -> np.ndarray:
        """
        Draw a single detection on an image.
        
        Args:
            image (np.ndarray): Image to draw on
            detection (DetectionResult): Detection to draw
            image_width (int): Width of the image
            image_height (int): Height of the image
            show_confidence (bool): Whether to show confidence score
            show_class_id (bool): Whether to show class ID
            
        Returns:
            np.ndarray: Image with detection drawn
        """
        # Get bounding box coordinates
        x1, y1, x2, y2 = detection.bbox.to_xyxy(image_width, image_height)
        
        # Get color for this class
        color = RESTRICTED_COLORS.get(detection.class_name, self.box_color)
        
        # Draw bounding box
        cv2.rectangle(
            image, 
            (x1, y1), 
            (x2, y2), 
            color, 
            self.box_thickness
        )
        
        # Prepare label text
        label_parts = [detection.class_name]
        
        if show_class_id:
            label_parts.append(f"ID:{detection.class_id}")
        
        if show_confidence:
            label_parts.append(f"{detection.confidence:.2f}")
        
        label = " | ".join(label_parts)
        
        # Calculate text size and position
        text_size = cv2.getTextSize(
            label, 
            self.font, 
            self.font_scale, 
            self.font_thickness
        )[0]
        
        # Draw text background
        text_bg_x1 = x1
        text_bg_y1 = y1 - text_size[1] - 10
        text_bg_x2 = x1 + text_size[0] + 5
        text_bg_y2 = y1
        
        # Ensure text background is within image bounds
        text_bg_y1 = max(0, text_bg_y1)
        text_bg_x2 = min(image_width, text_bg_x2)
        
        cv2.rectangle(
            image,
            (text_bg_x1, text_bg_y1),
            (text_bg_x2, text_bg_y2),
            color,
            -1  # Filled rectangle
        )
        
        # Draw text
        text_x = x1 + 2
        text_y = y1 - 5
        text_y = max(text_size[1], text_y)  # Ensure text is visible
        
        cv2.putText(
            image,
            label,
            (text_x, text_y),
            self.font,
            self.font_scale,
            self.text_color,
            self.font_thickness
        )
        
        return image
    
    def draw_frame_info(
        self,
        image: np.ndarray,
        detections: FrameDetections,
        position: str = "top-left"
    ) -> np.ndarray:
        """
        Draw frame information (detection count, processing time, etc.).
        
        Args:
            image (np.ndarray): Image to draw on
            detections (FrameDetections): Detection results
            position (str): Position for the info text
            
        Returns:
            np.ndarray: Image with frame info drawn
        """
        info_text = [
            f"Frame: {detections.frame_id}",
            f"Detections: {detections.detection_count}",
            f"Time: {detections.processing_time:.3f}s"
        ]
        
        if detections.model_info:
            model_type = detections.model_info.get('model_type', 'Unknown')
            info_text.append(f"Model: {model_type}")
        
        return self._draw_text_block(image, info_text, position)
    
    def _draw_text_block(
        self,
        image: np.ndarray,
        text_lines: List[str],
        position: str
    ) -> np.ndarray:
        """
        Draw a block of text at the specified position.
        
        Args:
            image (np.ndarray): Image to draw on
            text_lines (List[str]): Lines of text to draw
            position (str): Position for the text block
            
        Returns:
            np.ndarray: Image with text block drawn
        """
        h, w = image.shape[:2]
        line_height = 25
        margin = 10
        
        # Calculate text positions
        if position == "top-left":
            start_x, start_y = margin, margin + line_height
        elif position == "top-right":
            start_x, start_y = w - 200, margin + line_height
        elif position == "bottom-left":
            start_x, start_y = margin, h - len(text_lines) * line_height - margin
        elif position == "bottom-right":
            start_x, start_y = w - 200, h - len(text_lines) * line_height - margin
        else:
            start_x, start_y = margin, margin + line_height
        
        # Draw text lines
        for i, line in enumerate(text_lines):
            y = start_y + i * line_height
            cv2.putText(
                image,
                line,
                (start_x, y),
                self.font,
                self.font_scale * 0.8,  # Slightly smaller for info text
                (255, 255, 255),  # White text
                self.font_thickness
            )
        
        return image
    
    # Legacy method for backward compatibility
    def draw_detections_legacy(self, frame, labels, boxes, id2label):
        """
        Legacy method for backward compatibility.
        
        Args:
            frame: The frame on which to draw the detections.
            labels (list): List of label IDs for the detected objects.
            boxes (list): List of bounding boxes for the detected objects.
            id2label (dict): Dictionary mapping label IDs to label names.

        Returns:
            The frame with the detections drawn on it.
        """
        h, w, _ = frame.shape
        for i in range(len(labels)):
            label_id = labels[i]
            coco_label = id2label[label_id]

            box = boxes[i]
            x_center, y_center, box_width, box_height = box
            x1 = int((x_center - box_width / 2) * w)
            y1 = int((y_center - box_height / 2) * h)
            x2 = int((x_center + box_width / 2) * w)
            y2 = int((y_center + box_height / 2) * h)

            color = RESTRICTED_COLORS.get(coco_label, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, coco_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, str(label_id), (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return frame