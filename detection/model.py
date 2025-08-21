# Copyright (c) 2024 Kinn Coelho Juliao <kinncj@gmail.com>
# All rights reserved.
#
# This software is licensed under the terms of the MIT License.
# See the LICENSE file in the project root for license terms.
from abc import ABC, abstractmethod
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
import cv2
import numpy as np
from config.config import device, RESTRICTED_CLASSES

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class ObjectDetectionModel(ABC):
    @abstractmethod
    def analyze_frame(self, frame):
        """
        Analyzes a frame to detect objects.

        Args:
            frame: The frame to analyze.

        Returns:
            A tuple containing the labels and bounding boxes of detected objects.
        """
        pass

class DETRModel(ObjectDetectionModel):
    def __init__(self):
        """
        Initializes the DETRModel by loading the pre-trained DETR model and image processor.
        """
        try:
            self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
            self.image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self.id2label = self.model.config.id2label
        except Exception as e:
            raise RuntimeError(f"Error loading DETR model: {e}")

    def analyze_frame(self, frame):
        """
        Analyzes a frame to detect objects using the DETR model.

        Args:
            frame: The frame to analyze.

        Returns:
            A tuple containing the filtered labels and bounding boxes of detected objects.
        """
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.image_processor(images=img, return_tensors="pt").to(device)
        outputs = self.model(**inputs)

        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.9  # Confidence threshold
        boxes = outputs.pred_boxes[0, keep].cpu().detach().numpy()
        labels = probas[keep].argmax(-1).cpu().detach().numpy()

        return self._filter_restricted_classes(labels, boxes)

    def _filter_restricted_classes(self, labels, boxes):
        """
        Filters out restricted classes from the detected objects.

        Args:
            labels: The labels of the detected objects.
            boxes: The bounding boxes of the detected objects.

        Returns:
            A tuple containing the filtered labels and bounding boxes.
        """
        filtered_labels = [label for label in labels if label in RESTRICTED_CLASSES.keys()]
        filtered_boxes = boxes[[i for i, label in enumerate(labels) if label in RESTRICTED_CLASSES.keys()]]
        return filtered_labels, filtered_boxes

    def get_model_info(self):
        """
        Returns information about the loaded model.
        """
        return {
            "model_type": "DETR",
            "model_name": "facebook/detr-resnet-50",
            "total_classes": len(self.id2label),
            "restricted_classes": len(RESTRICTED_CLASSES),
            "class_mapping": RESTRICTED_CLASSES
        }


class YOLOv8Model(ObjectDetectionModel):
    def __init__(self, model_size="n"):
        """
        Initializes the YOLOv8Model by loading the pre-trained YOLOv8 model from Ultralytics.
        
        Args:
            model_size (str): Size of the YOLOv8 model. Options: 'n', 's', 'm', 'l', 'x'
                            - n: nano (fastest, least accurate)
                            - s: small
                            - m: medium  
                            - l: large
                            - x: extra large (slowest, most accurate)
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("Ultralytics YOLO is not installed. Please install it using: pip install ultralytics")
        
        # Validate model size
        valid_sizes = ['n', 's', 'm', 'l', 'x']
        if model_size not in valid_sizes:
            raise ValueError(f"Invalid model size '{model_size}'. Choose from: {valid_sizes}")
            
        self.model_size = model_size
        model_name = f"yolov8{model_size}.pt"
        
        try:
            print(f"Loading YOLOv8{model_size.upper()} model...")
            # Load YOLOv8 model - this will download from Ultralytics automatically
            self.model = YOLO(model_name)
            
            # YOLO class names (COCO dataset with 80 classes)
            self.class_names = self.model.names
            print(f"YOLOv8{model_size.upper()} model loaded successfully with {len(self.class_names)} classes")
            
            # Create a mapping from YOLO class IDs to DETR-compatible IDs for consistency
            self._create_class_mapping()
            print(f"Class mapping created for {len(self.yolo_to_detr)} restricted classes")
            
        except Exception as e:
            raise RuntimeError(f"Error loading YOLOv8 model: {e}")

    def _create_class_mapping(self):
        """
        Creates a mapping between YOLO class names and DETR class IDs for consistency.
        YOLO uses COCO dataset class names, DETR uses different class IDs.
        """
        # YOLO COCO class names to DETR class ID mapping for restricted classes
        yolo_to_detr_mapping = {
            'person': 1,           # YOLO class 0 -> DETR class 1
            'cell phone': 77,      # YOLO class 67 -> DETR class 77  
            'laptop': 73,          # YOLO class 63 -> DETR class 73
            'tv': 72,              # YOLO class 62 -> DETR class 72
            'keyboard': 76,        # YOLO class 66 -> DETR class 76
            'mouse': 74,           # YOLO class 64 -> DETR class 74
            'clock': 85            # YOLO class 74 -> DETR class 85
        }
        
        self.yolo_to_detr = {}
        
        # Create reverse mapping: YOLO class ID -> DETR class ID
        for yolo_id, class_name in self.class_names.items():
            if class_name in yolo_to_detr_mapping:
                self.yolo_to_detr[yolo_id] = yolo_to_detr_mapping[class_name]
                print(f"  Mapped YOLO class {yolo_id} ('{class_name}') -> DETR class {yolo_to_detr_mapping[class_name]}")
        
        # Create id2label mapping for compatibility with DETR interface
        self.id2label = {}
        for yolo_id, detr_id in self.yolo_to_detr.items():
            self.id2label[detr_id] = self.class_names[yolo_id]

    def analyze_frame(self, frame):
        """
        Analyzes a frame to detect objects using the YOLOv8 model.

        Args:
            frame: The frame to analyze (numpy array).

        Returns:
            A tuple containing the filtered labels and bounding boxes of detected objects.
        """
        try:
            # Run YOLOv8 inference with confidence threshold
            results = self.model(frame, conf=0.5, verbose=False)  # confidence threshold 0.5
            
            # Check if any detections were found
            if len(results) == 0 or len(results[0].boxes) == 0:
                return [], np.array([])
            
            boxes = results[0].boxes
            labels = []
            bboxes = []
            
            # Process each detection
            for box in boxes:
                # Get class ID from YOLO prediction
                yolo_class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                
                # Only process if it's a restricted class we care about
                if yolo_class_id in self.yolo_to_detr:
                    detr_class_id = self.yolo_to_detr[yolo_class_id]
                    labels.append(detr_class_id)
                    
                    # Convert bbox format from xyxy to normalized xywh (DETR format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    h, w = frame.shape[:2]
                    
                    # Convert to center coordinates and normalize (0-1 range)
                    center_x = (x1 + x2) / 2 / w
                    center_y = (y1 + y2) / 2 / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    bboxes.append([center_x, center_y, width, height])
                    
                    class_name = self.class_names[yolo_class_id]
                    print(f"  Detected {class_name} (confidence: {confidence:.2f})")
            
            return labels, np.array(bboxes) if bboxes else np.array([])
        
        except Exception as e:
            print(f"Error during YOLOv8 inference: {e}")
            return [], np.array([])

    def get_model_info(self):
        """
        Returns information about the loaded model.
        """
        return {
            "model_type": "YOLOv8",
            "model_size": self.model_size,
            "model_name": f"yolov8{self.model_size}.pt",
            "total_classes": len(self.class_names),
            "restricted_classes": len(self.yolo_to_detr),
            "class_mapping": self.yolo_to_detr
        }


def create_model(model_type="detr", model_size="n"):
    """
    Factory function to create the appropriate object detection model.
    
    Args:
        model_type (str): Type of model to create ("detr" or "yolo")
        model_size (str): For YOLO models, size variant ('n', 's', 'm', 'l', 'x')
                         For DETR models, this parameter is ignored
    
    Returns:
        ObjectDetectionModel: The requested model instance
    """
    if model_type.lower() == "yolo":
        return YOLOv8Model(model_size=model_size)
    elif model_type.lower() == "detr":
        return DETRModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: 'detr', 'yolo'")