# Copyright (c) 2024 Kinn Coelho Juliao <kinncj@gmail.com>
# All rights reserved.
#
# This software is licensed under the terms of the MIT License.
# See the LICENSE file in the project root for license terms.
from abc import ABC, abstractmethod
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
import cv2
from config.config import device, RESTRICTED_CLASSES

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