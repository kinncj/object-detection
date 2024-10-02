# Copyright (c) 2024 Kinn Coelho Juliao <kinncj@gmail.com>
# All rights reserved.
#
# This software is licensed under the terms of the MIT License.
# See the LICENSE file in the project root for license terms.
import cv2
from config import RESTRICTED_CLASSES, RESTRICTED_COLORS

class DetectionDrawer:
    def draw_detections(self, frame, labels, boxes, id2label):
        h, w, _ = frame.shape
        for i in range(len(labels)):
            label_id = labels[i]
            label = id2label[label_id]
            if label not in RESTRICTED_CLASSES.values():
                continue

            box = boxes[i]
            x_center, y_center, box_width, box_height = box
            x1 = int((x_center - box_width / 2) * w)
            y1 = int((y_center - box_height / 2) * h)
            x2 = int((x_center + box_width / 2) * w)
            y2 = int((y_center + box_height / 2) * h)

            color = RESTRICTED_COLORS.get(label, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return frame
import cv2
from config import RESTRICTED_CLASSES, RESTRICTED_COLORS

class DetectionDrawer:
    def draw_detections(self, frame, labels, boxes, id2label):
        """
        Draws detections on a given frame.

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
            # if coco_label not in RESTRICTED_CLASSES.values():
            #     continue

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