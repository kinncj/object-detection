import unittest
import numpy as np
import cv2
from detection.drawer import DetectionDrawer

class TestDetectionDrawer(unittest.TestCase):
    def test_draws_detections_correctly(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        labels = [1, 77]
        boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.7, 0.7, 0.3, 0.3]])
        id2label = {1: "person", 77: "cell phone"}

        drawer = DetectionDrawer()
        result_frame = drawer.draw_detections(frame, labels, boxes, id2label)

        self.assertIsNotNone(result_frame)
        self.assertEqual(result_frame.shape, (480, 640, 3))

    def test_skips_unrestricted_classes(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        labels = [2, 3]
        boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.7, 0.7, 0.3, 0.3]])
        id2label = {2: "dog", 3: "cat"}

        drawer = DetectionDrawer()
        result_frame = drawer.draw_detections(frame, labels, boxes, id2label)

        self.assertIsNotNone(result_frame)
        self.assertEqual(result_frame.shape, (480, 640, 3))

    def test_handles_empty_labels_and_boxes(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        labels = []
        boxes = np.array([])
        id2label = {}

        drawer = DetectionDrawer()
        result_frame = drawer.draw_detections(frame, labels, boxes, id2label)

        self.assertIsNotNone(result_frame)
        self.assertEqual(result_frame.shape, (480, 640, 3))

if __name__ == '__main__':
    unittest.main()