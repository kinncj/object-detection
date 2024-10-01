import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from detection.model import DETRModel


class TestDETRModel(unittest.TestCase):
    @patch('detection.model.DetrForObjectDetection.from_pretrained')
    @patch('detection.model.DetrImageProcessor.from_pretrained')
    def test_initializes_DETRModel_successfully(self, mock_image_processor, mock_model):
        mock_model.return_value = MagicMock()
        mock_image_processor.return_value = MagicMock()

        model = DETRModel()

        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.image_processor)
        self.assertIsNotNone(model.id2label)

    @patch('detection.model.DetrForObjectDetection.from_pretrained')
    @patch('detection.model.DetrImageProcessor.from_pretrained')
    def test_raises_error_if_model_loading_fails(self, mock_image_processor, mock_model):
        mock_model.side_effect = Exception("Model loading error")

        with self.assertRaises(RuntimeError):
            DETRModel()

    @patch('detection.model.DetrForObjectDetection.from_pretrained')
    @patch('detection.model.DetrImageProcessor.from_pretrained')
    def test_analyzes_frame_correctly(self, mock_image_processor, mock_model):
        mock_model.return_value = MagicMock()
        mock_image_processor.return_value = MagicMock()
        model = DETRModel()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        model.model.return_value.logits.softmax.return_value = torch.tensor([[[0.1, 0.9]]])
        model.model.return_value.pred_boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])

        labels, boxes = model.analyze_frame(frame)

        self.assertEqual(labels, [])
        self.assertTrue((boxes == np.array([[0.5, 0.5, 0.2, 0.2]])).all())

    @patch('detection.model.DetrForObjectDetection.from_pretrained')
    @patch('detection.model.DetrImageProcessor.from_pretrained')
    def test_filters_restricted_classes_correctly(self, mock_image_processor, mock_model):
        mock_model.return_value = MagicMock()
        mock_image_processor.return_value = MagicMock()
        model = DETRModel()

        labels = np.array([1, 2, 77])
        boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.6, 0.6, 0.3, 0.3], [0.7, 0.7, 0.4, 0.4]])

        filtered_labels, filtered_boxes = model._filter_restricted_classes(labels, boxes)

        self.assertEqual(filtered_labels, [1, 77])
        self.assertTrue((filtered_boxes == np.array([[0.5, 0.5, 0.2, 0.2], [0.7, 0.7, 0.4, 0.4]])).all())


if __name__ == '__main__':
    unittest.main()