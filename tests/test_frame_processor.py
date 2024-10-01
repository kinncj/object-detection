import unittest
from unittest.mock import MagicMock, patch
import cv2
import os
from processor.frame_processor import FrameProcessor

class TestFrameProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the class before any tests are run."""
        cls.video_path = os.getcwd() + "/tests/test_video.mp4"

    def test_extracts_frames_correctly(self):
        model = MagicMock()
        drawer = MagicMock()
        processor = FrameProcessor(model, drawer)
        frames = processor.extract_frames(self.video_path, frame_rate=1)

        self.assertEqual(len(frames), 15)  # Check if at least one frame was extracted

    @patch('cv2.VideoCapture')
    def test_raises_error_if_video_cannot_be_opened(self, mock_video_capture):
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = False

        model = MagicMock()
        drawer = MagicMock()
        processor = FrameProcessor(model, drawer)

        with self.assertRaises(IOError):
            processor.extract_frames("dummy_path", frame_rate=1)

    @patch('cv2.imwrite')
    def test_saves_frame_correctly(self, mock_imwrite):
        model = MagicMock()
        drawer = MagicMock()
        processor = FrameProcessor(model, drawer)
        frame = MagicMock()

        processor._save_frame(frame, 1)

        self.assertTrue(mock_imwrite.called)

    @patch('cv2.imwrite')
    def test_processes_frame_correctly(self, mock_imwrite):
        model = MagicMock()
        drawer = MagicMock()
        processor = FrameProcessor(model, drawer)
        frame = MagicMock()
        model.analyze_frame.return_value = (["label"], [[0.1, 0.1, 0.2, 0.2]])
        drawer.draw_detections.return_value = frame

        processor.process_frame(frame, 1)

        self.assertTrue(mock_imwrite.called)

if __name__ == '__main__':
    unittest.main()
