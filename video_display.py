import cv2
import numpy as np
import sys
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

# Check for MPS support
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Check if video file path is provided
if len(sys.argv) != 2:
    print("Usage: python video_display.py <video_file.mp4>")
    sys.exit(1)

video_file_path = sys.argv[1]

# Load the DETR model and processor from Hugging Face
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Initialize video capture from the provided file
cap = cv2.VideoCapture(video_file_path)

def detect_objects(frame):
    # Process the frame for the model
    inputs = processor(images=frame, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # Get the predicted boxes and scores
    target_sizes = torch.tensor([frame.shape[:2]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
        if score > 0.5:  # Check confidence threshold
            box = box.detach().cpu().numpy().astype(int)  # Move to CPU for NumPy conversion
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.putText(frame, f"{model.config.id2label[label.item()]}: {score:.2f}",
                        (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in video frame
    frame_with_objects = detect_objects(frame)
    cv2.imshow("Video Feed", frame_with_objects)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
