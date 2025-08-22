# Troubleshooting Guide

This document provides solutions to common issues you may encounter while using the Object Detection application.

## 🔧 Installation Issues

### Issue: Conda Environment Creation Fails

**Symptoms:**
```bash
CondaError: Could not find conda environment: object-detection
```

**Solutions:**
1. **Check Conda Installation:**
   ```bash
   conda --version
   # If not found, install Miniconda/Anaconda
   ```

2. **Update Conda:**
   ```bash
   conda update conda
   ```

3. **Create Environment Manually:**
   ```bash
   conda create -n object-detection python=3.9 -y
   conda activate object-detection
   pip install -r requirements.txt
   ```

### Issue: CUDA Not Available

**Symptoms:**
```python
torch.cuda.is_available()  # Returns False
```

**Solutions:**
1. **Install CUDA Toolkit:**
   ```bash
   # Check NVIDIA driver
   nvidia-smi
   
   # Install CUDA-enabled PyTorch
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

2. **Verify GPU:**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU count: {torch.cuda.device_count()}")
   if torch.cuda.is_available():
       print(f"Current GPU: {torch.cuda.get_device_name(0)}")
   ```

3. **Check Environment Variables:**
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```

### Issue: Model Download Failures

**Symptoms:**
```
ConnectionError: Failed to download model
OSError: Unable to load model weights
```

**Solutions:**
1. **Check Internet Connection:**
   ```bash
   ping huggingface.co
   ping ultralytics.com
   ```

2. **Manual Model Download:**
   ```bash
   # Create model cache directory
   mkdir -p ~/.cache/huggingface/transformers/
   mkdir -p ~/.ultralytics/
   
   # Download models manually
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
   python -c "from transformers import DetrForObjectDetection; DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')"
   ```

3. **Use Different Model Cache:**
   ```bash
   export TORCH_HOME=/path/to/custom/cache
   export TRANSFORMERS_CACHE=/path/to/custom/cache
   ```

### Issue: Import Errors

**Symptoms:**
```
ImportError: No module named 'models'
ModuleNotFoundError: No module named 'ultralytics'
```

**Solutions:**
1. **Check Python Path:**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Verify Installation:**
   ```bash
   pip list | grep ultralytics
   pip list | grep transformers
   ```

3. **Reinstall Dependencies:**
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

## 🎥 Processing Issues

### Issue: Video Not Loading

**Symptoms:**
```
cv2.error: OpenCV(4.x.x) error: (-215:Assertion failed)
FileNotFoundError: [Errno 2] No such file or directory
```

**Solutions:**
1. **Check File Path:**
   ```bash
   ls -la /path/to/video.mp4
   file /path/to/video.mp4  # Check file type
   ```

2. **Verify Video Format:**
   ```python
   import cv2
   cap = cv2.VideoCapture("video.mp4")
   print(f"Video opened: {cap.isOpened()}")
   print(f"Frame count: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
   print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
   cap.release()
   ```

3. **Install Additional Codecs:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Conda
   conda install ffmpeg -c conda-forge
   ```

4. **Convert Video Format:**
   ```bash
   ffmpeg -i input.mov -c:v libx264 -c:a aac output.mp4
   ```

### Issue: No Detections Found

**Symptoms:**
```
Found 0 objects in frame
Processing complete: 0 total detections
```

**Solutions:**
1. **Lower Confidence Threshold:**
   ```bash
   python main.py video.mp4 --confidence 0.3
   ```

2. **Try Different Model:**
   ```bash
   # Try DETR for better detection
   python main.py video.mp4 --model detr
   
   # Try larger YOLO model
   python main.py video.mp4 --model yolo --model-size l
   ```

3. **Check Video Content:**
   ```bash
   # Display first frame to verify content
   python -c "
   import cv2
   cap = cv2.VideoCapture('video.mp4')
   ret, frame = cap.read()
   cv2.imwrite('first_frame.jpg', frame)
   cap.release()
   "
   ```

4. **Debug Detection:**
   ```python
   from models.factory import ModelFactory
   import cv2
   
   model = ModelFactory.create_model("yolo", model_size="n")
   frame = cv2.imread("first_frame.jpg")
   
   # Lower confidence for debugging
   model.confidence_threshold = 0.1
   detections = model.detect_objects(frame)
   
   print(f"Found {len(detections.detections)} detections")
   for det in detections.detections:
       print(f"  {det.class_name}: {det.bounding_box.confidence:.3f}")
   ```

### Issue: Slow Processing Speed

**Symptoms:**
```
Processing very slowly (< 1 FPS)
High CPU usage, low GPU usage
```

**Solutions:**
1. **Use GPU Acceleration:**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   
   # Force GPU usage
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   ```

2. **Use Faster Model:**
   ```bash
   # Use nano model for speed
   python main.py video.mp4 --model yolo --model-size n
   ```

3. **Optimize Video Resolution:**
   ```bash
   # Resize video for faster processing
   ffmpeg -i input.mp4 -vf scale=640:480 -c:a copy output_small.mp4
   ```

4. **Batch Processing:**
   ```python
   # Process every nth frame
   cap = cv2.VideoCapture("video.mp4")
   frame_skip = 2  # Process every 2nd frame
   
   frame_count = 0
   while True:
       ret, frame = cap.read()
       if not ret:
           break
       
       if frame_count % frame_skip == 0:
           detections = model.detect_objects(frame)
           # Process detections
       
       frame_count += 1
   ```

## 🧠 Memory Issues

### Issue: Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
MemoryError: Unable to allocate array
```

**Solutions:**
1. **Clear GPU Cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

2. **Use Smaller Model:**
   ```bash
   python main.py video.mp4 --model yolo --model-size n  # Smallest
   ```

3. **Reduce Batch Size:**
   ```python
   # Process frames one at a time instead of batches
   with torch.no_grad():
       detections = model.detect_objects(frame)
   ```

4. **Monitor Memory Usage:**
   ```python
   import psutil
   import torch
   
   def check_memory():
       # CPU memory
       cpu_percent = psutil.virtual_memory().percent
       print(f"CPU Memory: {cpu_percent}%")
       
       # GPU memory
       if torch.cuda.is_available():
           gpu_memory = torch.cuda.memory_allocated() / 1024**3
           gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
           print(f"GPU Memory: {gpu_memory:.1f}GB / {gpu_total:.1f}GB")
   
   check_memory()
   ```

5. **Increase System Memory:**
   ```bash
   # Increase swap space (Linux)
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Issue: Memory Leaks

**Symptoms:**
```
Memory usage increasing over time
System becoming unresponsive
```

**Solutions:**
1. **Use Context Managers:**
   ```python
   import cv2
   
   # Proper resource management
   with cv2.VideoCapture("video.mp4") as cap:
       while True:
           ret, frame = cap.read()
           if not ret:
               break
           # Process frame
   ```

2. **Explicit Cleanup:**
   ```python
   import gc
   import torch
   
   # After processing
   del model
   del detections
   gc.collect()
   
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
   ```

3. **Process in Chunks:**
   ```python
   def process_video_chunks(video_path, chunk_size=100):
       cap = cv2.VideoCapture(video_path)
       frame_count = 0
       
       while True:
           frames = []
           
           # Read chunk
           for _ in range(chunk_size):
               ret, frame = cap.read()
               if not ret:
                   break
               frames.append(frame)
           
           if not frames:
               break
           
           # Process chunk
           for frame in frames:
               detections = model.detect_objects(frame)
               # Handle detections
           
           # Clear chunk
           del frames
           gc.collect()
           
           frame_count += len(frames)
           print(f"Processed {frame_count} frames")
       
       cap.release()
   ```

## 🐛 Model Issues

### Issue: Model Loading Errors

**Symptoms:**
```
RuntimeError: Error loading model
ValueError: Unknown model type
```

**Solutions:**
1. **Verify Model Files:**
   ```bash
   ls -la ~/.ultralytics/
   ls -la ~/.cache/huggingface/transformers/
   ```

2. **Re-download Models:**
   ```python
   # Remove cached models
   import shutil
   shutil.rmtree(os.path.expanduser("~/.ultralytics"), ignore_errors=True)
   
   # Re-download
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   ```

3. **Check Model Compatibility:**
   ```python
   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA version: {torch.version.cuda}")
   
   # Check if models are compatible
   try:
       from ultralytics import YOLO
       model = YOLO("yolov8n.pt")
       print("✅ YOLO model compatible")
   except Exception as e:
       print(f"❌ YOLO error: {e}")
   ```

### Issue: Inconsistent Detection Results

**Symptoms:**
```
Detection results vary between runs
Some objects detected, then not detected
```

**Solutions:**
1. **Set Deterministic Behavior:**
   ```python
   import torch
   import random
   import numpy as np
   
   # Set random seeds
   torch.manual_seed(42)
   random.seed(42)
   np.random.seed(42)
   
   # Set deterministic behavior
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```

2. **Check Confidence Threshold:**
   ```python
   # Use consistent threshold
   model = ModelFactory.create_model("yolo", confidence_threshold=0.5)
   
   # Verify threshold
   print(f"Confidence threshold: {model.confidence_threshold}")
   ```

3. **Validate Input:**
   ```python
   # Check frame consistency
   import hashlib
   
   def frame_hash(frame):
       return hashlib.md5(frame.tobytes()).hexdigest()
   
   # Verify same frame produces same hash
   frame_hash_1 = frame_hash(frame)
   frame_hash_2 = frame_hash(frame)
   assert frame_hash_1 == frame_hash_2
   ```

## 🔧 Performance Issues

### Issue: High CPU Usage

**Symptoms:**
```
CPU usage at 100%
System becomes slow/unresponsive
```

**Solutions:**
1. **Limit CPU Threads:**
   ```bash
   export OMP_NUM_THREADS=4
   export MKL_NUM_THREADS=4
   ```

2. **Use GPU Processing:**
   ```python
   import torch
   
   # Ensure GPU usage
   if torch.cuda.is_available():
       torch.cuda.set_device(0)
       print("Using GPU for processing")
   else:
       print("Warning: Using CPU processing")
   ```

3. **Process at Lower FPS:**
   ```python
   # Skip frames for lower CPU usage
   cap = cv2.VideoCapture("video.mp4")
   original_fps = cap.get(cv2.CAP_PROP_FPS)
   target_fps = 15  # Lower target FPS
   
   frame_skip = int(original_fps / target_fps)
   
   frame_count = 0
   while True:
       ret, frame = cap.read()
       if not ret:
           break
       
       if frame_count % frame_skip == 0:
           # Process this frame
           detections = model.detect_objects(frame)
       
       frame_count += 1
   ```

### Issue: Low GPU Utilization

**Symptoms:**
```
nvidia-smi shows low GPU usage
Processing slower than expected
```

**Solutions:**
1. **Verify GPU Usage:**
   ```python
   import torch
   
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Current device: {torch.cuda.current_device()}")
   print(f"Device name: {torch.cuda.get_device_name(0)}")
   
   # Check model device
   if hasattr(model, 'model'):
       print(f"Model device: {next(model.model.parameters()).device}")
   ```

2. **Force GPU Usage:**
   ```python
   # Explicitly move model to GPU
   if torch.cuda.is_available():
       device = torch.device("cuda:0")
       model.to(device)
   ```

3. **Increase Batch Size:**
   ```python
   # Process multiple frames together (if supported)
   frames_batch = [frame1, frame2, frame3, frame4]
   # Note: Current implementation processes one frame at a time
   ```

## 🌐 Network Issues

### Issue: Model Download Timeouts

**Symptoms:**
```
ReadTimeoutError: HTTPSConnectionPool
requests.exceptions.ConnectionError
```

**Solutions:**
1. **Increase Timeout:**
   ```python
   import torch
   
   # Set longer timeout
   torch.hub.set_dir('/custom/path')
   
   # Or set environment variable
   import os
   os.environ['TORCH_HUB_TIMEOUT'] = '300'  # 5 minutes
   ```

2. **Use Mirror/Proxy:**
   ```bash
   # Use different mirror
   export HF_ENDPOINT=https://hf-mirror.com
   ```

3. **Manual Download:**
   ```bash
   # Download models manually
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   mkdir -p ~/.ultralytics/
   mv yolov8n.pt ~/.ultralytics/
   ```

## 🔍 Debugging Techniques

### Enable Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add to your processing code
logger.debug(f"Processing frame {frame_count}")
logger.debug(f"Found {len(detections.detections)} detections")
```

### Performance Profiling

```python
import cProfile
import pstats
from pstats import SortKey

def profile_detection():
    # Your detection code here
    model = ModelFactory.create_model("yolo")
    detections = model.detect_objects(frame)

# Profile the function
cProfile.run('profile_detection()', 'profile_output')

# Analyze results
p = pstats.Stats('profile_output')
p.sort_stats(SortKey.TIME)
p.print_stats(20)  # Top 20 time-consuming functions
```

### Memory Profiling

```python
import tracemalloc

# Start memory tracing
tracemalloc.start()

# Your code here
model = ModelFactory.create_model("yolo")
detections = model.detect_objects(frame)

# Get memory snapshot
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("Top 10 memory consuming lines:")
for stat in top_stats[:10]:
    print(stat)
```

## 🆘 Getting Help

### Collecting System Information

```python
# system_info.py
import torch
import cv2
import platform
import psutil

def collect_system_info():
    """Collect system information for troubleshooting."""
    
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "opencv_version": cv2.__version__,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
        })
    
    print("System Information:")
    print("=" * 40)
    for key, value in info.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    collect_system_info()
```

### Creating Bug Reports

When reporting issues, include:

1. **System Information** (use script above)
2. **Full Error Message** (with stack trace)
3. **Steps to Reproduce**
4. **Expected vs Actual Behavior**
5. **Sample Files** (if possible)

### Community Resources

- **GitHub Issues**: [Repository Issues](https://github.com/kinncj/object-detection/issues)
- **Documentation**: [docs/](./README.md)
- **Stack Overflow**: Tag with `object-detection`, `yolo`, `detr`

## 📚 Prevention Tips

### Best Practices

1. **Always Use Virtual Environments**
2. **Pin Dependency Versions**
3. **Monitor Resource Usage**
4. **Implement Proper Error Handling**
5. **Use Logging Instead of Print**
6. **Test with Sample Data First**

### Common Gotchas

1. **File Paths**: Use absolute paths when possible
2. **Memory Management**: Always release resources
3. **Model Caching**: Be aware of disk space usage
4. **GPU Memory**: Monitor and clear when needed
5. **Video Formats**: Not all formats are supported equally

### Quick Diagnostics

```bash
# Quick system check
python -c "
import torch, cv2, platform
print(f'Platform: {platform.platform()}')
print(f'Python: {platform.python_version()}')
print(f'OpenCV: {cv2.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# Test basic functionality
python tests/test_architecture.py
python tests/test_simple.py tests/test_video.mp4
```

This troubleshooting guide should help you resolve most common issues. If you encounter problems not covered here, please check the [GitHub Issues](https://github.com/kinncj/object-detection/issues) or create a new issue with detailed information.
