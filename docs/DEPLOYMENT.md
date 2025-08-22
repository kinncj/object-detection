# Deployment and Production Setup

This document covers deployment strategies, production configurations, and best practices for running the Object Detection application in production environments.

## 🏗️ Deployment Options

### 1. Local Server Deployment

#### Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 50GB+ storage for models and outputs

#### Setup
```bash
# Clone and setup
git clone <repository-url>
cd object-detection
./setup.sh

# Activate environment
conda activate object-detection

# Test installation
python main.py tests/test_video.mp4 --model yolo
```

### 2. Docker Deployment

#### CPU-Only Container
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directory
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONPATH=/app
ENV TORCH_HOME=/app/.torch

# Expose port for web service (if applicable)
EXPOSE 8000

# Default command
CMD ["python", "main.py", "--help"]
```

#### GPU-Enabled Container
```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Install Python
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    git \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set environment
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python3", "main.py", "--help"]
```

#### Build and Run
```bash
# Build CPU container
docker build -t object-detection:cpu -f Dockerfile.cpu .

# Build GPU container  
docker build -t object-detection:gpu -f Dockerfile.gpu .

# Run CPU container
docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output object-detection:cpu python main.py /app/input/video.mp4

# Run GPU container
docker run --gpus all -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output object-detection:gpu python main.py /app/input/video.mp4
```

### 3. Cloud Deployment

#### AWS EC2 with GPU

```bash
# Launch GPU instance (p3.2xlarge recommended)
# Install NVIDIA drivers and Docker with GPU support

# Clone repository
git clone <repository-url>
cd object-detection

# Build and run with GPU
docker build -t object-detection:gpu .
docker run --gpus all -v /data:/app/data object-detection:gpu
```

#### Google Cloud Platform

```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: object-detection
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/memory: "4Gi"
        run.googleapis.com/cpu: "2"
    spec:
      containers:
      - image: gcr.io/PROJECT_ID/object-detection
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: MODEL_TYPE
          value: "yolo"
        - name: MODEL_SIZE  
          value: "n"
```

#### Azure Container Instances

```json
{
  "location": "eastus",
  "properties": {
    "containers": [
      {
        "name": "object-detection",
        "properties": {
          "image": "your-registry/object-detection:latest",
          "resources": {
            "requests": {
              "cpu": 2,
              "memoryInGb": 4
            }
          },
          "environmentVariables": [
            {
              "name": "MODEL_TYPE",
              "value": "yolo"
            }
          ]
        }
      }
    ],
    "osType": "Linux",
    "restartPolicy": "OnFailure"
  }
}
```

## 🔧 Production Configuration

### Environment Variables

```bash
# Model Configuration
export MODEL_TYPE=yolo
export MODEL_SIZE=n
export CONFIDENCE_THRESHOLD=0.5

# Performance Settings
export CUDA_VISIBLE_DEVICES=0
export TORCH_HOME=/app/.torch
export OMP_NUM_THREADS=4

# Logging
export LOG_LEVEL=INFO
export LOG_FORMAT=json

# Security
export API_KEY=your-secret-key
export MAX_FILE_SIZE=100MB
```

### Configuration File

```yaml
# config/production.yml
application:
  name: "Object Detection Service"
  version: "1.0.0"
  environment: "production"

models:
  default_type: "yolo"
  default_size: "n"
  confidence_threshold: 0.5
  max_batch_size: 8
  
performance:
  max_workers: 4
  timeout: 300
  memory_limit: "4GB"
  
storage:
  input_path: "/app/input"
  output_path: "/app/output"
  temp_path: "/tmp"
  cleanup_after: 3600  # seconds
  
logging:
  level: "INFO"
  format: "json"
  file: "/app/logs/app.log"
  max_size: "100MB"
  backup_count: 5
  
security:
  api_key_required: true
  max_file_size: "100MB"
  allowed_extensions: [".mp4", ".avi", ".mov"]
  rate_limit: 10  # requests per minute
```

### Production Application Code

```python
# production_app.py
import os
import logging
import yaml
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import tempfile

from models.factory import ModelFactory
from detection.drawer import DetectionDrawer
import cv2

# Load configuration
with open('config/production.yml', 'r') as f:
    config = yaml.safe_load(f)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Setup logging
logging.basicConfig(
    level=getattr(logging, config['logging']['level']),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize model
model = ModelFactory.create_model(
    config['models']['default_type'],
    model_size=config['models']['default_size'],
    confidence_threshold=config['models']['confidence_threshold']
)
drawer = DetectionDrawer()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": config['application']['name'],
        "version": config['application']['version']
    })

@app.route('/detect', methods=['POST'])
def detect_objects():
    """Object detection endpoint."""
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        allowed_extensions = config['security']['allowed_extensions']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            return jsonify({"error": "Invalid file type"}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix)
        file.save(temp_input.name)
        
        # Process video
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        
        process_video_file(temp_input.name, temp_output.name)
        
        # Return processed file
        return send_file(
            temp_output.name,
            as_attachment=True,
            download_name=f"detected_{filename}"
        )
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return jsonify({"error": "Processing failed"}), 500
    
    finally:
        # Cleanup temporary files
        try:
            os.unlink(temp_input.name)
            os.unlink(temp_output.name)
        except:
            pass

def process_video_file(input_path, output_path):
    """Process video file with object detection."""
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            detections = model.detect_objects(frame)
            total_detections += len(detections.detections)
            
            # Draw detections
            annotated_frame = drawer.draw_detections(frame, detections)
            
            # Write frame
            out.write(annotated_frame)
            frame_count += 1
            
            # Log progress
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames")
                
    finally:
        cap.release()
        out.release()
    
    logger.info(f"Processing complete: {frame_count} frames, {total_detections} detections")

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 8000)),
        debug=False,
        threaded=True
    )
```

## 📊 Monitoring and Metrics

### Application Metrics

```python
# metrics.py
import time
import psutil
import torch
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Define metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')
ACTIVE_REQUESTS = Gauge('active_requests', 'Active requests')
MODEL_INFERENCE_TIME = Histogram('model_inference_seconds', 'Model inference time', ['model_type'])
GPU_MEMORY_USAGE = Gauge('gpu_memory_bytes', 'GPU memory usage')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')

def record_metrics(model_type, inference_time, gpu_memory=None):
    """Record application metrics."""
    MODEL_INFERENCE_TIME.labels(model_type=model_type).observe(inference_time)
    
    if gpu_memory:
        GPU_MEMORY_USAGE.set(gpu_memory)
    
    CPU_USAGE.set(psutil.cpu_percent())

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()
```

### Health Checks

```python
# health.py
import torch
from models.factory import ModelFactory

def check_model_health():
    """Check if models are working correctly."""
    try:
        # Test model creation
        model = ModelFactory.create_model("yolo", model_size="n")
        
        # Test inference with dummy data
        import numpy as np
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = model.detect_objects(dummy_frame)
        
        return True, "Model check passed"
    except Exception as e:
        return False, f"Model check failed: {str(e)}"

def check_gpu_availability():
    """Check GPU availability and memory."""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    try:
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        memory_allocated = torch.cuda.memory_allocated()
        memory_cached = torch.cuda.memory_reserved()
        
        return True, {
            "gpu_count": gpu_count,
            "current_device": current_device,
            "gpu_name": gpu_name,
            "memory_allocated": memory_allocated,
            "memory_cached": memory_cached
        }
    except Exception as e:
        return False, f"GPU check failed: {str(e)}"

@app.route('/health/detailed')
def detailed_health():
    """Detailed health check."""
    checks = {}
    
    # Model health
    model_ok, model_info = check_model_health()
    checks['model'] = {"status": "ok" if model_ok else "error", "details": model_info}
    
    # GPU health
    gpu_ok, gpu_info = check_gpu_availability()
    checks['gpu'] = {"status": "ok" if gpu_ok else "error", "details": gpu_info}
    
    # Overall status
    overall_status = "ok" if all([model_ok, gpu_ok]) else "error"
    
    return jsonify({
        "status": overall_status,
        "timestamp": time.time(),
        "checks": checks
    })
```

## 🔒 Security Considerations

### API Security

```python
# security.py
from functools import wraps
from flask import request, jsonify
import time
from collections import defaultdict

# Rate limiting
request_counts = defaultdict(list)

def rate_limit(max_requests=10, window=60):
    """Rate limiting decorator."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            now = time.time()
            
            # Clean old requests
            request_counts[client_ip] = [
                req_time for req_time in request_counts[client_ip]
                if now - req_time < window
            ]
            
            # Check rate limit
            if len(request_counts[client_ip]) >= max_requests:
                return jsonify({"error": "Rate limit exceeded"}), 429
            
            # Record request
            request_counts[client_ip].append(now)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_api_key(f):
    """API key authentication decorator."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != os.environ.get('API_KEY'):
            return jsonify({"error": "Invalid API key"}), 401
        return f(*args, **kwargs)
    return decorated_function

# Apply to endpoints
@app.route('/detect', methods=['POST'])
@rate_limit(max_requests=5, window=60)
@require_api_key
def secure_detect_objects():
    """Secured detection endpoint."""
    return detect_objects()
```

### File Validation

```python
# validation.py
import magic
import os

def validate_file(file_path, max_size=100*1024*1024):
    """Validate uploaded file."""
    
    # Check file size
    if os.path.getsize(file_path) > max_size:
        raise ValueError(f"File too large (max {max_size} bytes)")
    
    # Check file type using python-magic
    file_type = magic.from_file(file_path, mime=True)
    
    allowed_types = [
        'video/mp4',
        'video/avi', 
        'video/quicktime',
        'video/x-msvideo'
    ]
    
    if file_type not in allowed_types:
        raise ValueError(f"Invalid file type: {file_type}")
    
    return True
```

## 🚀 Performance Optimization

### Model Optimization

```python
# optimization.py
import torch
from models.factory import ModelFactory

def optimize_model_for_production(model):
    """Optimize model for production inference."""
    
    # Set to evaluation mode
    if hasattr(model, 'model'):
        model.model.eval()
    
    # Disable gradient computation
    for param in model.model.parameters():
        param.requires_grad = False
    
    # Enable optimized inference
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    return model

def create_optimized_model(model_type="yolo", model_size="n"):
    """Create and optimize model for production."""
    model = ModelFactory.create_model(model_type, model_size=model_size)
    return optimize_model_for_production(model)
```

### Batch Processing

```python
# batch_processing.py
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

class BatchProcessor:
    """Batch processor for handling multiple requests efficiently."""
    
    def __init__(self, model, batch_size=4, max_workers=2):
        self.model = model
        self.batch_size = batch_size
        self.request_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_batches)
        self.processing_thread.start()
    
    def _process_batches(self):
        """Process requests in batches."""
        while self.running:
            batch = []
            
            # Collect batch
            try:
                for _ in range(self.batch_size):
                    request = self.request_queue.get(timeout=1.0)
                    batch.append(request)
            except queue.Empty:
                if batch:  # Process partial batch
                    self._process_batch(batch)
                continue
            
            # Process full batch
            self._process_batch(batch)
    
    def _process_batch(self, batch):
        """Process a batch of requests."""
        futures = []
        for request_data in batch:
            future = self.executor.submit(self._process_single, request_data)
            futures.append(future)
        
        # Wait for all to complete
        for future in futures:
            future.result()
    
    def _process_single(self, request_data):
        """Process a single request."""
        frame, callback = request_data
        detections = self.model.detect_objects(frame)
        callback(detections)
    
    def submit_request(self, frame, callback):
        """Submit a processing request."""
        self.request_queue.put((frame, callback))
    
    def shutdown(self):
        """Shutdown the processor."""
        self.running = False
        self.processing_thread.join()
        self.executor.shutdown()
```

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Environment variables configured
- [ ] Dependencies installed and tested
- [ ] Models downloaded and verified
- [ ] Storage paths configured
- [ ] Logging setup completed
- [ ] Security measures implemented

### Production Readiness
- [ ] Health checks implemented
- [ ] Monitoring and metrics setup
- [ ] Rate limiting configured
- [ ] Error handling comprehensive
- [ ] Resource limits defined
- [ ] Backup and recovery plan

### Post-Deployment
- [ ] Monitor application logs
- [ ] Check resource utilization
- [ ] Verify model performance
- [ ] Test all endpoints
- [ ] Monitor error rates
- [ ] Validate security measures

## 🔄 Maintenance and Updates

### Model Updates
```bash
# Download new model versions
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Test new models
python tests/test_simple.py tests/test_video.mp4

# Deploy with zero downtime
# (Implementation depends on your deployment strategy)
```

### Application Updates
```bash
# Pull latest code
git pull origin main

# Run tests
python -m pytest tests/

# Update dependencies
pip install -r requirements.txt

# Restart services
# (Implementation depends on your deployment method)
```

### Performance Monitoring
- Monitor inference times
- Track memory usage
- Check error rates
- Analyze request patterns
- Optimize based on metrics

## 🆘 Troubleshooting

### Common Production Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Use smaller models
   - Implement memory cleanup

2. **Slow Performance**
   - Check GPU utilization
   - Optimize model settings
   - Implement caching

3. **Model Loading Failures**
   - Verify model files
   - Check permissions
   - Validate environment

4. **Network Issues**
   - Check firewall settings
   - Verify port configuration
   - Test connectivity

For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
