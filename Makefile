.PHONY: help setup test clean format lint run-detr run-yolo

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Set up the conda environment
	@echo "🚀 Setting up environment..."
	./setup.sh

test: ## Run tests
	@echo "🧪 Running tests..."
	pytest tests/ -v

test-coverage: ## Run tests with coverage
	@echo "🧪 Running tests with coverage..."
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing

clean: ## Clean up temporary files and cache
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/
	rm -rf output/ frames/ temp/

format: ## Format code with black
	@echo "🎨 Formatting code..."
	black .
	
lint: ## Run linting
	@echo "🔍 Running linting..."
	flake8 .
	mypy .

run-detr: ## Run with DETR model (requires VIDEO_PATH)
	@if [ -z "$(VIDEO_PATH)" ]; then \
		echo "❌ Please provide VIDEO_PATH. Example: make run-detr VIDEO_PATH=tests/test_video.mp4"; \
		exit 1; \
	fi
	@echo "🎯 Running with DETR model..."
	python main.py $(VIDEO_PATH) --model detr

run-yolo: ## Run with YOLO model (requires VIDEO_PATH, optional MODEL_SIZE)
	@if [ -z "$(VIDEO_PATH)" ]; then \
		echo "❌ Please provide VIDEO_PATH. Example: make run-yolo VIDEO_PATH=tests/test_video.mp4"; \
		exit 1; \
	fi
	@if [ -z "$(MODEL_SIZE)" ]; then \
		echo "🎯 Running with YOLOv8n model..."; \
		python main.py $(VIDEO_PATH) --model yolo --model_size n; \
	else \
		echo "🎯 Running with YOLOv8$(MODEL_SIZE) model..."; \
		python main.py $(VIDEO_PATH) --model yolo --model_size $(MODEL_SIZE); \
	fi

run-yolo-small: ## Run with YOLOv8s model (requires VIDEO_PATH)
	@if [ -z "$(VIDEO_PATH)" ]; then \
		echo "❌ Please provide VIDEO_PATH. Example: make run-yolo-small VIDEO_PATH=tests/test_video.mp4"; \
		exit 1; \
	fi
	@echo "🎯 Running with YOLOv8s model..."
	python main.py $(VIDEO_PATH) --model yolo --model_size s

run-yolo-medium: ## Run with YOLOv8m model (requires VIDEO_PATH)
	@if [ -z "$(VIDEO_PATH)" ]; then \
		echo "❌ Please provide VIDEO_PATH. Example: make run-yolo-medium VIDEO_PATH=tests/test_video.mp4"; \
		exit 1; \
	fi
	@echo "🎯 Running with YOLOv8m model..."
	python main.py $(VIDEO_PATH) --model yolo --model_size m

demo: ## Run demo with test video
	@echo "🎬 Running demo..."
	python main.py tests/test_video.mp4 --model detr --display_video true

test-yolo: ## Test YOLOv8 integration
	@echo "🧪 Testing YOLOv8 integration..."
	python test_yolo_integration.py

install-dev: ## Install development dependencies
	@echo "🛠️ Installing development dependencies..."
	conda env update -f environment.yml

update-env: ## Update conda environment from environment.yml
	@echo "🔄 Updating conda environment..."
	conda env update -f environment.yml

export-env: ## Export current environment to environment.yml
	@echo "📦 Exporting environment..."
	conda env export --no-builds > environment.yml

check-gpu: ## Check if GPU/CUDA is available
	@echo "🔍 Checking GPU availability..."
	python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')"
