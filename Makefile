setup-environment:
	@echo "Creating conda environment"
	conda create -n video_analyzer python=3.11
	@echo "Conda environment created"
	conda init && conda activate video_analyzer
	@echo "Conda environment activated"
	@pip install -r requirements.txt
	@echo "Dependencies installed"