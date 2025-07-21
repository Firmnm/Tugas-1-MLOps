install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

format:
	black *.py App/*.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md

	echo '\n## Data Exploration Plot' >> report.md
	echo '![Data Exploration](./Results/data_exploration.png)' >> report.md

	echo '\n## Model Evaluation Plot' >> report.md
	echo '![Model Evaluation](./Results/model_evaluation.png)' >> report.md

	@echo "Report generated: report.md"

update-branch:
	git config user.name "$(USER_NAME)"
	git config user.email "$(USER_EMAIL)"
	git add Results/ Model/ report.md
	git commit -m "Update: training and evaluation results" || echo "Nothing to commit"
	git push --force origin HEAD:update || echo "Nothing to push"

deploy:
	@echo "Deploying to Hugging Face Spaces..."
	@echo "Checking HF token..."
	@if [ -z "$(HF_TOKEN)" ]; then \
		echo "❌ Error: HF token is empty!"; \
		echo "Please check that HF_TOKEN secret is set in GitHub repository"; \
		exit 1; \
	else \
		echo "✅ HF token is present"; \
	fi
	pip install huggingface_hub[cli]
	huggingface-cli login --token "$(HF_TOKEN)"
	@echo "🚀 Uploading main app file..."
	huggingface-cli upload firmnnm/Tugas1MLOps ./App/app.py app.py --repo-type=space --commit-message="Deploy personality classifier app"
	@echo "📁 Uploading model files..."
	huggingface-cli upload firmnnm/Tugas1MLOps ./Model/personality_classifier.skops Model/personality_classifier.skops --repo-type=space --commit-message="Upload personality classifier model"
	huggingface-cli upload firmnnm/Tugas1MLOps ./Model/label_encoder.skops Model/label_encoder.skops --repo-type=space --commit-message="Upload label encoder"
	huggingface-cli upload firmnnm/Tugas1MLOps ./Model/feature_names.skops Model/feature_names.skops --repo-type=space --commit-message="Upload feature names"
	@echo "📋 Uploading requirements..."
	huggingface-cli upload firmnnm/Tugas1MLOps ./requirements.txt requirements.txt --repo-type=space --commit-message="Upload requirements"
	@echo "📄 Uploading README..."
	huggingface-cli upload firmnnm/Tugas1MLOps ./README.md README.md --repo-type=space --commit-message="Upload README for Spaces"
	@echo "✅ Deployment to Hugging Face Spaces completed!"

run:
	@echo "🚀 Starting Personality Classifier app..."
	python App/app.py

# ========================
# Docker Commands
# ========================

docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t personality-classifier:latest .

docker-run:
	@echo "🚀 Running Docker container..."
	docker run -p 7860:7860 --name personality-app personality-classifier:latest

docker-run-detached:
	@echo "🚀 Running Docker container in background..."
	docker run -d -p 7860:7860 --name personality-app personality-classifier:latest

docker-stop:
	@echo "⏹️ Stopping Docker container..."
	docker stop personality-app || echo "Container not running"
	docker rm personality-app || echo "Container not found"

docker-compose-up:
	@echo "🐳 Starting services with Docker Compose..."
	docker-compose up -d

docker-compose-down:
	@echo "⏹️ Stopping Docker Compose services..."
	docker-compose down

docker-train:
	@echo "🎯 Running training in Docker..."
	docker-compose --profile training up mlops-training

docker-clean:
	@echo "🧹 Cleaning Docker resources..."
	docker system prune -f

# ========================
# MLOps Commands
# ========================

drift-detect:
	@echo "🔍 Running data drift detection..."
	python data_drift.py

monitoring:
	@echo "📊 Starting MLOps monitoring..."
	python monitoring.py

mlflow-server:
	@echo "🚀 Starting MLflow tracking server..."
	mlflow server --host 0.0.0.0 --port 5000

full-pipeline:
	@echo "🚀 Running complete MLOps pipeline..."
	python train.py
	python data_drift.py
	python monitoring.py

test-mlops:
	@echo "🧪 Running MLOps test suite..."
	python test_mlops.py

# ========================
# Monitoring Infrastructure
# ========================

start-monitoring-stack:
	@echo "🚀 Starting complete monitoring stack..."
	docker-compose up -d mlflow prometheus grafana

stop-monitoring-stack:
	@echo "⏹️ Stopping monitoring stack..."
	docker-compose down

monitoring-logs:
	@echo "📋 Showing monitoring logs..."
	docker-compose logs -f mlflow prometheus grafana

help:
	@echo "Available commands:"
	@echo "  install              - Install dependencies"
	@echo "  train                - Train the model"
	@echo "  run                  - Run app locally"
	@echo "  drift-detect         - Run data drift detection"
	@echo "  monitoring           - Start MLOps monitoring"
	@echo "  mlflow-server        - Start MLflow server"
	@echo "  full-pipeline        - Run complete MLOps pipeline"
	@echo "  test-mlops           - Run MLOps test suite"
	@echo "  start-monitoring-stack - Start monitoring infrastructure"
	@echo "  stop-monitoring-stack  - Stop monitoring infrastructure"
	@echo "  docker-build         - Build Docker image"
	@echo "  docker-run       - Run container"
	@echo "  docker-compose-up - Start all services"

.PHONY: install format train eval update-branch deploy run docker-build docker-run docker-compose-up help
