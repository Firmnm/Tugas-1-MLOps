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
	git push origin Firman

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
	python App/app.py

test-model:
	@echo "Testing model loading..."
	python -c "import skops.io as sio; from skops.io import get_untrusted_types; \
	unknown_types = get_untrusted_types(file='Model/personality_classifier.skops'); \
	model = sio.load('Model/personality_classifier.skops', trusted=unknown_types); \
	print('✅ Model loaded successfully!')"

clean:
	rm -rf Results/
	rm -rf Model/
	rm -f report.md
	@echo "🧹 Cleaned up generated files"

.PHONY: install format test train eval update-branch deploy run test-model clean
