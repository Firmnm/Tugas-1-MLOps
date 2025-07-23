# MLOps Personality Classification Project 🧠

A comprehensive MLOps implementation for personality classification using machine learning with complete monitoring, versioning, and deployment pipeline.

## 🎯 Project Overview

This project implements a full MLOps pipeline for personality classification based on behavioral features. The system includes:

- **Machine Learning Model**: Random Forest classifier for personality prediction
- **Data Management**: Data versioning with DVC and synthetic data generation
- **Experiment Tracking**: MLflow for model versioning and metrics tracking  
- **Monitoring**: Real-time model performance and data drift monitoring
- **Deployment**: Containerized application with Gradio interface
- **CI/CD**: Automated retraining pipeline

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  Data Pipeline  │───▶│   ML Pipeline   │
│                 │    │   (DVC + ETL)   │    │ (Train/Retrain)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │◀───│   Deployment    │◀───│  Model Registry │
│ (Drift + Perf)  │    │ (Gradio + API)  │    │    (MLflow)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## � Dataset

**Personality Dataset**: Behavioral features for personality type classification
- **Features**: 12 behavioral indicators (social habits, preferences, etc.)
- **Target**: Personality types (Extrovert/Introvert)
- **Size**: ~1000 samples
- **Format**: CSV with numerical and categorical features

### Features Description:
- `Time_spent_Alone`: Hours spent alone per day
- `Social_event_attendance`: Frequency of social events
- `Going_outside`: Outdoor activity frequency
- `Friends_circle_size`: Number of close friends
- `Stage_fear`: Public speaking anxiety (Yes/No)
- `Drained_after_socializing`: Post-social energy level (Yes/No)
- And more...

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Git

### 1. Clone Repository
```bash
git clone https://github.com/Firmnm/Tugas-1-MLOps.git
cd Tugas-1-MLOps
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Data Setup (DVC)
```bash
dvc pull  # Download dataset from remote storage
```

### 4. Run Complete Pipeline
```bash
# Option 1: Docker Compose (Recommended)
docker-compose up --build

# Option 2: Manual Setup
python train.py          # Train model
python monitoring.py &   # Start monitoring
python App/app.py        # Launch Gradio app
```

## 📁 Project Structure

```
├── 📊 Data/
│   ├── personality_datasert.csv.dvc    # Original dataset (DVC tracked)
│   └── synthetic_ctgan_data.csv.dvc    # Synthetic data (DVC tracked)
├── 🤖 Model/
│   ├── personality_classifier.skops    # Trained model
│   ├── label_encoder.skops             # Label encoder
│   └── feature_names.skops             # Feature names
├── 📱 App/
│   └── app.py                          # Gradio web interface
├── 📈 Results/
│   ├── metrics.txt                     # Model performance metrics
│   ├── data_exploration.png            # EDA visualizations
│   └── model_evaluation.png            # Model evaluation plots
├── 🔍 Monitoring/
│   ├── monitoring.py                   # Performance monitoring
│   ├── data_drift.py                   # Data drift detection
│   └── prometheus.yml                  # Prometheus configuration
├── 🐳 Docker/
│   ├── Dockerfile                      # Main application container
│   ├── Dockerfile.retrain              # Retraining container
│   └── docker-compose.yml              # Multi-service orchestration
├── 📋 Pipeline/
│   ├── train.py                        # Model training script
│   ├── predict.py                      # Inference script
│   ├── retrain_api.py                  # Automated retraining API
│   └── data_synthetic_generator.py     # Synthetic data generation
├── 📓 Analysis/
│   └── notebook.ipynb                  # Jupyter notebook with EDA & modeling
├── ⚙️ Configuration/
│   ├── dvc.yaml                        # DVC pipeline definition
│   ├── dvc.lock                        # DVC lock file
│   ├── requirements.txt                # Python dependencies
│   └── adaptive_drift_config.json      # Drift detection config
└── 📚 Documentation/
    ├── README.md                       
    └── report.md                       # Technical report
```

## 🔬 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 94.2% |
| **Precision** | 93.8% |
| **Recall** | 94.6% |
| **F1-Score** | 94.2% |
| **AUC-ROC** | 0.97 |

### Model Details:
- **Algorithm**: Random Forest Classifier
- **Features**: 12 behavioral indicators
- **Cross-validation**: 5-fold CV
- **Hyperparameter Tuning**: GridSearchCV
- **Feature Engineering**: StandardScaler normalization

## 🛠️ MLOps Components

### 1. **Data Version Control (DVC)**
```bash
dvc add Data/personality_datasert.csv
dvc push  # Upload to remote storage
dvc pull  # Download latest version
```

### 2. **Experiment Tracking (MLflow)**
- Model versioning and registry
- Metrics and parameter logging
- Model comparison and selection
- Artifact management

### 3. **Data Drift Monitoring**
- Statistical drift detection
- Feature distribution monitoring
- Automated alerts and retraining triggers
- Performance degradation tracking

### 4. **Automated Retraining**
```bash
# Trigger retraining when drift detected
curl -X POST http://localhost:8000/retrain
```

### 5. **Model Deployment**
- Gradio web interface for user interaction
- REST API for programmatic access
- Docker containerization for scalability
- Health checks and monitoring endpoints

## 🔍 Monitoring & Observability

### Data Drift Detection
- **Method**: Statistical tests (KS test, Population Stability Index)
- **Threshold**: Configurable drift sensitivity
- **Action**: Automatic retraining trigger

### Performance Monitoring
- **Metrics**: Accuracy, prediction latency, throughput
- **Visualization**: Grafana dashboards
- **Alerting**: Prometheus-based alerts

### Model Health Checks
```bash
curl http://localhost:8000/health     # Service health
curl http://localhost:8000/metrics    # Performance metrics
curl http://localhost:8000/drift      # Drift status
```

## 🚀 Usage Examples

### Web Interface (Gradio)
1. Open http://localhost:7860
2. Input behavioral features using sliders/dropdowns
3. Click "Predict Personality" 
4. View prediction with confidence score

### Python API
```python
import requests

# Prediction endpoint
data = {
    "Time_spent_Alone": 5.0,
    "Social_event_attendance": 3.0,
    "Going_outside": 4.0,
    # ... other features
}

response = requests.post("http://localhost:8000/predict", json=data)
prediction = response.json()
print(f"Personality: {prediction['personality']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

### Jupyter Notebook
```python
# Load the trained model
import skops.io as sio
model = sio.load("Model/personality_classifier.skops")

# Make predictions
import pandas as pd
features = pd.DataFrame([{
    "Time_spent_Alone": 5.0,
    "Social_event_attendance": 3.0,
    # ... other features
}])

prediction = model.predict(features)
probability = model.predict_proba(features)
```

## 🧪 Development Workflow

### 1. **Data Preparation**
```bash
python data_synthetic_generator.py  # Generate synthetic data
dvc add Data/new_data.csv           # Version new data
dvc push                            # Upload to remote
```

### 2. **Model Development**
```bash
# Experiment in Jupyter
jupyter notebook notebook.ipynb

# Train new model version
python train.py

# Compare models in MLflow UI
mlflow ui
```

### 3. **Testing & Validation**
```bash
python predict.py --test          # Test inference
python monitoring.py --validate   # Check monitoring
```

### 4. **Deployment**
```bash
docker-compose build              # Build containers
docker-compose up -d              # Deploy services
```

## 📋 Configuration

### Environment Variables
```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=personality_classification

# Monitoring Configuration  
DRIFT_THRESHOLD=0.1
MONITORING_INTERVAL=3600  # seconds

# Application Configuration
GRADIO_PORT=7860
API_PORT=8000
```

### DVC Configuration
```yaml
# .dvc/config
[core]
    remote = myremote
['remote "myremote"']
    url = s3://your-bucket/dvc-storage
```

