# 🧠 Personality Classifier MLOps Pipeline

[![CI Status](https://github.com/firmnnm/Tugas-1-MLOps/workflows/CI%20-%20Continuous%20Integration/badge.svg?branch=Firman)](https://github.com/firmnnm/Tugas-1-MLOps/actions)
[![CD Status](https://github.com/firmnnm/Tugas-1-MLOps/workflows/CD%20-%20Continuous%20Deployment/badge.svg?branch=Firman)](https://github.com/firmnnm/Tugas-1-MLOps/actions)

> **Tugas 1 MLOps**: Complete MLOps pipeline for personality classification with automated CI/CD

## 🎯 Project Overview

This project implements a complete MLOps pipeline for personality classification using machine learning. The system includes automated training, testing, deployment, and monitoring with modern DevOps practices.

### ✨ Key Features

- 🤖 **Machine Learning**: Random Forest classifier for personality type prediction
- 🎨 **Modern UI**: Interactive Gradio web interface with real-time predictions
- 🔄 **CI/CD Pipeline**: Automated GitHub Actions workflows
- 🧪 **Testing Suite**: Comprehensive unit and integration tests
- 🔒 **Security**: Automated security scanning and validation
- 📊 **Monitoring**: Model performance tracking and visualization
- 🚀 **Deployment**: Automated deployment to Hugging Face Spaces

## 🏗️ Architecture

```
📦 MLOps Pipeline
├── 🔍 Data Processing & Feature Engineering
├── 🤖 Model Training (Random Forest)
├── 🧪 Automated Testing & Validation
├── 🔒 Security Scanning
├── 📊 Performance Monitoring
├── 🚀 Deployment to Hugging Face Spaces
└── 📈 Continuous Monitoring
```

## 🛠️ Technical Stack

### Machine Learning

- **Framework**: scikit-learn
- **Model**: Random Forest Classifier
- **Model Storage**: skops (secure format)
- **Features**: 8 personality dimensions
- **Target**: Personality type classification

### Development & Deployment

- **Web Interface**: Gradio 4.44.0
- **CI/CD**: GitHub Actions
- **Testing**: pytest, unittest
- **Code Quality**: Black, bandit, safety
- **Deployment**: Hugging Face Spaces

### Data Pipeline

- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Model Persistence**: skops, joblib

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/firmnnm/Tugas-1-MLOps.git
cd Tugas-1-MLOps

# Install dependencies
make install
```

### 2. Train Model

```bash
# Train and evaluate model
make train
make eval
```

### 3. Test Application

```bash
# Run tests
make test-app
make test-model

# Format code
make format
```

### 4. Run Application

```bash
# Start Gradio app
make run
```

### 5. Deploy

```bash
# Deploy to Hugging Face Spaces (requires HF_TOKEN)
make deploy HF_TOKEN=your_token
```

## 📁 Project Structure

```
Tugas-1-MLOps/
├── 📊 Data/
│   ├── personality_datasert.csv     # Training dataset
│   └── README.md                    # Data documentation
├── 🤖 Model/
│   ├── best_personality_classifier.skops  # Trained model
│   ├── label_encoder.skops          # Label encoder
│   ├── feature_names.skops          # Feature names
│   └── README.md                    # Model documentation
├── 🎨 App/
│   └── app.py                       # Gradio web application
├── 📈 Results/
│   ├── data_exploration.png         # Data visualization
│   ├── model_evaluation.png         # Model metrics
│   └── README.md                    # Results documentation
├── 🔄 .github/workflows/
│   ├── ci.yml                       # CI pipeline
│   └── cd.yml                       # CD pipeline
├── 🧪 Tests/
│   ├── test_app.py                  # Application tests
│   └── test_ui.py                   # UI tests
├── 📚 Documentation/
│   ├── CICD_DOCUMENTATION.md        # CI/CD guide
│   ├── APP_DOCUMENTATION.md         # Application guide
│   ├── UI_IMPROVEMENTS.md           # UI documentation
│   └── ERROR_FIX_DOCUMENTATION.md   # Troubleshooting
├── train.py                         # Training pipeline
├── requirements.txt                 # Dependencies
├── Makefile                         # Automation commands
└── README.md                        # This file
```

## 🔄 CI/CD Pipeline

### Continuous Integration (CI)

- ✅ **Code Quality**: Formatting, linting, compilation
- 🧪 **Testing**: Unit tests, integration tests, UI tests
- 🔒 **Security**: Vulnerability scanning, static analysis
- 🎯 **Model Validation**: Performance validation, artifacts check

### Continuous Deployment (CD)

- 🚀 **Auto Deployment**: Deploy to Hugging Face Spaces
- 📊 **Health Checks**: Application testing, model validation
- 📢 **Notifications**: Deployment status, access URLs

## 🧪 Testing

### Test Coverage

- ✅ Model loading and prediction
- ✅ Application functionality
- ✅ UI components and interactions
- ✅ Error handling and edge cases

### Run Tests

```bash
# Run all tests
make test-app
make test-model

# Run specific tests
python test_app.py
python test_ui.py
```

## 📊 Model Performance

The Random Forest classifier achieves excellent performance on personality classification:

- **High Accuracy**: Robust prediction performance
- **Feature Importance**: Balanced feature utilization
- **Generalization**: Good performance on unseen data
- **Interpretability**: Clear decision boundaries

View detailed metrics in `Results/metrics.txt` after training.

## 🎨 Web Application

### Features

- 📝 **Interactive Form**: Easy personality questionnaire
- 📊 **Real-time Predictions**: Instant personality classification
- 📈 **Confidence Visualization**: Bar charts showing prediction confidence
- 🎯 **Example Data**: Pre-filled examples for testing
- 🎨 **Modern UI**: Clean, responsive design with custom CSS

### Access

- **Local**: `http://localhost:7860` (when running locally)
- **Production**: [Hugging Face Spaces](https://huggingface.co/spaces/your-space-name)

## 🔧 Configuration

### Environment Variables

```bash
HF_TOKEN=your_hugging_face_token  # For deployment
```

### GitHub Secrets

```
HF_TOKEN: Your Hugging Face token for deployment
```

## 📈 Monitoring & Maintenance

### Automated Monitoring

- 🔍 **CI Status**: Continuous integration health
- 🚀 **Deployment Status**: Production deployment health
- 🔒 **Security Alerts**: Vulnerability notifications
- 📊 **Model Performance**: Accuracy and prediction metrics

### Manual Checks

- [ ] Review security scan results
- [ ] Monitor model performance
- [ ] Update dependencies
- [ ] Check application logs

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 Documentation

- 📚 **[CI/CD Guide](CICD_DOCUMENTATION.md)**: Complete pipeline documentation
- 🎨 **[App Guide](APP_DOCUMENTATION.md)**: Application usage and features
- 🔧 **[UI Guide](UI_IMPROVEMENTS.md)**: UI components and styling
- 🐛 **[Troubleshooting](ERROR_FIX_DOCUMENTATION.md)**: Common issues and fixes

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- scikit-learn team for the ML framework
- Gradio team for the web interface framework
- Hugging Face for the deployment platform
- GitHub Actions for CI/CD automation

---

**🎓 Academic Project**: Tugas 1 MLOps - Implementation of complete MLOps pipeline for personality classification

_Built with ❤️ by [Firman](https://github.com/firmnnm)_
