# 🚀 DVC + MLflow + MinIO Automated Versioning Setup

## 📋 Overview

Proyek ini sekarang dilengkapi dengan **otomatisasi versioning lengkap** menggunakan:

- **DVC (Data Version Control)**: Versioning data dan model
- **MLflow**: Experiment tracking dan model registry
- **MinIO**: S3-compatible storage sebagai DVC remote dan MLflow artifact store

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Training      │    │     MLflow      │    │     MinIO       │
│   Pipeline      │◄──►│   Tracking      │◄──►│   Storage       │
│                 │    │                 │    │                 │
│ • Data Loading  │    │ • Experiments   │    │ • Data Files    │
│ • Preprocessing │    │ • Parameters    │    │ • Model Files   │
│ • Model Train   │    │ • Metrics       │    │ • Artifacts     │
│ • DVC Versioning│    │ • Artifacts     │    │ • Versions      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                    ┌─────────────────┐
                    │   Prometheus    │
                    │   + Grafana     │
                    │   Monitoring    │
                    └─────────────────┘
```

## 🚀 Quick Start

### 1. Start All Services
```bash
docker-compose up -d
```

Ini akan menjalankan:
- **MinIO**: http://localhost:9001 (admin/admin: minioadmin)
- **MLflow**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001 (admin/admin)
- **App**: http://localhost:7860

### 2. Run Training dengan Full Versioning
```bash
# Training normal
python train.py

# Retraining dengan data baru
python train.py --retrain --data_path Data/new_data.csv --old_data_path Data/personality_datasert.csv
```

### 3. Verify Setup
```bash
python test_dvc_mlflow_integration.py
```

## 📊 What Gets Versioned

### 🗂️ Data Versioning (DVC + MinIO)
- ✅ `Data/personality_datasert.csv` → MinIO bucket `dvc-storage`
- ✅ `Data/synthetic_ctgan_data.csv` → MinIO bucket `dvc-storage`
- ✅ All data transformations tracked
- ✅ Data lineage preserved

### 🤖 Model Versioning (DVC + MLflow + MinIO)
- ✅ `Model/personality_classifier.skops` → MinIO
- ✅ `Model/label_encoder.skops` → MinIO  
- ✅ `Model/feature_names.skops` → MinIO
- ✅ `Model/model_version_info.json` → MinIO
- ✅ Model metrics → MLflow
- ✅ Model artifacts → MinIO bucket `mlflow-artifacts`

### 📈 Experiment Tracking (MLflow)
- ✅ Training parameters
- ✅ Model metrics (accuracy, AUC)
- ✅ Training duration
- ✅ Data version info
- ✅ DVC commit hashes
- ✅ Model signatures

## 🔄 Automated Workflow

### Training Pipeline Flow:
1. **Data Loading** → Load data dengan DVC version tracking
2. **DVC Add** → Tambahkan data files ke DVC tracking
3. **Training** → Train model dengan parameter logging ke MLflow
4. **Model Save** → Simpan model dengan version info
5. **DVC Add Model** → Tambahkan model ke DVC tracking  
6. **Git Commit** → Commit DVC files dengan metadata
7. **DVC Push** → Push ke MinIO remote storage
8. **MLflow Log** → Log model dan artifacts ke MLflow + MinIO

### Container Integration:
```yaml
# docker-compose.yml
mlops-app:
  environment:
    - MLFLOW_TRACKING_URI=http://mlflow:5000
    - AWS_ACCESS_KEY_ID=minioadmin
    - AWS_SECRET_ACCESS_KEY=minioadmin
    - AWS_ENDPOINT_URL=http://minio:9000
  command: >
    sh -c "
    dvc remote add -d minio s3://dvc-storage/ -f &&
    dvc remote modify minio endpointurl http://minio:9000 &&
    python monitoring.py &
    python App/app.py"
```

## 🗄️ Storage Layout

### MinIO Buckets:
```
minio:9000/
├── dvc-storage/          # DVC data dan model versions
│   ├── files/
│   │   ├── md5/xx/yy/... # Data files dengan hash
│   │   └── ...
│   └── cache/
└── mlflow-artifacts/     # MLflow experiment artifacts
    ├── 0/
    │   ├── exp_id/
    │   │   ├── artifacts/
    │   │   └── metrics/
    └── ...
```

### Local Project Structure:
```
project/
├── Data/
│   ├── *.csv              # Data files
│   └── *.csv.dvc          # DVC tracking files
├── Model/
│   ├── *.skops            # Model files  
│   ├── *.json             # Version info
│   └── Model.dvc          # DVC model tracking
├── .dvc/
│   ├── config             # DVC remote config
│   └── cache/             # Local DVC cache
└── dvc.yaml               # DVC pipeline definition
```

## 🔧 Configuration

### DVC Remote (Automatic Setup):
```bash
dvc remote add -d minio s3://dvc-storage/
dvc remote modify minio endpointurl http://minio:9000  
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin
```

### MLflow Config (Environment):
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_ENDPOINT_URL=http://localhost:9000
```

## 🧪 Testing & Validation

### Run Full Integration Test:
```bash
python test_dvc_mlflow_integration.py
```

### Manual Verification:
```bash
# Check DVC status
dvc status

# Check DVC remote
dvc remote list

# Check MLflow experiments  
mlflow experiments list

# Check MinIO buckets
curl http://localhost:9000/minio/health/live
```

## 📱 Web Interfaces

1. **MLflow UI**: http://localhost:5000
   - View experiments
   - Compare models  
   - Download artifacts

2. **MinIO Console**: http://localhost:9001
   - Username: `minioadmin`
   - Password: `minioadmin`
   - Browse buckets and files

3. **Grafana**: http://localhost:3001
   - Username: `admin`  
   - Password: `admin`
   - Monitor system metrics

4. **App Interface**: http://localhost:7860
   - ML model interface
   - Real-time predictions

## 🔍 Troubleshooting

### Common Issues:

1. **DVC Push Failed**:
   ```bash
   # Check MinIO connection
   dvc remote list
   dvc doctor
   ```

2. **MLflow Artifacts Error**:
   ```bash
   # Verify MinIO buckets exist
   docker logs minio-init
   ```

3. **Git Commit Issues**:
   ```bash
   # Configure git in container
   git config --global user.email "you@example.com"
   git config --global user.name "Your Name"
   ```

### Debug Commands:
```bash
# Check container logs
docker-compose logs mlops-app
docker-compose logs minio
docker-compose logs mlflow

# Check DVC status
docker exec -it personality-app dvc status

# Check MinIO buckets
docker exec -it minio-storage mc ls myminio/
```

## 🎯 Success Indicators

✅ **Full Setup Working When**:
- All containers running without errors
- DVC remote configured to MinIO
- MLflow accessible at :5000
- Training creates versions in both DVC and MLflow
- Data and models stored in MinIO buckets
- Git commits track DVC file changes
- Web interfaces accessible

Your MLOps pipeline now has **complete automated versioning**! 🚀
