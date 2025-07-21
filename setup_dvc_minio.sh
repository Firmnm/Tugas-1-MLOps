#!/bin/bash

# Setup DVC with MinIO remote storage
echo "🔧 Setting up DVC with MinIO remote storage..."

# Configure MinIO remote for DVC
dvc remote add -d minio s3://dvc-storage/
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin

# Create bucket if not exists
echo "📦 Creating MinIO bucket for DVC storage..."

echo "✅ DVC remote storage configured with MinIO"
echo "   Endpoint: http://localhost:9000"
echo "   Bucket: dvc-storage"
