#!/usr/bin/env python3
"""
Test script untuk validasi DVC + MLflow + MinIO integration
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def check_dependencies():
    """Check if all required tools are available"""
    print("🔍 Checking dependencies...")
    
    dependencies = ['python', 'dvc', 'git', 'docker']
    missing = []
    
    for dep in dependencies:
        try:
            result = subprocess.run([dep, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {dep}: Available")
            else:
                missing.append(dep)
                print(f"❌ {dep}: Not available")
        except FileNotFoundError:
            missing.append(dep)
            print(f"❌ {dep}: Not found")
    
    return len(missing) == 0

def test_dvc_setup():
    """Test DVC setup and MinIO connection"""
    print("\n📦 Testing DVC setup...")
    
    try:
        # Check DVC status
        result = subprocess.run(['dvc', 'status'], capture_output=True, text=True)
        print(f"DVC Status: {result.stdout}")
        
        # Check DVC remote
        result = subprocess.run(['dvc', 'remote', 'list'], capture_output=True, text=True)
        print(f"DVC Remotes: {result.stdout}")
        
        return True
        
    except Exception as e:
        print(f"❌ DVC setup error: {e}")
        return False

def test_minio_connection():
    """Test MinIO connection"""
    print("\n🗄️ Testing MinIO connection...")
    
    try:
        # Test MinIO with PowerShell on Windows
        import platform
        if platform.system() == "Windows":
            result = subprocess.run(['powershell', '-Command', 
                                   'Invoke-WebRequest -Uri http://localhost:9000/minio/health/live -UseBasicParsing'], 
                                  capture_output=True, text=True)
        else:
            result = subprocess.run(['curl', '-f', 'http://localhost:9000/minio/health/live'], 
                                  capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ MinIO is running")
            return True
        else:
            print("❌ MinIO is not accessible")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"⚠️ Cannot test MinIO connection: {e}")
        return False

def test_mlflow_connection():
    """Test MLflow connection"""
    print("\n📊 Testing MLflow connection...")
    
    try:
        # Test MLflow with PowerShell on Windows
        import platform
        if platform.system() == "Windows":
            result = subprocess.run(['powershell', '-Command', 
                                   'Invoke-WebRequest -Uri http://localhost:5000 -UseBasicParsing'], 
                                  capture_output=True, text=True)
        else:
            result = subprocess.run(['curl', '-f', 'http://localhost:5000/health'], 
                                  capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ MLflow is running")
            return True
        else:
            print("❌ MLflow is not accessible")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"⚠️ Cannot test MLflow connection: {e}")
        return False

def validate_project_structure():
    """Validate project has all necessary files for DVC+MLflow"""
    print("\n📁 Validating project structure...")
    
    required_files = [
        'dvc.yaml',
        'requirements.txt',
        'train.py',
        'docker-compose.yml',
        'Data/personality_datasert.csv',
        'Data/synthetic_ctgan_data.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def main():
    print("=" * 60)
    print("🚀 TESTING DVC + MLflow + MinIO INTEGRATION")
    print("=" * 60)
    
    # Test results
    tests = {
        'dependencies': check_dependencies(),
        'project_structure': validate_project_structure(),
        'dvc_setup': test_dvc_setup(),
        'minio_connection': test_minio_connection(),
        'mlflow_connection': test_mlflow_connection()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in tests.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! DVC + MLflow + MinIO setup is ready!")
        print("\n📚 Next steps:")
        print("   1. Run: docker-compose up -d")
        print("   2. Run: python train.py")
        print("   3. Check MLflow: http://localhost:5000")
        print("   4. Check MinIO: http://localhost:9001")
        return True
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
