# 🧪 Local CI/CD Testing Guide

## Overview

Panduan lengkap untuk testing CI/CD pipeline di lokal sebelum push ke GitHub Actions.

## 🛠️ Prerequisites

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/firmnnm/Tugas-1-MLOps.git
cd Tugas-1-MLOps

# Install dependencies
pip install -r requirements.txt

# Install testing tools
pip install yamllint safety bandit
```

### 2. Model Training (Optional)

```bash
# Train model jika belum ada
python train.py
```

## 🔍 Testing Methods

### Method 1: Manual Component Testing

#### A. Test Individual CI Components

```bash
# 1. Code formatting check
black --check --diff *.py App/*.py

# 2. Python compilation
python -m py_compile train.py
python -m py_compile App/app.py
python -m py_compile test_app.py

# 3. Model loading test
python test_app.py

# 4. UI application test
python test_ui.py

# 5. Training pipeline test
python train.py

# 6. Model validation test
make test-model
```

#### B. Test CD Components

```bash
# 1. Test app locally (with timeout)
timeout 30s python App/app.py

# 2. Prepare deployment files
cp App/app.py app.py

# 3. Test HuggingFace Hub import
python -c "from huggingface_hub import HfApi; print('✅ Success')"
```

#### C. Security Testing

```bash
# 1. Install security tools
pip install safety bandit

# 2. Check dependency vulnerabilities
safety check --json

# 3. Static security analysis
bandit -r . -f json -o bandit-report.json
```

### Method 2: Automated Testing Script

#### Quick Testing

```bash
# Make script executable
chmod +x test_cicd_local.sh

# Test specific components
./test_cicd_local.sh ci        # Test CI components only
./test_cicd_local.sh cd        # Test CD components only
./test_cicd_local.sh security  # Test security scanning
./test_cicd_local.sh yaml      # Test YAML syntax

# Test everything
./test_cicd_local.sh full      # Default: test all components
```

### Method 3: Using act (GitHub Actions Runner)

#### Install act

```bash
# macOS
brew install act

# Linux
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Windows
choco install act-cli
```

#### Run GitHub Actions Locally

```bash
# Test CI workflow
act -W .github/workflows/ci.yml

# Test CD workflow (requires secrets)
act -W .github/workflows/cd.yml --secret-file .secrets

# List available workflows
act --list
```

#### Setup .secrets file for act

```bash
# Create .secrets file for local testing
cat > .secrets << EOF
HF_TOKEN=your_hugging_face_token_here
EOF
```

### Method 4: Docker Testing

#### Create Testing Container

```bash
# Build test container
docker build -t mlops-test .

# Run tests in container
docker run --rm -v $(pwd):/app mlops-test ./test_cicd_local.sh
```

## 📊 Testing Outputs

### Test Reports

After running tests, check these files:

- `test_report.md` - Overall test summary
- `safety_report.json` - Dependency security report
- `bandit_report.json` - Code security analysis
- `Results/metrics.txt` - Model performance metrics

### Expected Results

```
✅ Code formatting passed
✅ All Python files compile successfully
✅ Model loading tests passed
✅ UI application tests passed
✅ Model files exist
✅ Model validation passed
✅ All CI components passed!
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Not Found

```bash
# Solution: Train model first
python train.py
```

#### 2. Import Errors

```bash
# Solution: Install missing dependencies
pip install -r requirements.txt
```

#### 3. Port Already in Use

```bash
# Solution: Kill process using port
lsof -ti:7861 | xargs kill -9
```

#### 4. YAML Syntax Errors

```bash
# Check syntax
yamllint .github/workflows/ci.yml
yamllint .github/workflows/cd.yml

# Fix indentation (use 2 spaces consistently)
```

#### 5. Security Scan Failures

```bash
# Update dependencies to fix vulnerabilities
pip install --upgrade package_name

# Review bandit report for false positives
```

## 🎯 CI/CD Validation Checklist

### ✅ Pre-Push Checklist

- [ ] Code formatted with Black
- [ ] All Python files compile
- [ ] Model loads successfully
- [ ] UI tests pass
- [ ] Security scans clean
- [ ] YAML syntax valid
- [ ] Documentation updated

### ✅ CD Readiness Checklist

- [ ] Model files exist and valid
- [ ] App starts successfully
- [ ] HuggingFace Hub connection works
- [ ] Deployment files prepared
- [ ] HF_TOKEN secret configured

## 🚀 Advanced Testing

### Performance Testing

```bash
# Test model prediction speed
time python -c "
import sys; sys.path.append('App')
from app import PersonalityClassifierApp
app = PersonalityClassifierApp()
for i in range(100):
    app.predict_personality(4, 'No', 5, 6, 'Yes', 10, 5)
"
```

### Load Testing

```bash
# Test multiple concurrent requests
pip install locust

# Create locustfile.py for load testing
# Run: locust -f locustfile.py --host=http://localhost:7861
```

### Memory Usage Testing

```bash
# Monitor memory usage during tests
pip install memory-profiler

# Profile memory usage
mprof run python test_app.py
mprof plot
```

## 📈 Continuous Testing

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: local
    hooks:
      - id: ci-tests
        name: CI Tests
        entry: ./test_cicd_local.sh ci
        language: system
        always_run: true
EOF

# Install hooks
pre-commit install
```

### Git Hooks

```bash
# Add to .git/hooks/pre-push
#!/bin/bash
echo "Running CI/CD tests before push..."
./test_cicd_local.sh full
if [ $? -ne 0 ]; then
    echo "Tests failed! Push aborted."
    exit 1
fi
```

## 🔄 Integration with IDE

### VSCode Tasks

Add to `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Test CI Components",
      "type": "shell",
      "command": "./test_cicd_local.sh ci",
      "group": "test",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    }
  ]
}
```

### PyCharm External Tools

1. Go to File → Settings → Tools → External Tools
2. Add new tool:
   - Name: "Test CI/CD"
   - Program: `./test_cicd_local.sh`
   - Arguments: `full`
   - Working directory: `$ProjectFileDir$`

## 📚 Best Practices

### 1. Regular Testing

- Test locally before every commit
- Run full test suite before major changes
- Monitor security reports regularly

### 2. Environment Consistency

- Use same Python version as CI
- Match dependency versions
- Test in clean environment

### 3. Documentation

- Update tests when adding features
- Document test failures and solutions
- Keep testing guide current

### 4. Performance

- Monitor test execution time
- Optimize slow tests
- Use parallel testing when possible

---

## 🎯 Summary

**Local CI/CD testing memungkinkan:**

- ✅ Validasi changes sebelum push
- ✅ Debug issues lebih cepat
- ✅ Confidence tinggi untuk deployment
- ✅ Development cycle yang lebih efficient

**Recommended workflow:**

1. Develop locally
2. Run `./test_cicd_local.sh full`
3. Fix any issues
4. Push to branch `Firman`
5. Monitor GitHub Actions execution

_Happy testing! 🚀_
