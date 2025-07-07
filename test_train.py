import os

def test_train_script_runs():
    exit_code = os.system("python train.py")
    assert exit_code == 0

def test_model_files_exist():
    assert os.path.exists("Model/personality_classifier.skops")
    assert os.path.exists("Model/label_encoder.skops")
    assert os.path.exists("Model/feature_names.skops")
