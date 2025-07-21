import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
import os
import warnings
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import time
import argparse
from datetime import datetime
import subprocess
import json

# Import skops untuk menyimpan model yang kompatibel dengan Hugging Face Spaces
import skops.io as sio
from skops.io import get_untrusted_types

# Import DVC API untuk versioning
try:
    import dvc.api
    import dvc.repo
    DVC_AVAILABLE = True
except ImportError:
    DVC_AVAILABLE = False
    print("⚠️ DVC not available, versioning will be skipped")

warnings.filterwarnings("ignore")


class PersonalityClassifier:
    def __init__(self, data_path="Data/personality_datasert.csv", retrain=False, old_data_path=None):
        self.data_path = data_path
        self.retrain = retrain
        self.old_data_path = old_data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def setup_git_config(self):
        """Setup git config untuk DVC commits"""
        try:
            # Check if git user is configured
            result = subprocess.run(['git', 'config', '--global', 'user.email'], 
                                  capture_output=True, text=True)
            if not result.stdout.strip():
                subprocess.run(['git', 'config', '--global', 'user.email', 'mlops@automated.local'])
                subprocess.run(['git', 'config', '--global', 'user.name', 'MLOps Automation'])
                print("✅ Git config setup for DVC versioning")
        except Exception as e:
            print(f"⚠️ Could not setup git config: {e}")

    def get_dvc_data_versions(self):
        """Get current DVC data versions"""
        versions = {}
        try:
            if DVC_AVAILABLE:
                repo = dvc.repo.Repo()
                # Get data file hashes
                for dvc_file in ['Data/personality_datasert.csv.dvc', 'Data/synthetic_ctgan_data.csv.dvc']:
                    if os.path.exists(dvc_file):
                        with open(dvc_file, 'r') as f:
                            import yaml
                            dvc_data = yaml.safe_load(f)
                            if 'outs' in dvc_data and len(dvc_data['outs']) > 0:
                                versions[dvc_file] = dvc_data['outs'][0].get('md5', 'unknown')
        except Exception as e:
            print(f"⚠️ Could not get DVC versions: {e}")
        return versions

    def dvc_add_data(self, data_path):
        """Add data file to DVC tracking"""
        try:
            if not DVC_AVAILABLE:
                print("⚠️ DVC not available, skipping data versioning")
                return False
                
            print(f"📦 Adding {data_path} to DVC...")
            result = subprocess.run(['dvc', 'add', data_path], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {data_path} added to DVC")
                return True
            else:
                print(f"⚠️ DVC add failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"⚠️ DVC add error: {e}")
            return False

    def dvc_add_model(self, model_dir="Model"):
        """Add model directory to DVC tracking"""
        try:
            if not DVC_AVAILABLE:
                print("⚠️ DVC not available, skipping model versioning")
                return False
                
            print(f"🤖 Adding {model_dir} to DVC...")
            result = subprocess.run(['dvc', 'add', model_dir], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {model_dir} added to DVC")
                return True
            else:
                print(f"⚠️ DVC add model failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"⚠️ DVC add model error: {e}")
            return False

    def dvc_commit_and_push(self, message="Automated DVC versioning"):
        """Commit DVC changes and push to remote"""
        try:
            if not DVC_AVAILABLE:
                return False
                
            # Git add DVC files
            subprocess.run(['git', 'add', '*.dvc', '.gitignore'], capture_output=True)
            
            # Git commit
            commit_message = f"{message} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            result = subprocess.run(['git', 'commit', '-m', commit_message], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Git commit successful: {commit_message}")
                
                # DVC push to remote
                push_result = subprocess.run(['dvc', 'push'], capture_output=True, text=True)
                if push_result.returncode == 0:
                    print("✅ DVC push to MinIO successful")
                    return True
                else:
                    print(f"⚠️ DVC push failed: {push_result.stderr}")
                    return False
            else:
                print(f"⚠️ Git commit failed (might be no changes): {result.stderr}")
                return False
                
        except Exception as e:
            print(f"⚠️ DVC commit/push error: {e}")
            return False

    def load_dvc_data(self, data_path):
        """Load data with DVC versioning info"""
        try:
            # Get current DVC version before loading
            dvc_versions = self.get_dvc_data_versions()
            
            # Load the actual data
            data = pd.read_csv(data_path)
            
            # Log DVC version info
            version_info = {
                'data_path': data_path,
                'data_shape': data.shape,
                'dvc_versions': dvc_versions,
                'load_timestamp': datetime.now().isoformat()
            }
            
            print(f"📊 Loaded data: {data.shape} with DVC version tracking")
            return data, version_info
            
        except Exception as e:
            print(f"⚠️ Error loading DVC data: {e}")
            # Fallback to regular loading
            data = pd.read_csv(data_path)
            return data, {'error': str(e)}


    def load_and_explore_data(self):
        print("📊 Memuat dataset dengan DVC versioning...")

        # Setup git config untuk DVC
        self.setup_git_config()

        # Load new data dengan DVC tracking
        new_data, version_info = self.load_dvc_data(self.data_path)
        print(f"Data baru: {new_data.shape}")

        if self.retrain and self.old_data_path and os.path.exists(self.old_data_path):
            print(f"Gabungkan dengan data lama dari: {self.old_data_path}")
            old_data, old_version_info = self.load_dvc_data(self.old_data_path)
            self.data = pd.concat([old_data, new_data], ignore_index=True)
            print(f"Total data setelah digabung: {self.data.shape}")
            
            # Store combined version info
            self.version_info = {
                'new_data': version_info,
                'old_data': old_version_info,
                'combined_shape': self.data.shape
            }
        else:
            self.data = new_data
            self.version_info = version_info

        print("\n📈 Distribusi target:")
        print(self.data["Personality"].value_counts())

        # DVC data versioning handled by pipeline


    def preprocess_data(self):
        print("\nPreprocessing data...")
        self.data = self.data.dropna()
        categorical_columns = ["Stage_fear", "Drained_after_socializing"]
        for col in categorical_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].map({"Yes": 1, "No": 0})
        X = self.data.drop("Personality", axis=1)
        y = self.data["Personality"]
        y_encoded = self.label_encoder.fit_transform(y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

    def visualize_data(self):
        print("\nMembuat visualisasi...")
        os.makedirs("Results", exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        self.data["Personality"].value_counts().plot(kind="bar", ax=axes[0, 0])
        axes[0, 0].set_title("Distribusi Personality")
        numeric_data = self.data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            correlation_matrix = numeric_data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=axes[0, 1])
            axes[0, 1].set_title("Heatmap Korelasi Fitur")
        features_to_plot = [
            "Time_spent_Alone",
            "Social_event_attendance",
            "Going_outside",
            "Friends_circle_size",
        ]
        for i, feature in enumerate(features_to_plot):
            if feature in self.data.columns:
                row, col = divmod(i + 2, 3)
                if row < 2:
                    sns.boxplot(
                        data=self.data, x="Personality", y=feature, ax=axes[row, col]
                    )
                    axes[row, col].set_title(f"{feature} berdasarkan Personality")
        plt.tight_layout()
        plt.savefig("Results/data_exploration.png", dpi=300, bbox_inches="tight")

    def train_best_model(self):
        print("\nMelatih model Random Forest dengan hyperparameter tuning...")
        best_model = RandomForestClassifier(random_state=42)
        best_params = {
            "model__n_estimators": [100],
            "model__max_depth": [None],
            "model__min_samples_split": [10],
            "model__min_samples_leaf": [1],
        }
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", best_model)])
        grid_search = GridSearchCV(
            pipeline, best_params, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)
        self.best_model = grid_search.best_estimator_
        print(f"Best CV Score: {grid_search.best_score_:.4f}")
        print(f"Best Params: {grid_search.best_params_}")
        model_results = {
            "Random Forest": {
                "best_score": grid_search.best_score_,
                "best_params": grid_search.best_params_,
                "model": self.best_model,
            }
        }
        return model_results

    def evaluate_model(self, model_results):
        print("\nEvaluasi model...")
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        accuracy = accuracy_score(self.y_test, y_pred)
        auc_score = roc_auc_score(self.y_test, y_pred_proba)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        target_names = self.label_encoder.classes_
        print(classification_report(self.y_test, y_pred, target_names=target_names))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[0],
            xticklabels=target_names,
            yticklabels=target_names,
        )
        axes[0].set_title("Confusion Matrix")
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        axes[1].plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        axes[1].plot([0, 1], [0, 1], linestyle="--")
        axes[1].set_title("ROC Curve")
        importances = self.best_model.named_steps["model"].feature_importances_
        feature_names = self.X_train.columns
        indices = np.argsort(importances)[::-1]
        axes[2].bar(range(len(importances)), importances[indices])
        axes[2].set_title("Feature Importance")
        axes[2].set_xticks(range(len(importances)))
        axes[2].set_xticklabels([feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig("Results/model_evaluation.png", dpi=300, bbox_inches="tight")
        with open("Results/metrics.txt", "w") as outfile:
            outfile.write(f"Accuracy = {round(accuracy, 4)}, AUC = {round(auc_score, 4)}")
            outfile.write(
                f"\nCV Score = {round(model_results['Random Forest']['best_score'], 4)}"
            )
        return accuracy, auc_score

    def save_model(self):
        print("\n🤖 Menyimpan model dengan DVC versioning...")
        os.makedirs("Model", exist_ok=True)
        
        # Save model and artifacts
        sio.dump(self.best_model, "Model/personality_classifier.skops")
        sio.dump(self.label_encoder, "Model/label_encoder.skops")
        sio.dump(list(self.X_train.columns), "Model/feature_names.skops")
        
        # Save version info
        version_info = {
            'model_timestamp': datetime.now().isoformat(),
            'model_type': 'RandomForestClassifier',
            'data_version_info': getattr(self, 'version_info', {}),
            'training_params': getattr(self, 'best_params', {}),
            'model_metrics': {
                'accuracy': getattr(self, 'final_accuracy', 0.0),
                'auc': getattr(self, 'final_auc', 0.0)
            }
        }
        
        with open("Model/model_version_info.json", 'w') as f:
            json.dump(version_info, f, indent=2)
        
        print("✅ Model & artifacts disimpan dengan version info")
        
        # Model versioning handled by DVC pipeline
        
        # Validate model
        unknown_types = get_untrusted_types(file="Model/personality_classifier.skops")
        try:
            sio.load("Model/personality_classifier.skops", trusted=unknown_types)
            print("✅ Model valid dan siap deploy!")
        except Exception as e:
            print(f"❌ Gagal load model: {e}")
        
        return True

    def run_complete_pipeline(self):
        print("=" * 60)
        print("MLFLOW TRACKING: RandomForest - Personality Classifier")
        print("=" * 60)

        mlflow.set_experiment("Personality Classification")

        with mlflow.start_run(run_name="RandomForest_PersonalityClassifier"):
            # Track training start time
            training_start_time = time.time()
            
            # Phase 1: Data Loading & Preprocessing dengan DVC
            print("📊 Phase 1: Data Loading & Preprocessing dengan DVC")
            self.load_and_explore_data()
            self.preprocess_data()
            self.visualize_data()

            # Phase 2: Model Training
            print("🤖 Phase 2: Model Training")
            model_results = self.train_best_model()
            accuracy, auc_score = self.evaluate_model(model_results)
            
            # Store metrics for model versioning
            self.final_accuracy = accuracy
            self.final_auc = auc_score
            self.best_params = model_results["Random Forest"]["best_params"]

            # Calculate training duration
            training_duration = time.time() - training_start_time

            # Log parameters to MLflow
            best_params = model_results["Random Forest"]["best_params"]
            for param, value in best_params.items():
                mlflow.log_param(param, value)

            # Log DVC version info to MLflow
            if hasattr(self, 'version_info'):
                mlflow.log_param("dvc_version_info", str(self.version_info))
                if DVC_AVAILABLE:
                    mlflow.log_param("dvc_enabled", True)
                    mlflow.log_param("dvc_remote", "MinIO S3")
                else:
                    mlflow.log_param("dvc_enabled", False)

            # Log training metadata
            mlflow.log_param("training_duration", training_duration)
            mlflow.log_param("training_date", datetime.now().isoformat())
            mlflow.log_param("data_size", len(self.data))
            mlflow.log_param("feature_count", len(self.X_train.columns))
            mlflow.log_param("retrain_mode", self.retrain)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("auc_score", auc_score)
            mlflow.log_metric("training_duration_seconds", training_duration)

            # ✅ Tambahkan input example & signature
            input_example = self.X_test.iloc[:5]
            signature = infer_signature(self.X_test, self.best_model.predict(self.X_test))

            mlflow.sklearn.log_model(
                sk_model=self.best_model,
                artifact_path="random_forest_model",
                registered_model_name="RandomForestPersonality",
                input_example=input_example,
                signature=signature
            )

            # Log artifacts to MLflow
            mlflow.log_artifact("Results/data_exploration.png")
            mlflow.log_artifact("Results/model_evaluation.png")
            mlflow.log_artifact("Results/metrics.txt")

            # Phase 3: Model & Data Versioning dengan DVC
            print("📦 Phase 3: Model & Data Versioning dengan DVC")
            model_versioned = self.save_model()
            
            # Use DVC pipeline for proper versioning
            if DVC_AVAILABLE and model_versioned:
                try:
                    # Setup git config for commits
                    self.setup_git_config()
                    
                    # Git add all DVC tracked files
                    subprocess.run(['git', 'add', '*.dvc', '.gitignore', 'dvc.yaml', 'dvc.lock'], 
                                 capture_output=True, text=True)
                    
                    # Git commit changes
                    commit_message = f"MLOps Training - Acc:{accuracy:.4f} AUC:{auc_score:.4f}"
                    result = subprocess.run(['git', 'commit', '-m', commit_message], 
                                          capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print(f"✅ Git commit successful: {commit_message}")
                        
                        # Push to DVC remote (MinIO)
                        push_result = subprocess.run(['dvc', 'push'], capture_output=True, text=True)
                        if push_result.returncode == 0:
                            print("✅ DVC push to MinIO successful")
                            mlflow.log_param("dvc_versioning_success", True)
                            mlflow.log_param("dvc_remote_push", True)
                        else:
                            print(f"⚠️ DVC push failed: {push_result.stderr}")
                            mlflow.log_param("dvc_versioning_success", False)
                            mlflow.log_param("dvc_remote_push", False)
                    else:
                        print(f"⚠️ Git commit skipped (no changes): {result.stderr}")
                        mlflow.log_param("dvc_versioning_success", True)
                        mlflow.log_param("dvc_remote_push", True)
                    
                except Exception as e:
                    print(f"⚠️ DVC versioning error: {e}")
                    mlflow.log_param("dvc_versioning_success", False)
                    mlflow.log_param("dvc_error", str(e))
            else:
                print("⚠️ DVC tidak tersedia - Hanya MLflow versioning yang aktif")
                mlflow.log_param("dvc_versioning_success", False)

            # Log version info as artifact
            if os.path.exists("Model/model_version_info.json"):
                mlflow.log_artifact("Model/model_version_info.json")

            print("\n✅ Training pipeline selesai!")
            print("📊 MLflow Tracking: http://localhost:5000")
            print("🗄️ MinIO Storage: http://localhost:9001")
            print(f"⏱️ Training duration: {training_duration:.2f} seconds")

            return self.best_model, accuracy, auc_score
    
def parse_args():
    parser = argparse.ArgumentParser(description="Training atau retraining Personality Classifier")
    parser.add_argument('--data_path', type=str, default="Data/personality_datasert.csv", help="Path ke data baru")
    parser.add_argument('--old_data_path', type=str, default=None, help="Path ke data lama (jika retrain)")
    parser.add_argument('--retrain', action='store_true', help="Aktifkan retrain dengan data baru")
    return parser.parse_args()


def main():
    args = parse_args()
    classifier = PersonalityClassifier(
        data_path=args.data_path,
        retrain=args.retrain,
        old_data_path=args.old_data_path
    )
    model, acc, auc = classifier.run_complete_pipeline()
    return model, acc, auc



if __name__ == "__main__":
    model, acc, auc = main()
