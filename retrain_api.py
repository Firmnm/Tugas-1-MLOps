from fastapi import FastAPI, Request
import subprocess
import os
import shutil
import time
from datetime import datetime
from fastapi.responses import JSONResponse

app = FastAPI()

class ModelFallbackManager:
    def __init__(self):
        self.model_path = "Model/personality_classifier.skops"
        self.backup_path = "Model/personality_classifier_backup.skops"
        self.metrics_path = "Results/metrics.txt"
        self.fallback_threshold = float(os.getenv('FALLBACK_THRESHOLD', '0.02'))  # 2%
    
    def backup_current_model(self):
        """Backup current model before retrain"""
        try:
            if os.path.exists(self.model_path):
                shutil.copy2(self.model_path, self.backup_path)
                print(f"✅ Model backed up to {self.backup_path}")
                return True
            else:
                print("⚠️ No current model to backup")
                return False
        except Exception as e:
            print(f"❌ Error backing up model: {e}")
            return False
    
    def get_model_metrics(self):
        """Get current model metrics from results file"""
        try:
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, 'r') as f:
                    content = f.read()
                
                accuracy = 0.0
                auc = 0.0
                
                if "Accuracy" in content:
                    try:
                        accuracy_part = content.split("Accuracy = ")[1].split(",")[0]
                        accuracy = float(accuracy_part.strip())
                    except (ValueError, IndexError):
                        pass
                
                if "AUC" in content:
                    try:
                        auc_part = content.split("AUC = ")[1]
                        auc_part = auc_part.split('\n')[0].split(',')[0].strip()
                        auc = float(auc_part)
                    except (ValueError, IndexError):
                        pass
                
                return accuracy, auc
            return 0.0, 0.0
        except Exception as e:
            print(f"❌ Error reading metrics: {e}")
            return 0.0, 0.0
    
    def evaluate_performance(self, old_accuracy, old_auc, new_accuracy, new_auc):
        """Evaluate if new model performance is acceptable"""
        if old_accuracy == 0.0 and old_auc == 0.0:
            print("✅ No previous metrics - accepting new model")
            return True, "no_previous_model"
        
        accuracy_diff = new_accuracy - old_accuracy
        auc_diff = new_auc - old_auc
        
        accuracy_degraded = accuracy_diff < -self.fallback_threshold
        auc_degraded = auc_diff < -self.fallback_threshold
        
        print(f"📊 Performance comparison:")
        print(f"   Accuracy: {old_accuracy:.4f} → {new_accuracy:.4f} (diff: {accuracy_diff:+.4f})")
        print(f"   AUC: {old_auc:.4f} → {new_auc:.4f} (diff: {auc_diff:+.4f})")
        print(f"   Threshold: {self.fallback_threshold}")
        
        if accuracy_degraded or auc_degraded:
            reasons = []
            if accuracy_degraded:
                reasons.append(f"accuracy_drop_{abs(accuracy_diff):.4f}")
            if auc_degraded:
                reasons.append(f"auc_drop_{abs(auc_diff):.4f}")
            
            reason = "_".join(reasons)
            print(f"⚠️ Performance degradation detected: {reason}")
            return False, reason
        else:
            print("✅ Performance acceptable - keeping new model")
            return True, "performance_improved"
    
    def trigger_fallback(self, reason):
        """Restore backup model"""
        try:
            if os.path.exists(self.backup_path):
                shutil.copy2(self.backup_path, self.model_path)
                print(f"🔄 MODEL FALLBACK TRIGGERED - Reason: {reason}")
                print(f"✅ Restored to previous model")
                return True
            else:
                print("❌ No backup model available for fallback")
                return False
        except Exception as e:
            print(f"❌ Error during model fallback: {e}")
            return False

fallback_manager = ModelFallbackManager()

@app.post("/retrain")
async def trigger_retrain_with_fallback():
    """Trigger retrain with automatic fallback protection"""
    try:
        print(f"🔄 Retrain triggered at {datetime.now()}")
        
        # 1. Backup current model
        backup_success = fallback_manager.backup_current_model()
        if not backup_success:
            print("⚠️ Proceeding without backup")
        
        # 2. Get current model performance before retrain
        old_accuracy, old_auc = fallback_manager.get_model_metrics()
        print(f"📊 Current model - Accuracy: {old_accuracy:.4f}, AUC: {old_auc:.4f}")
        
        # 3. Run retrain
        print("🚀 Starting model retraining...")
        result = subprocess.run(
            ["python", "train.py", "--retrain", "--data_path", "Data/synthetic_ctgan_data.csv", "--old_data_path", "Data/personality_datasert.csv"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("❌ Retraining failed")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error", 
                    "message": "Retraining failed", 
                    "error": result.stderr,
                    "fallback_triggered": False
                }
            )
        
        print("✅ Retraining completed successfully")
        
        # 4. Get new model performance
        time.sleep(2)  # Wait for metrics to be written
        new_accuracy, new_auc = fallback_manager.get_model_metrics()
        print(f"📊 New model - Accuracy: {new_accuracy:.4f}, AUC: {new_auc:.4f}")
        
        # 5. Evaluate performance and decide fallback
        is_acceptable, reason = fallback_manager.evaluate_performance(
            old_accuracy, old_auc, new_accuracy, new_auc
        )
        
        if is_acceptable:
            return {
                "status": "success",
                "message": "Retraining completed successfully",
                "old_metrics": {"accuracy": old_accuracy, "auc": old_auc},
                "new_metrics": {"accuracy": new_accuracy, "auc": new_auc},
                "fallback_triggered": False,
                "reason": reason,
                "output": result.stdout
            }
        else:
            # 6. Trigger fallback
            fallback_success = fallback_manager.trigger_fallback(reason)
            
            return {
                "status": "fallback",
                "message": "Model fallback triggered due to performance degradation",
                "old_metrics": {"accuracy": old_accuracy, "auc": old_auc},
                "new_metrics": {"accuracy": new_accuracy, "auc": new_auc},
                "fallback_triggered": True,
                "fallback_success": fallback_success,
                "reason": reason,
                "output": result.stdout
            }
            
    except Exception as e:
        print(f"❌ Internal server error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "message": "Internal server error", 
                "details": str(e),
                "fallback_triggered": False
            }
        )

@app.post("/trigger-retrain")
async def trigger_retrain(request: Request):
    """Legacy endpoint - redirects to new fallback-protected retrain"""
    try:
        # Safely parse body
        try:
            body = await request.json()
        except Exception:
            body = {"warning": "No JSON body or malformed JSON received"}

        print("🔔 Alert diterima, mulai retrain dengan fallback protection:", body)
        
        # Call the new fallback-protected retrain
        return await trigger_retrain_with_fallback()

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Internal server error", "details": str(e)}
        )

@app.get("/model-status")
async def get_model_status():
    """Get current model status and metrics"""
    try:
        accuracy, auc = fallback_manager.get_model_metrics()
        
        # Check if current model is a fallback
        is_fallback = os.path.exists(fallback_manager.backup_path) and \
                     os.path.getmtime(fallback_manager.backup_path) > os.path.getmtime(fallback_manager.model_path)
        
        return {
            "status": "success",
            "current_metrics": {"accuracy": accuracy, "auc": auc},
            "is_fallback": is_fallback,
            "model_path": fallback_manager.model_path,
            "backup_available": os.path.exists(fallback_manager.backup_path),
            "fallback_threshold": fallback_manager.fallback_threshold
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/manual-fallback")
async def manual_fallback():
    """Manually trigger model fallback"""
    try:
        fallback_success = fallback_manager.trigger_fallback("manual_trigger")
        
        if fallback_success:
            return {"status": "success", "message": "Model fallback completed"}
        else:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Fallback failed - no backup available"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
