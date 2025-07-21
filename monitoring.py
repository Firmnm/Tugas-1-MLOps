import time
import psutil
import os
import json
import pandas as pd
from datetime import datetime
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
import threading
import mlflow
import mlflow.sklearn
from data_drift import DataDriftDetector  # Re-enabled
from data_synthetic_generator import adaptiveDriftGenerator


class MLOpsMonitoring:
    def __init__(self, port=8000):
        self.port = port
        
        # Initialize all metrics with error handling
        try:
            # System metrics
            self.cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
            self.memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')
            self.disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage')
            
            # ML metrics
            self.model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
            self.model_auc = Gauge('model_auc', 'Current model AUC score')
            self.training_duration = Histogram('model_training_duration_seconds', 'Model training duration')
            
            # Model fallback metrics (NEW)
            self.previous_model_accuracy = Gauge('previous_model_accuracy', 'Previous model accuracy for comparison')
            self.previous_model_auc = Gauge('previous_model_auc', 'Previous model AUC for comparison')
            self.model_fallback_triggered = Counter('model_fallback_triggered_total', 'Number of times model fallback was triggered')
            self.model_is_fallback = Gauge('model_is_fallback', 'Whether current model is a fallback model (1=fallback, 0=current)')
            self.retrain_attempts = Counter('model_retrain_attempts_total', 'Total number of retrain attempts')
            self.successful_retrains = Counter('model_successful_retrains_total', 'Total number of successful retrains')
            
            # Data drift metrics
            self.drift_features_count = Gauge('data_drift_features_count', 'Number of features with drift')
            self.drift_share = Gauge('data_drift_share_percent', 'Percentage of features with drift')
            self.dataset_drift = Gauge('dataset_drift_detected', 'Dataset drift score for alerting (resets after retrain, increases with synthetic data)')
            
            # Synthetic Data Generator metrics
            self.synthetic_drift_cycle = Gauge('synthetic_drift_cycle', 'Current drift generation cycle')
            self.synthetic_drift_strength = Gauge('synthetic_drift_strength', 'Current adaptive drift strength')
            self.synthetic_failed_detections = Gauge('synthetic_failed_detections', 'Number of failed drift detections')
            self.synthetic_data_size = Gauge('synthetic_data_size_total', 'Total size of synthetic dataset')
            self.synthetic_new_samples = Gauge('synthetic_new_samples_last', 'New samples generated in last cycle')
            self.synthetic_introvert_ratio = Gauge('synthetic_introvert_ratio', 'Ratio of introverts in synthetic data')
            self.synthetic_extrovert_ratio = Gauge('synthetic_extrovert_ratio', 'Ratio of extroverts in synthetic data')
            self.synthetic_generation_counter = Counter('synthetic_generation_total', 'Total number of synthetic data generations')
            
            # Request metrics
            self.prediction_counter = Counter('predictions_total', 'Total number of predictions made')
            self.prediction_latency = Histogram('prediction_duration_seconds', 'Prediction latency')
            
            # Model info
            self.model_info = Info('model_info', 'Information about the current model')
            
            # Synthetic data info
            self.synthetic_info = Info('synthetic_data_info', 'Information about synthetic data generation')
            
            # Monitoring status
            self.monitoring_active = False
            
            # Model versioning and fallback management (NEW)
            self.current_model_metrics = {'accuracy': 0.0, 'auc': 0.0}
            self.previous_model_metrics = {'accuracy': 0.0, 'auc': 0.0}
            self.model_backup_path = "Model/personality_classifier_backup.skops"
            self.model_metrics_history = []  # Store metrics history
            
            # Set initial values to ensure metrics are visible
            self._set_initial_values()
            
            # Initialize data generator with error handling
            try:
                self.data_generator = adaptiveDriftGenerator()
                print("Data generator initialized successfully")
            except Exception as e:
                print(f"Could not initialize data generator: {e}")
                self.data_generator = None
                
        except Exception as e:
            print(f"Error initializing monitoring metrics: {e}")
            raise
    
    def _set_initial_values(self):
        """Set initial values for all metrics to ensure they appear in Prometheus"""
        try:
            # Set system metrics to 0 initially
            self.cpu_usage.set(0)
            self.memory_usage.set(0)
            self.disk_usage.set(0)
            
            # Set model metrics to 0 initially - will be updated by _load_model_metrics()
            self.model_accuracy.set(0.0)  # Will be updated with real data
            self.model_auc.set(0.0)       # Will be updated with real data
            
            # Set fallback metrics (NEW)
            self.previous_model_accuracy.set(0.0)
            self.previous_model_auc.set(0.0)
            self.model_is_fallback.set(0)  # 0 = current model, 1 = fallback model
            
            # Set drift metrics
            self.drift_features_count.set(0)
            self.drift_share.set(0.0)
            self.dataset_drift.set(0.0)  # Initialize dataset drift score
            
            # Set synthetic data metrics
            self.synthetic_drift_cycle.set(0)
            self.synthetic_drift_strength.set(0.1)
            self.synthetic_failed_detections.set(0)
            self.synthetic_data_size.set(0)
            self.synthetic_new_samples.set(0)
            self.synthetic_introvert_ratio.set(0.5)
            self.synthetic_extrovert_ratio.set(0.5)
            
            # Set info metrics
            self.model_info.info({
                'model_name': 'RandomForestPersonality',
                'model_version': 'latest',
                'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'is_fallback': 'false',
                'fallback_reason': 'none'
            })
            
            self.synthetic_info.info({
                'cycle': '0',
                'drift_strength': '0.100',
                'failed_detections': '0',
                'last_generation': 'Never',
                'data_size': '0'
            })
            
            print("Initial metric values set")
            
        except Exception as e:
            print(f"Error setting initial values: {e}")
        
    def start_monitoring(self):
        """Start Prometheus metrics server"""
        print(f"Starting Prometheus metrics server on port {self.port}...")
        start_http_server(self.port)
        self.monitoring_active = True
        
        # Start background monitoring
        self._start_background_monitoring()
        print(f"Prometheus metrics available at http://localhost:{self.port}/metrics")
        
    def _start_background_monitoring(self):
        """Start background thread for system monitoring"""
        def monitor_system():
            while self.monitoring_active:
                self._update_system_metrics()
                time.sleep(10)  # Update every 10 seconds
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
        
    def _update_system_metrics(self):
        """Update system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.percent)
            
            # Disk usage (Windows compatible)
            import platform
            if platform.system() == "Windows":
                disk = psutil.disk_usage('C:\\')
            else:
                disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.disk_usage.set(disk_percent)
            
        except Exception as e:
            print(f"Error updating system metrics: {e}")
            # Set default values if system metrics fail
            self.cpu_usage.set(0)
            self.memory_usage.set(0)
            self.disk_usage.set(0)
    
    def update_model_metrics(self, accuracy=None, auc=None, training_time=None):
        """Update model performance metrics"""
        if accuracy is not None:
            self.model_accuracy.set(accuracy)
        if auc is not None:
            self.model_auc.set(auc)
        if training_time is not None:
            self.training_duration.observe(training_time)
    
    def update_drift_metrics(self, drift_metrics):
        """Update data drift metrics"""
        self.drift_features_count.set(drift_metrics.get('n_drifted_features', 0))
        self.drift_share.set(drift_metrics.get('share_drifted_features', 0.0) * 100)
        # Keep dataset_drift as simple boolean for now - will be updated by calculate_dataset_drift_score
        basic_drift = 1 if drift_metrics.get('dataset_drift', False) else 0
        if basic_drift == 0:  # Only set to 0 if no drift detected, otherwise preserve score
            self.dataset_drift.set(0.0)
    
    def calculate_dataset_drift_score(self, drift_metrics, synthetic_data_added=False):
        """Calculate and update dataset drift score based on drift metrics and synthetic data addition"""
        try:
            current_score = float(self.dataset_drift._value._value)
            
            # Base drift contribution from detected features
            n_drifted = drift_metrics.get('n_drifted_features', 0)
            drift_share = drift_metrics.get('share_drifted_features', 0.0)
            dataset_drift = drift_metrics.get('dataset_drift', False)
            
            # Calculate drift increment based on severity
            drift_increment = 0.0
            if n_drifted > 0:
                # Increment based on number of drifted features (0.1 per feature)
                drift_increment += n_drifted * 0.1
                
                # Additional increment based on drift share percentage
                drift_increment += drift_share * 0.5
                
                # Bonus increment if dataset-level drift is detected
                if dataset_drift:
                    drift_increment += 0.3
            
            # Additional increment when synthetic data is added
            if synthetic_data_added:
                drift_increment += 0.2  # Each synthetic data addition increases drift
            
            # Update dataset drift score (cap at maximum of 5.0)
            new_score = min(current_score + drift_increment, 5.0)
            self.dataset_drift.set(new_score)
            
            print(f"Dataset drift score updated: {current_score:.2f} → {new_score:.2f} (increment: +{drift_increment:.2f})")
            
            return new_score
            
        except Exception as e:
            print(f"Error calculating dataset drift score: {e}")
            return 0.0
    
    def reset_dataset_drift_after_retrain(self):
        """Reset dataset drift score to 0.0 after model retrain"""
        try:
            self.dataset_drift.set(0.0)
            print("Dataset drift score reset to 0.0 after model retrain")
        except Exception as e:
            print(f"Error resetting dataset drift score: {e}")
    
    def check_and_reset_dataset_drift_on_retrain(self):
        """Check if model was recently retrained and reset dataset drift score if needed"""
        try:
            model_path = "Model/personality_classifier.skops"
            if os.path.exists(model_path):
                # Get file modification time
                mod_time = os.path.getmtime(model_path)
                current_time = time.time()
                
                # If model was modified within last 5 minutes, consider it a retrain
                time_since_modified = current_time - mod_time
                if time_since_modified < 300:  # 5 minutes = 300 seconds
                    self.reset_dataset_drift_after_retrain()
                    return True
            return False
        except Exception as e:
            print(f"Error checking retrain status: {e}")
            return False
    
    def update_synthetic_metrics(self, generator_status, synthetic_data=None):
        """Update synthetic data generator metrics"""
        try:
            # Safe conversion for numeric values
            cycle = generator_status.get('cycle') or 0
            drift_strength = generator_status.get('drift_strength') or 0.0
            failed_detections = generator_status.get('failed_detections') or 0
            current_data_size = generator_status.get('current_data_size') or 0
            
            # Ensure values are numeric
            cycle = int(cycle) if isinstance(cycle, (int, float, str)) and str(cycle).isdigit() else 0
            drift_strength = float(drift_strength) if isinstance(drift_strength, (int, float, str)) else 0.0
            failed_detections = int(failed_detections) if isinstance(failed_detections, (int, float, str)) and str(failed_detections).isdigit() else 0
            current_data_size = int(current_data_size) if isinstance(current_data_size, (int, float, str)) and str(current_data_size).isdigit() else 0
            
            self.synthetic_drift_cycle.set(cycle)
            self.synthetic_drift_strength.set(drift_strength)
            self.synthetic_failed_detections.set(failed_detections)
            self.synthetic_data_size.set(current_data_size)

            # Update personality distribution if available
            personality_dist = generator_status.get('personality_distribution', {})
            total_samples = sum(personality_dist.values()) if personality_dist else 0
            if total_samples > 0:
                introvert_count = personality_dist.get('Introvert', 0)
                extrovert_count = personality_dist.get('Extrovert', 0)
                self.synthetic_introvert_ratio.set(introvert_count / total_samples)
                self.synthetic_extrovert_ratio.set(extrovert_count / total_samples)

            # Update info safely with no None values
            last_generation = generator_status.get('last_drift_detected') or 'Never'
            self.synthetic_info.info({
                'cycle': str(cycle),
                'drift_strength': f"{drift_strength:.3f}",
                'failed_detections': str(failed_detections),
                'last_generation': str(last_generation),
                'data_size': str(current_data_size)
            })
        except Exception as e:
            print(f"Error updating synthetic metrics: {e}")
            # Set fallback values
            self.synthetic_drift_cycle.set(0)
            self.synthetic_drift_strength.set(0.0)
            self.synthetic_failed_detections.set(0)
            self.synthetic_data_size.set(0)
    
    def record_prediction(self, latency=None):
        """Record a prediction event"""
        self.prediction_counter.inc()
        if latency is not None:
            self.prediction_latency.observe(latency)
    
    def set_model_info(self, model_name, model_version, training_date):
        """Set model information"""
        self.model_info.info({
            'model_name': model_name,
            'model_version': model_version,
            'training_date': training_date
        })
    
    def run_monitoring_cycle(self):
        """Run a complete monitoring cycle with fallback evaluation"""
        print("\n" + "="*60)
        print("RUNNING MLOPS MONITORING CYCLE")
        print("="*60)
        
        # 1. Check for recent retrain and evaluate performance (NEW - replaces old check)
        retrain_detected = self.check_model_retrain_and_evaluate()
        
        # 2. Load latest model metrics (if no retrain detected)
        if not retrain_detected:
            self._load_model_metrics()
        
        # 3. Run drift detection
        self._run_drift_detection()
        
        # 4. Update synthetic data generator metrics
        self._update_synthetic_data_metrics()
        
        # 5. Update model info (if no retrain detected)
        if not retrain_detected:
            self._update_model_info()
        
        # 6. Adaptive drift management for fallback models
        self._manage_drift_for_fallback_models()
        
        print("Monitoring cycle completed")
        print("="*60)
    
    def _load_model_metrics(self):
        """Load model metrics from results"""
        try:
            metrics_path = "Results/metrics.txt"
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    content = f.read()
                    
                # Parse metrics with better error handling
                accuracy_found = False
                auc_found = False
                
                if "Accuracy" in content:
                    try:
                        accuracy_part = content.split("Accuracy = ")[1].split(",")[0]
                        accuracy = float(accuracy_part.strip())
                        self.model_accuracy.set(accuracy)
                        self.current_model_metrics['accuracy'] = accuracy  # Update current metrics
                        accuracy_found = True
                        print(f"Updated model accuracy: {accuracy:.4f}")
                    except (ValueError, IndexError) as e:
                        print(f"Could not parse accuracy: {e}")
                
                if "AUC" in content:
                    try:
                        # Better parsing for AUC - handle newlines and extra text
                        auc_part = content.split("AUC = ")[1]
                        # Remove everything after newline or comma
                        auc_part = auc_part.split('\n')[0].split(',')[0].strip()
                        auc = float(auc_part)
                        self.model_auc.set(auc)
                        self.current_model_metrics['auc'] = auc  # Update current metrics
                        auc_found = True
                        print(f"Updated model AUC: {auc:.4f}")
                    except (ValueError, IndexError) as e:
                        print(f"Could not parse AUC: {e}")
                
                # Set to 0 if parsing failed
                if not accuracy_found:
                    self.model_accuracy.set(0.0)
                    self.current_model_metrics['accuracy'] = 0.0
                    print("Accuracy not found or parsing failed - set to 0.0")
                if not auc_found:
                    self.model_auc.set(0.0)
                    self.current_model_metrics['auc'] = 0.0
                    print("AUC not found or parsing failed - set to 0.0")
            else:
                # File doesn't exist - set metrics to 0
                self.model_accuracy.set(0.0)
                self.model_auc.set(0.0)
                self.current_model_metrics['accuracy'] = 0.0
                self.current_model_metrics['auc'] = 0.0
                print(f"Metrics file not found: {metrics_path} - set accuracy and AUC to 0.0")
                        
        except Exception as e:
            print(f"Could not load model metrics: {e}")
            # Set to 0 on any error
            self.model_accuracy.set(0.0)
            self.model_auc.set(0.0)
            self.current_model_metrics['accuracy'] = 0.0
            self.current_model_metrics['auc'] = 0.0
            print("Error loading metrics - set accuracy and AUC to 0.0")
    
    def _run_drift_detection(self):
        """Run data drift detection and update metrics"""
        try:
            detector = DataDriftDetector()
            drift_metrics = detector.detect_drift()
            self.update_drift_metrics(drift_metrics)
            
            # Calculate and update dataset drift score based on detection results
            self.calculate_dataset_drift_score(drift_metrics, synthetic_data_added=False)
            
            print(f"Updated drift metrics - Features drifted: {drift_metrics['n_drifted_features']}")
        except Exception as e:
            print(f"Could not run drift detection: {e}")
            # Fallback to default metrics
            drift_metrics = {
                'n_drifted_features': 0,
                'share_drifted_features': 0.0,
                'dataset_drift': False
            }
            self.update_drift_metrics(drift_metrics)
            self.calculate_dataset_drift_score(drift_metrics, synthetic_data_added=False)
    
    def _update_synthetic_data_metrics(self):
        """Update synthetic data generator metrics"""
        try:
            if self.data_generator:
                # Get status from data generator
                generator_status = self.data_generator.get_status()
                self.update_synthetic_metrics(generator_status)

                # Safe formatting for print statements
                cycle = generator_status.get('cycle', 0)
                drift_strength = generator_status.get('drift_strength', 0.0)
                data_size = generator_status.get('current_data_size', 'N/A')
                
                if isinstance(drift_strength, (int, float)):
                    print(f"Updated synthetic data metrics - Cycle: {cycle}, "
                        f"Strength: {drift_strength:.3f}, "
                        f"Data size: {data_size}")
                else:
                    print(f"Updated synthetic data metrics - Cycle: {cycle}, "
                        f"Strength: {drift_strength}, "
                        f"Data size: {data_size}")
            else:
                print("Data generator not available, using default synthetic metrics")
                default_status = {
                    'cycle': 0,
                    'drift_strength': 0.1,
                    'failed_detections': 0,
                    'current_data_size': 0
                }
                self.update_synthetic_metrics(default_status)

        except Exception as e:
            print(f"Could not update synthetic data metrics: {e}")
            fallback_status = {
                'cycle': 0,
                'drift_strength': 0.0,
                'failed_detections': 0,
                'current_data_size': 0
            }
            self.update_synthetic_metrics(fallback_status)

    
    def generate_synthetic_data_cycle(self, n_samples=200):
        """Generate new synthetic data and update metrics"""
        try:
            print(f"\nGenerating {n_samples} synthetic samples...")
            data, drift_info = self.data_generator.generate_drifted_data(n_new_samples=n_samples)
            
            # Increment generation counter
            self.synthetic_generation_counter.inc()
            
            # Update new samples metric
            self.synthetic_new_samples.set(drift_info['new_samples'])
            
            # Update all synthetic metrics
            generator_status = self.data_generator.get_status()
            self.update_synthetic_metrics(generator_status, data)
            
            # After adding synthetic data, recalculate drift detection and update dataset drift score
            try:
                detector = DataDriftDetector()
                drift_metrics = detector.detect_drift()
                self.update_drift_metrics(drift_metrics)
                
                # Calculate dataset drift score with synthetic data addition flag
                self.calculate_dataset_drift_score(drift_metrics, synthetic_data_added=True)
                
                print(f"Drift metrics recalculated after synthetic data addition")
            except Exception as e:
                print(f"Could not recalculate drift after synthetic data: {e}")
                # Still update dataset drift score for synthetic data addition even if detection fails
                fallback_metrics = {
                    'n_drifted_features': 1,  # Assume some drift from synthetic data
                    'share_drifted_features': 0.1,
                    'dataset_drift': False
                }
                self.calculate_dataset_drift_score(fallback_metrics, synthetic_data_added=True)
            
            print(f"Generated {drift_info['new_samples']} new synthetic samples")
            print(f"Total dataset size: {drift_info['total_size']}")
            print(f"Drift type: {drift_info['drift_type']}")
            
            # Safe formatting for drift strength
            drift_strength = drift_info.get('drift_strength', 0.0)
            if isinstance(drift_strength, (int, float)):
                print(f"Drift strength: {drift_strength:.3f}")
            else:
                print(f"Drift strength: {drift_strength}")
            
            return data, drift_info
            
        except Exception as e:
            print(f"Error generating synthetic data: {e}")
            return None, None
    
    def _update_model_info(self):
        """Update model information"""
        try:
            model_path = "Model/personality_classifier.skops"
            if os.path.exists(model_path):
                # Get file modification time
                mod_time = os.path.getmtime(model_path)
                training_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
                
                self.set_model_info(
                    model_name="RandomForestPersonality",
                    model_version="latest",
                    training_date=training_date
                )
                print(f"Updated model info - Training date: {training_date}")
        except Exception as e:
            print(f"Could not update model info: {e}")
    
    def run_extended_monitoring_cycle(self, generate_data=True, n_samples=200):
        """Run extended monitoring cycle with optional synthetic data generation"""
        print("\n" + "="*70)
        print("RUNNING EXTENDED MLOPS MONITORING CYCLE WITH SYNTHETIC DATA")
        print("="*70)
        
        # 1. Load latest model metrics
        self._load_model_metrics()
        
        # 2. Check for recent retrain and reset dataset drift score if needed
        self.check_and_reset_dataset_drift_on_retrain()
        
        # 3. Run drift detection
        self._run_drift_detection()
        
        # 4. Generate new synthetic data if requested
        if generate_data:
            data, drift_info = self.generate_synthetic_data_cycle(n_samples)
            if data is not None:
                print("Synthetic data generation completed")
        
        # 5. Update synthetic data generator metrics (final update)
        self._update_synthetic_data_metrics()
        
        # 6. Update model info
        self._update_model_info()
        
        print("Extended monitoring cycle completed")
        print("="*70)


    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        print("Monitoring stopped")
    
    def manual_reset_dataset_drift_score(self):
        """Manually reset dataset drift score - useful after model retrain"""
        self.reset_dataset_drift_after_retrain()
        print("Manual dataset drift score reset completed - use this after retraining your model")

    def backup_current_model(self):
        """Backup current model before retrain"""
        try:
            import shutil
            model_path = "Model/personality_classifier.skops"
            
            if os.path.exists(model_path):
                shutil.copy2(model_path, self.model_backup_path)
                
                # Store current metrics as previous
                self.previous_model_metrics = self.current_model_metrics.copy()
                self.previous_model_accuracy.set(self.previous_model_metrics['accuracy'])
                self.previous_model_auc.set(self.previous_model_metrics['auc'])
                
                print(f"Model backed up to {self.model_backup_path}")
                print(f"Previous model metrics - Accuracy: {self.previous_model_metrics['accuracy']:.4f}, AUC: {self.previous_model_metrics['auc']:.4f}")
                return True
        except Exception as e:
            print(f"Error backing up model: {e}")
            return False
        
    def evaluate_model_performance(self, new_accuracy, new_auc, fallback_threshold=0.02):
        """Evaluate if new model performance is acceptable or needs fallback"""
        try:
            # Update current metrics
            self.current_model_metrics = {'accuracy': new_accuracy, 'auc': new_auc}
            
            # Check if we have previous metrics to compare
            if self.previous_model_metrics['accuracy'] == 0.0 and self.previous_model_metrics['auc'] == 0.0:
                print("No previous model metrics for comparison - accepting new model")
                return True, "no_previous_model"
            
            # Calculate performance difference
            accuracy_diff = new_accuracy - self.previous_model_metrics['accuracy']
            auc_diff = new_auc - self.previous_model_metrics['auc']
            
            # Determine if fallback is needed
            accuracy_degraded = accuracy_diff < -fallback_threshold
            auc_degraded = auc_diff < -fallback_threshold
            
            print(f"Performance comparison:")
            print(f"  Accuracy: {self.previous_model_metrics['accuracy']:.4f} → {new_accuracy:.4f} (diff: {accuracy_diff:+.4f})")
            print(f"  AUC: {self.previous_model_metrics['auc']:.4f} → {new_auc:.4f} (diff: {auc_diff:+.4f})")
            print(f"  Threshold: {fallback_threshold}")
            
            if accuracy_degraded or auc_degraded:
                reason = []
                if accuracy_degraded:
                    reason.append(f"accuracy_drop_{abs(accuracy_diff):.4f}")
                if auc_degraded:
                    reason.append(f"auc_drop_{abs(auc_diff):.4f}")
                
                fallback_reason = "_".join(reason)
                print(f"Performance degradation detected: {fallback_reason}")
                return False, fallback_reason
            else:
                print("Performance acceptable - keeping new model")
                return True, "performance_improved"
                
        except Exception as e:
            print(f"Error evaluating model performance: {e}")
            return True, "evaluation_error"  # Default to accepting new model on error
    
    def trigger_model_fallback(self, reason="performance_degradation"):
        """Trigger model fallback to previous version"""
        try:
            import shutil
            model_path = "Model/personality_classifier.skops"
            
            if os.path.exists(self.model_backup_path):
                # Restore backup model
                shutil.copy2(self.model_backup_path, model_path)
                
                # Update metrics
                self.model_fallback_triggered.inc()
                self.model_is_fallback.set(1)
                
                # Restore previous model metrics
                self.model_accuracy.set(self.previous_model_metrics['accuracy'])
                self.model_auc.set(self.previous_model_metrics['auc'])
                self.current_model_metrics = self.previous_model_metrics.copy()
                
                # Update model info
                self.model_info.info({
                    'model_name': 'RandomForestPersonality',
                    'model_version': 'fallback',
                    'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'is_fallback': 'true',
                    'fallback_reason': reason
                })
                
                print(f"MODEL FALLBACK TRIGGERED - Reason: {reason}")
                print(f"Restored to previous model with Accuracy: {self.previous_model_metrics['accuracy']:.4f}, AUC: {self.previous_model_metrics['auc']:.4f}")
                
                # Keep dataset drift score elevated to encourage drift handling
                current_drift_score = float(self.dataset_drift._value._value)
                if current_drift_score < 2.0:  # Maintain minimum drift awareness
                    self.dataset_drift.set(2.0)
                    print(f"Maintained dataset drift score at 2.0 to encourage drift handling")
                
                return True
            else:
                print("No backup model available for fallback")
                return False
                
        except Exception as e:
            print(f"Error during model fallback: {e}")
            return False
    
    def check_model_retrain_and_evaluate(self):
        """Check if model was retrained and evaluate if fallback is needed"""
        try:
            model_path = "Model/personality_classifier.skops"
            if not os.path.exists(model_path):
                return False
            
            # Get file modification time
            mod_time = os.path.getmtime(model_path)
            current_time = time.time()
            
            # If model was modified within last 5 minutes, consider it a retrain
            time_since_modified = current_time - mod_time
            if time_since_modified < 300:  # 5 minutes = 300 seconds
                print(f"Recent model retrain detected (modified {time_since_modified:.1f} seconds ago)")
                
                # Increment retrain attempts counter
                self.retrain_attempts.inc()
                
                # Load new model metrics
                self._load_model_metrics()
                new_accuracy = float(self.model_accuracy._value._value)
                new_auc = float(self.model_auc._value._value)
                
                # Evaluate performance
                is_acceptable, reason = self.evaluate_model_performance(new_accuracy, new_auc)
                
                if is_acceptable:
                    # Accept new model
                    self.successful_retrains.inc()
                    self.model_is_fallback.set(0)  # Mark as current model
                    
                    # Only reset drift score if performance significantly improved
                    if reason == "performance_improved":
                        self.reset_dataset_drift_after_retrain()
                    else:
                        # Keep some drift awareness for marginal improvements
                        current_drift_score = float(self.dataset_drift._value._value)
                        reduced_score = max(current_drift_score * 0.5, 1.0)
                        self.dataset_drift.set(reduced_score)
                        print(f"Reduced dataset drift score to {reduced_score:.1f} (partial reset)")
                    
                    # Update model info
                    self.model_info.info({
                        'model_name': 'RandomForestPersonality',
                        'model_version': 'latest',
                        'training_date': datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S"),
                        'is_fallback': 'false',
                        'fallback_reason': 'none'
                    })
                    
                    print(f"New model accepted - Accuracy: {new_accuracy:.4f}, AUC: {new_auc:.4f}")
                    return True
                else:
                    # Trigger fallback
                    self.trigger_model_fallback(reason)
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking retrain and evaluation: {e}")
            return False
    
    def _manage_drift_for_fallback_models(self):
        """Manage drift detection for fallback models to encourage retraining"""
        try:
            is_fallback = float(self.model_is_fallback._value._value) == 1.0
            current_drift_score = float(self.dataset_drift._value._value)
            
            if is_fallback:
                # For fallback models, maintain higher drift sensitivity
                if current_drift_score < 1.5:
                    self.dataset_drift.set(1.5)
                    print("Fallback model detected - maintaining elevated drift score for retraining encouragement")
                
                # Generate additional synthetic data to create more drift pressure
                if self.data_generator and current_drift_score < 3.0:
                    try:
                        print("Generating additional synthetic data for fallback model...")
                        self.generate_synthetic_data_cycle(n_samples=100)
                    except Exception as e:
                        print(f"Could not generate additional synthetic data: {e}")
            
        except Exception as e:
            print(f"Error managing drift for fallback models: {e}")


def main():
    """Main function to run monitoring"""
    monitoring = MLOpsMonitoring(port=8000)
    
    try:
        # Start monitoring server
        monitoring.start_monitoring()
        
        # Backup current model before starting monitoring
        monitoring.backup_current_model()
        
        # Run initial monitoring cycle
        monitoring.run_monitoring_cycle()
        
        print("\nMonitoring is now active with Model Fallback Protection!")
        print("Prometheus metrics: http://localhost:8000/metrics")
        print("You can now configure Grafana to visualize these metrics")
        print("\nNew Model Fallback Metrics Available:")
        print("   - previous_model_accuracy: Previous model accuracy for comparison")
        print("   - previous_model_auc: Previous model AUC for comparison")
        print("   - model_fallback_triggered_total: Number of times model fallback was triggered")
        print("   - model_is_fallback: Whether current model is a fallback model (1=fallback, 0=current)")
        print("   - model_retrain_attempts_total: Total number of retrain attempts")
        print("   - model_successful_retrains_total: Total number of successful retrains")
        print("\nSynthetic Data Generator Metrics Available:")
        print("   - synthetic_drift_cycle: Current drift generation cycle")
        print("   - synthetic_drift_strength: Current adaptive drift strength")
        print("   - synthetic_failed_detections: Number of failed drift detections")
        print("   - synthetic_data_size_total: Total size of synthetic dataset")
        print("   - synthetic_new_samples_last: New samples generated in last cycle")
        print("   - synthetic_introvert_ratio: Ratio of introverts in synthetic data")
        print("   - synthetic_extrovert_ratio: Ratio of extroverts in synthetic data")
        print("   - synthetic_generation_total: Total number of synthetic data generations")
        print("   - drift_score: Dataset drift score for alerting (resets after retrain, increases with synthetic data)")
        print("\nFallback Protection:")
        print("   - Models with performance degradation > 2% will trigger fallback")
        print("   - Fallback models maintain elevated drift scores to encourage retraining")
        print("   - Additional synthetic data generated for fallback models")
        print("\nPress Ctrl+C to stop monitoring...")
        
        cycle_count = 0
        
        # Keep monitoring running
        while True:
            time.sleep(60)  # Wait 1 minute
            cycle_count += 1
            
            # Backup model before potential retrain cycles
            if cycle_count % 10 == 0:  # Backup every 10 minutes
                monitoring.backup_current_model()
            
            # Run extended monitoring cycle every 5 minutes (generate data every 5 cycles)
            if cycle_count % 5 == 0:
                print(f"\nRunning extended cycle #{cycle_count//5} with synthetic data generation...")
                monitoring.run_extended_monitoring_cycle(generate_data=True, n_samples=150)
            else:
                # Regular monitoring cycle
                monitoring.run_monitoring_cycle()
            
    except KeyboardInterrupt:
        print("\n\nStopping monitoring...")
        monitoring.stop_monitoring()
    except Exception as e:
        print(f"Error in monitoring: {e}")
        monitoring.stop_monitoring()


if __name__ == "__main__":
    main()
