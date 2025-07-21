import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import mlflow
import mlflow.sklearn
import scipy.stats as stats
from prometheus_client import Gauge
from scipy.stats import ks_2samp, chi2_contingency, levene
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Gauge Prometheus untuk metrik jarak Wasserstein per fitur
feature_drift_gauge = Gauge(
    'feature_drift_wasserstein',
    'Wasserstein distance per feature between reference and current data',
    ['feature', 'method']
)

# Gauge Prometheus untuk distribusi kelas target
class_distribution_gauge = Gauge(
    'class_distribution_ratio', 
    'Ratio of each personality class',
    ['personality_type']
)

# Gauge Prometheus untuk Jensen-Shannon divergence pada target
target_drift_gauge = Gauge(
    'target_distribution_drift',
    'Jensen-Shannon divergence for target class distribution'
)

# Gauge Prometheus untuk model performance
model_performance_gauge = Gauge(
    'model_accuracy_current',
    'Current model accuracy on new data'
)


class DataDriftDetector:
    def __init__(self, reference_data_path="Data/personality_datasert.csv"):
        self.reference_data_path = reference_data_path
        self.reference_data = None
        self.current_data = None
        self.results_dir = "Results"
        
        # More sensitive configuration for subtle drift detection
        self.significance_level = 0.05  # Less stringent p-value for subtle drift
        self.effect_size_threshold = 0.1  # Lower Cohen's d threshold
        self.psi_threshold = 0.05  # Lower PSI threshold for categorical
        self.distance_threshold = 0.05  # Lower distance threshold
        self.auc_threshold = 0.65  # Lower model-based detection threshold
        
        # Dataset-level thresholds (more sensitive)
        self.feature_drift_threshold = 0.10  # 10% of features for dataset drift
        self.critical_drift_threshold = 0.25   # 25% for critical alert
        
        # Define which columns should be treated as numerical vs categorical
        self.numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 
                                 'Going_outside', 'Friends_circle_size', 'Post_frequency']
        self.categorical_features = ['Stage_fear', 'Drained_after_socializing']
        
        # Target column untuk monitoring distribusi kelas
        self.target_column = 'Personality'

    def load_reference_data(self):
        print("Loading reference data...")
        self.reference_data = pd.read_csv(self.reference_data_path)
        categorical_columns = ["Stage_fear", "Drained_after_socializing"]
        for col in categorical_columns:
            if col in self.reference_data.columns:
                self.reference_data[col] = self.reference_data[col].map({"Yes": 1, "No": 0})
        print(f"Reference data shape: {self.reference_data.shape}")
        return self.reference_data

    def load_current_data(self, current_data_path="Data/synthetic_ctgan_data.csv"):
        print("Loading current data...")
        self.current_data = pd.read_csv(current_data_path)
        categorical_columns = ["Stage_fear", "Drained_after_socializing"]
        for col in categorical_columns:
            if col in self.current_data.columns:
                self.current_data[col] = self.current_data[col].map({"Yes": 1, "No": 0})
        print(f"Current data shape: {self.current_data.shape}")
        return self.current_data

    def detect_drift(self):
        if self.reference_data is None:
            self.load_reference_data()
        if self.current_data is None:
            self.load_current_data()

        print("Running robust statistical drift detection...")

        common_columns = set(self.reference_data.columns).intersection(set(self.current_data.columns))
        common_columns.discard('Personality')

        ref_data = self.reference_data[list(common_columns)]
        curr_data = self.current_data[list(common_columns)]

        total_features = len(common_columns)
        drift_results = {}
        drifted_features = 0
        high_severity_features = 0

        for col in common_columns:
            if col in ref_data.columns and col in curr_data.columns:
                feature_result = self._detect_feature_drift(
                    ref_data[col], curr_data[col], col
                )
                drift_results[col] = feature_result
                
                if feature_result['drift_detected']:
                    drifted_features += 1
                    
                if feature_result.get('severity') == 'HIGH':
                    high_severity_features += 1

        # Detect target distribution drift (class imbalance)
        if (self.target_column in self.reference_data.columns and 
            self.target_column in self.current_data.columns):
            target_result = self._detect_target_drift(
                self.reference_data[self.target_column], 
                self.current_data[self.target_column]
            )
            drift_results['_target_distribution'] = target_result
            
            if target_result['drift_detected']:
                drifted_features += 1
                if target_result.get('severity') == 'HIGH':
                    high_severity_features += 1

        # Detect model performance drift
        performance_result = self._detect_model_performance_drift()
        if performance_result:
            drift_results['_model_performance'] = performance_result

        # Calculate overall drift metrics
        drift_metrics = {
            'n_drifted_features': drifted_features,
            'share_drifted_features': drifted_features / total_features if total_features > 0 else 0.0,
            'dataset_drift': drifted_features > total_features * self.feature_drift_threshold,
            'critical_drift': drifted_features > total_features * self.critical_drift_threshold,
            'high_severity_features': high_severity_features,
            'feature_details': drift_results
        }

        # Assess drift severity
        severity_assessment = self._assess_drift_severity(drift_metrics)
        drift_metrics.update(severity_assessment)

        # Save results
        self._save_drift_results(drift_metrics, ref_data, curr_data)

        print(f"✓ Drift detection completed. Features drifted: {drifted_features}/{total_features}")
        print(f"✓ Severity level: {drift_metrics['severity_level']}")
        return drift_metrics

    def run_drift_tests(self):
        drift_metrics = self.detect_drift()
        
        # More comprehensive test suite
        dataset_drift_test = 1 if not drift_metrics['dataset_drift'] else 0
        feature_drift_test = 1 if drift_metrics['share_drifted_features'] < self.feature_drift_threshold else 0
        critical_drift_test = 1 if not drift_metrics['critical_drift'] else 0
        severity_test = 1 if drift_metrics['severity_level'] in ['GREEN', 'YELLOW'] else 0
        
        total_tests = 4
        passed_tests = dataset_drift_test + feature_drift_test + critical_drift_test + severity_test

        test_results = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'test_success_rate': passed_tests / total_tests,
            'dataset_drift_test': bool(dataset_drift_test),
            'feature_drift_test': bool(feature_drift_test), 
            'critical_drift_test': bool(critical_drift_test),
            'severity_test': bool(severity_test),
            'overall_health': 'HEALTHY' if passed_tests >= 3 else 'WARNING' if passed_tests >= 2 else 'CRITICAL'
        }

        print(f"✓ Robust drift tests completed: {passed_tests}/{total_tests} passed")
        print(f"✓ Overall system health: {test_results['overall_health']}")
        return test_results

    def _detect_feature_drift(self, ref_data, curr_data, feature_name):
        """Robust feature-level drift detection using multiple statistical tests"""
        
        # Remove NaN values
        ref_clean = ref_data.dropna()
        curr_clean = curr_data.dropna()
        
        if len(ref_clean) == 0 or len(curr_clean) == 0:
            return {
                'drift_detected': False,
                'p_value': 1.0,
                'method': 'insufficient_data',
                'severity': 'NONE'
            }
        
        # Detect feature type using predefined lists
        if feature_name in self.categorical_features:
            return self._detect_categorical_drift(ref_clean, curr_clean, feature_name)
        elif feature_name in self.numerical_features:
            return self._detect_numerical_drift(ref_clean, curr_clean, feature_name)
        else:
            # Fallback to automatic detection
            if self._is_categorical(ref_clean):
                return self._detect_categorical_drift(ref_clean, curr_clean, feature_name)
            else:
                return self._detect_numerical_drift(ref_clean, curr_clean, feature_name)
    
    def _detect_target_drift(self, ref_target, curr_target):
        """Khusus untuk mendeteksi perubahan distribusi kelas target"""
        try:
            from scipy.spatial.distance import jensenshannon
            
            # Hitung distribusi kelas
            ref_dist = ref_target.value_counts(normalize=True).sort_index()
            curr_dist = curr_target.value_counts(normalize=True).sort_index()
            
            # Align indices untuk memastikan konsistensi
            all_classes = sorted(set(ref_dist.index) | set(curr_dist.index))
            ref_aligned = ref_dist.reindex(all_classes, fill_value=0)
            curr_aligned = curr_dist.reindex(all_classes, fill_value=0)
            
            # Jensen-Shannon divergence untuk distribusi kelas
            js_distance = jensenshannon(ref_aligned, curr_aligned)
            
            # Log distribusi kelas ke Prometheus
            for personality_type in all_classes:
                class_distribution_gauge.labels(
                    personality_type=str(personality_type)
                ).set(curr_aligned.get(personality_type, 0))
            
            # Log JS distance ke Prometheus
            target_drift_gauge.set(js_distance)
            
            # Tentukan severity berdasarkan JS distance
            if js_distance > 0.2:
                severity = 'HIGH'
            elif js_distance > 0.1:
                severity = 'MEDIUM'  
            elif js_distance > 0.05:
                severity = 'LOW'
            else:
                severity = 'NONE'
                
            return {
                'drift_detected': js_distance > 0.05,  # 5% threshold untuk sensitivitas
                'js_distance': js_distance,
                'severity': severity,
                'method': 'jensen_shannon_divergence',
                'ref_distribution': ref_aligned.to_dict(),
                'curr_distribution': curr_aligned.to_dict()
            }
            
        except Exception as e:
            print(f"⚠️ Error in target drift detection: {e}")
            return {
                'drift_detected': False,
                'js_distance': 0.0,
                'method': 'error',
                'severity': 'NONE',
                'error': str(e)
            }

    def _detect_model_performance_drift(self):
        """Deteksi drift berdasarkan performa model pada data baru"""
        try:
            import skops.io as sio
            from skops.io import get_untrusted_types
            
            # Coba load model dari berbagai lokasi
            model_paths = [
                "Model/personality_classifier.skops",
                "../Model/personality_classifier.skops",
                "/app/Model/personality_classifier.skops"
            ]
            
            model = None
            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        unknown_types = get_untrusted_types(file=model_path)
                        model = sio.load(model_path, trusted=unknown_types)
                        break
                    except Exception as e:
                        continue
            
            if model is None:
                return {
                    'current_accuracy': 0.0,
                    'performance_drop': 0.0,
                    'drift_detected': False,
                    'method': 'model_not_found',
                    'severity': 'NONE'
                }
            
            # Evaluate model pada current data
            if (self.current_data is not None and 
                self.target_column in self.current_data.columns):
                
                X_curr = self.current_data.drop(self.target_column, axis=1, errors='ignore')
                y_curr = self.current_data[self.target_column]
                
                # Encode categorical features seperti di training
                categorical_columns = ["Stage_fear", "Drained_after_socializing"]
                for col in categorical_columns:
                    if col in X_curr.columns:
                        X_curr = X_curr.copy()
                        X_curr[col] = X_curr[col].map({"Yes": 1, "No": 0})
                
                # Encode target labels like in training (convert string labels to numeric)
                # Load label encoder if available
                label_encoder_paths = [
                    "Model/label_encoder.joblib",
                    "../Model/label_encoder.joblib", 
                    "/app/Model/label_encoder.joblib"
                ]
                
                y_curr_encoded = y_curr.copy()
                for encoder_path in label_encoder_paths:
                    if os.path.exists(encoder_path):
                        try:
                            import joblib
                            label_encoder = joblib.load(encoder_path)
                            # Only encode if y_curr contains string labels
                            if y_curr.dtype == 'object' or isinstance(y_curr.iloc[0], str):
                                y_curr_encoded = label_encoder.transform(y_curr)
                            break
                        except Exception as e:
                            continue
                
                # If no encoder found and labels are strings, do manual encoding
                if y_curr.dtype == 'object' or isinstance(y_curr.iloc[0], str):
                    # Check if encoding was applied
                    import numpy as np
                    encoding_applied = False
                    
                    if isinstance(y_curr_encoded, np.ndarray):
                        # If it's a numpy array, encoding was applied
                        encoding_applied = True
                    elif hasattr(y_curr_encoded, 'equals') and hasattr(y_curr, 'equals'):
                        # For pandas series, use equals method
                        encoding_applied = not y_curr_encoded.equals(y_curr)
                    else:
                        # Fallback: check if they're the same object
                        encoding_applied = y_curr_encoded is not y_curr
                    
                    if not encoding_applied:  # No encoding was applied
                        label_mapping = {"Introvert": 0, "Extrovert": 1}
                        y_curr_encoded = y_curr.map(label_mapping)
                        # Fill any NaN values with 0 as fallback
                        y_curr_encoded = y_curr_encoded.fillna(0)
                
                # Hitung accuracy
                accuracy = model.score(X_curr, y_curr_encoded)
                
                # Log ke Prometheus
                model_performance_gauge.set(accuracy)
                
                # Baseline accuracy (ambil dari hasil training sebelumnya atau set manual)
                baseline_accuracy = 0.85  # Bisa diambil dari metrics.txt atau MLflow
                performance_drop = baseline_accuracy - accuracy
                
                # Tentukan severity
                if performance_drop > 0.15:  # 15% drop
                    severity = 'HIGH'
                elif performance_drop > 0.10:  # 10% drop
                    severity = 'MEDIUM'
                elif performance_drop > 0.05:  # 5% drop
                    severity = 'LOW'
                else:
                    severity = 'NONE'
                
                return {
                    'current_accuracy': accuracy,
                    'baseline_accuracy': baseline_accuracy,
                    'performance_drop': performance_drop,
                    'drift_detected': performance_drop > 0.05,
                    'severity': severity,
                    'method': 'model_accuracy_comparison'
                }
            
        except Exception as e:
            print(f"⚠️ Error in model performance drift detection: {e}")
            return {
                'current_accuracy': 0.0,
                'performance_drop': 0.0,
                'drift_detected': False,
                'method': 'error',
                'severity': 'NONE',
                'error': str(e)
            }

    def _detect_numerical_drift(self, ref_data, curr_data, feature_name):
        """Detect drift in numerical features using multiple statistical tests"""
        
        try:
            # 1. Kolmogorov-Smirnov test (distribution comparison)
            ks_stat, ks_p_value = ks_2samp(ref_data, curr_data)
            
            # 2. Effect size (Cohen's d)
            cohens_d = self._calculate_cohens_d(ref_data, curr_data)
            
            # 3. Welch's t-test (mean comparison with unequal variances)
            t_stat, t_p_value = stats.ttest_ind(ref_data, curr_data, equal_var=False)
            
            # 4. F-test for variance (Levene's test)
            f_stat, f_p_value = levene(ref_data, curr_data)
            
            # 5. Wasserstein distance
            wasserstein_dist = stats.wasserstein_distance(ref_data, curr_data)
            
            # Prometheus gauge - catat nilai Wasserstein ke Prometheus
            feature_drift_gauge.labels(feature=feature_name, method='wasserstein').set(wasserstein_dist)
            
            data_range = max(ref_data.max(), curr_data.max()) - min(ref_data.min(), curr_data.min())
            normalized_distance = wasserstein_dist / data_range if data_range > 0 else 0
            
            # Drift detected if ANY condition met (more sensitive):
            drift_conditions = [
                ks_p_value < self.significance_level,  # Distribution shift
                abs(cohens_d) > self.effect_size_threshold,  # Effect size (lowered)
                t_p_value < self.significance_level and abs(cohens_d) > 0.05,  # Mean shift (lowered)
                f_p_value < self.significance_level,  # Variance change
                normalized_distance > self.distance_threshold  # Distance-based (lowered)
            ]
            
            # Determine severity
            severity = self._determine_severity(cohens_d, min(ks_p_value, t_p_value, f_p_value))
            
            return {
                'drift_detected': any(drift_conditions),
                'p_value': min(ks_p_value, t_p_value, f_p_value),
                'effect_size': cohens_d,
                'normalized_distance': normalized_distance,
                'ks_test': {'statistic': ks_stat, 'p_value': ks_p_value},
                't_test': {'statistic': t_stat, 'p_value': t_p_value},
                'variance_test': {'statistic': f_stat, 'p_value': f_p_value},
                'wasserstein_distance': wasserstein_dist,
                'severity': severity,
                'method': 'statistical_tests'
            }
            
        except Exception as e:
            print(f"⚠️ Error in numerical drift detection for {feature_name}: {e}")
            return {
                'drift_detected': False,
                'p_value': 1.0,
                'method': 'error',
                'severity': 'NONE',
                'error': str(e)
            }
    
    def _detect_categorical_drift(self, ref_data, curr_data, feature_name):
        """Detect drift in categorical features"""
        
        try:
            # Get unique categories from both datasets
            all_categories = sorted(set(ref_data.unique()) | set(curr_data.unique()))
            
            # Create contingency table
            ref_counts = ref_data.value_counts().reindex(all_categories, fill_value=0)
            curr_counts = curr_data.value_counts().reindex(all_categories, fill_value=0)
            
            # Chi-square test
            contingency_table = np.array([ref_counts.values, curr_counts.values])
            
            # Avoid chi-square test if expected frequencies are too low
            expected_freq = contingency_table.sum() * 0.05
            if (contingency_table < 5).any() or contingency_table.sum() < expected_freq:
                # Fall back to proportion comparison
                ref_props = ref_counts / ref_counts.sum()
                curr_props = curr_counts / curr_counts.sum()
                max_prop_diff = abs(ref_props - curr_props).max()
                
                return {
                    'drift_detected': max_prop_diff > 0.1,  # 10% proportion change
                    'p_value': 0.05 if max_prop_diff > 0.1 else 0.5,
                    'max_proportion_difference': max_prop_diff,
                    'method': 'proportion_comparison',
                    'severity': 'HIGH' if max_prop_diff > 0.2 else 'MEDIUM' if max_prop_diff > 0.1 else 'LOW'
                }
            
            chi2_stat, chi2_p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Population Stability Index (PSI)
            psi = self._calculate_psi(ref_counts, curr_counts)
            
            # Drift conditions
            drift_conditions = [
                chi2_p_value < self.significance_level,  # Statistical significance
                psi > self.psi_threshold  # PSI threshold
            ]
            
            # Determine severity
            if psi > 0.25:
                severity = 'HIGH'
            elif psi > 0.2:
                severity = 'MEDIUM'
            elif psi > 0.1:
                severity = 'LOW'
            else:
                severity = 'NONE'
            
            return {
                'drift_detected': any(drift_conditions),
                'p_value': chi2_p_value,
                'psi': psi,
                'chi2_test': {'statistic': chi2_stat, 'p_value': chi2_p_value, 'dof': dof},
                'severity': severity,
                'method': 'categorical_tests'
            }
            
        except Exception as e:
            print(f"⚠️ Error in categorical drift detection for {feature_name}: {e}")
            return {
                'drift_detected': False,
                'p_value': 1.0,
                'method': 'error',
                'severity': 'NONE',
                'error': str(e)
            }
    
    def _calculate_cohens_d(self, ref_data, curr_data):
        """Calculate Cohen's d effect size"""
        ref_mean, curr_mean = ref_data.mean(), curr_data.mean()
        ref_std, curr_std = ref_data.std(), curr_data.std()
        
        # Pooled standard deviation
        n1, n2 = len(ref_data), len(curr_data)
        pooled_std = np.sqrt(((n1 - 1) * ref_std**2 + (n2 - 1) * curr_std**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0
        
        return (curr_mean - ref_mean) / pooled_std
    
    def _calculate_psi(self, ref_counts, curr_counts):
        """Calculate Population Stability Index"""
        ref_props = ref_counts / ref_counts.sum()
        curr_props = curr_counts / curr_counts.sum()
        
        # Avoid division by zero
        ref_props = ref_props.replace(0, 1e-7)
        curr_props = curr_props.replace(0, 1e-7)
        
        psi = sum((curr_props - ref_props) * np.log(curr_props / ref_props))
        return abs(psi)  # Return absolute value
    
    def _is_categorical(self, data):
        """Determine if feature is categorical"""
        unique_ratio = data.nunique() / len(data)
        return unique_ratio < 0.05 or data.dtype == 'object'
    
    def _determine_severity(self, effect_size, p_value):
        """Determine drift severity based on effect size and p-value (more sensitive)"""
        if abs(effect_size) > 0.5 or p_value < 0.001:
            return 'HIGH'
        elif abs(effect_size) > 0.3 or p_value < 0.01:
            return 'MEDIUM'
        elif abs(effect_size) > 0.1 or p_value < 0.05:
            return 'LOW'
        else:
            return 'NONE'
    
    def _assess_drift_severity(self, drift_metrics):
        """Assess overall drift severity with more sensitive alert levels"""
        
        drift_ratio = drift_metrics['share_drifted_features']
        high_severity_features = drift_metrics['high_severity_features']
        
        # More sensitive severity determination
        if drift_ratio >= 0.25 or high_severity_features >= 2:
            severity_level = 'RED'
            action = 'URGENT: Stop predictions, retrain model immediately'
        elif drift_ratio >= 0.10 or high_severity_features >= 1:
            severity_level = 'ORANGE'
            action = 'Investigate data pipeline, consider retraining'
        elif drift_ratio >= 0.05:
            severity_level = 'YELLOW'
            action = 'Increase monitoring frequency'
        else:
            severity_level = 'GREEN'
            action = 'Continue monitoring'
        
        return {
            'severity_level': severity_level,
            'recommended_action': action,
            'drift_ratio': drift_ratio
        }
    
    def _save_drift_results(self, drift_metrics, ref_data, curr_data):
        """Save comprehensive drift results"""
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Save detailed summary
        summary_path = os.path.join(self.results_dir, "drift_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Robust Data Drift Detection Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Reference data shape: {ref_data.shape}\n")
            f.write(f"Current data shape: {curr_data.shape}\n\n")
            
            f.write(f"Overall Drift Metrics:\n")
            f.write(f"- Number of drifted features: {drift_metrics['n_drifted_features']}\n")
            
            # Safe formatting for share of drifted features
            share_value = drift_metrics['share_drifted_features']
            if isinstance(share_value, (int, float)):
                f.write(f"- Share of drifted features: {share_value:.2%}\n")
            else:
                f.write(f"- Share of drifted features: {share_value}\n")
            f.write(f"- Dataset drift detected: {drift_metrics['dataset_drift']}\n")
            f.write(f"- Critical drift detected: {drift_metrics['critical_drift']}\n")
            f.write(f"- High severity features: {drift_metrics['high_severity_features']}\n")
            f.write(f"- Severity level: {drift_metrics['severity_level']}\n")
            f.write(f"- Recommended action: {drift_metrics['recommended_action']}\n\n")
            
            # Target distribution info
            if '_target_distribution' in drift_metrics['feature_details']:
                target_info = drift_metrics['feature_details']['_target_distribution']
                f.write(f"Target Distribution Analysis:\n")
                f.write(f"- JS Distance: {target_info.get('js_distance', 'N/A'):.6f}\n")
                f.write(f"- Target drift detected: {target_info.get('drift_detected', False)}\n")
                f.write(f"- Target severity: {target_info.get('severity', 'N/A')}\n\n")
            
            # Model performance info
            if '_model_performance' in drift_metrics['feature_details']:
                perf_info = drift_metrics['feature_details']['_model_performance']
                f.write(f"Model Performance Analysis:\n")
                f.write(f"- Current accuracy: {perf_info.get('current_accuracy', 'N/A'):.4f}\n")
                f.write(f"- Performance drop: {perf_info.get('performance_drop', 'N/A'):.4f}\n")
                f.write(f"- Performance drift detected: {perf_info.get('drift_detected', False)}\n")
                f.write(f"- Performance severity: {perf_info.get('severity', 'N/A')}\n\n")
            
            f.write(f"Feature-level Results:\n")
            f.write(f"{'='*30}\n")
            for feature, result in drift_metrics['feature_details'].items():
                f.write(f"\n{feature}:\n")
                f.write(f"  - Drift detected: {result['drift_detected']}\n")
                f.write(f"  - Severity: {result.get('severity', 'N/A')}\n")
                
                # Safe formatting for p_value
                p_value = result.get('p_value', 'N/A')
                if isinstance(p_value, (int, float)):
                    f.write(f"  - P-value: {p_value:.6f}\n")
                else:
                    f.write(f"  - P-value: {p_value}\n")
                    
                f.write(f"  - Method: {result.get('method', 'N/A')}\n")
                
                # Safe formatting for effect_size
                if 'effect_size' in result:
                    effect_size = result['effect_size']
                    if isinstance(effect_size, (int, float)):
                        f.write(f"  - Effect size (Cohen's d): {effect_size:.4f}\n")
                    else:
                        f.write(f"  - Effect size (Cohen's d): {effect_size}\n")
                        
                # Safe formatting for psi
                if 'psi' in result:
                    psi = result['psi']
                    if isinstance(psi, (int, float)):
                        f.write(f"  - PSI: {psi:.4f}\n")
                    else:
                        f.write(f"  - PSI: {psi}\n")
        
        # Save JSON results
        # Convert numpy types to Python types for JSON serialization
        json_metrics = self._convert_for_json(drift_metrics)
        with open(os.path.join(self.results_dir, "drift_results.json"), 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print("✓ Comprehensive drift summary saved.")
    
    def _convert_for_json(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, int, float, str)):
            return obj
        elif obj is None:
            return None
        else:
            # Try to convert to string as fallback
            return str(obj)

def main():
    detector = DataDriftDetector()
    drift_metrics = detector.detect_drift()
    test_results = detector.run_drift_tests()

    try:
        with mlflow.start_run(run_name="Data_Drift_Detection"):
            # Flatten the metrics to avoid nested dict errors
            drift_metrics_flat = {k: v for k, v in drift_metrics.items() if isinstance(v, (int, float, bool))}
            test_results_flat = {k: v for k, v in test_results.items() if isinstance(v, (int, float, bool))}
            
            mlflow.log_metrics(drift_metrics_flat)
            mlflow.log_metrics(test_results_flat)
            mlflow.log_artifact("Results/drift_results.json")
            mlflow.log_artifact("Results/drift_summary.txt")
        print("✓ Drift metrics logged to MLflow")
    except Exception as e:
        print(f"⚠️ Could not log to MLflow: {e}")

    print("\n" + "=" * 60)
    print("ROBUST DATA DRIFT DETECTION SUMMARY")
    print("=" * 60)
    print(f"Number of drifted features: {drift_metrics['n_drifted_features']}")
    
    # Safe formatting for share of drifted features
    share_value = drift_metrics['share_drifted_features']
    if isinstance(share_value, (int, float)):
        print(f"Share of drifted features: {share_value:.2%}")
    else:
        print(f"Share of drifted features: {share_value}")
        
    print(f"Dataset drift detected: {drift_metrics['dataset_drift']}")
    print(f"Critical drift detected: {drift_metrics['critical_drift']}")
    print(f"High severity features: {drift_metrics['high_severity_features']}")
    print(f"Severity level: {drift_metrics['severity_level']}")
    print(f"Recommended action: {drift_metrics['recommended_action']}")
    print(f"Drift tests passed: {test_results['passed_tests']}/{test_results['total_tests']}")
    print(f"Overall system health: {test_results['overall_health']}")
    print("=" * 60)

    return drift_metrics, test_results



if __name__ == "__main__":
    drift_metrics, test_results = main()
