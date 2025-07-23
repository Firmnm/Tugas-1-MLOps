import pandas as pd
import numpy as np
import os
import json
import warnings
import time
import argparse
from datetime import datetime
from scipy import stats

# Try to import SDV for CTGAN, fallback if not available
try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    CTGAN_AVAILABLE = True
except ImportError:
    CTGAN_AVAILABLE = False
    print("SDV not available, will use statistical sampling")

warnings.filterwarnings("ignore")


class adaptiveDriftGenerator:
    def __init__(self, 
                 original_data_path="Data/personality_datasert.csv",
                 output_path="Data/synthetic_ctgan_data.csv",
                 config_path="adaptive_drift_config.json"):

        self.original_data_path = original_data_path
        self.output_path = output_path
        self.config_path = config_path
        
        # Adaptive drift parameters
        self.base_drift_strength = 0.1
        self.max_drift_strength = 0.8
        self.escalation_factor = 1.2
        
        # Load or initialize configuration
        self.config = self._load_or_create_config()
        
        # Data containers
        self.original_data = None
        self.current_data = None
        
        # Define feature categories for targeted drift
        self.numerical_features = [
            'Time_spent_Alone', 'Social_event_attendance', 
            'Going_outside', 'Friends_circle_size', 'Post_frequency'
        ]
        self.categorical_features = ['Stage_fear', 'Drained_after_socializing']
        self.target_feature = 'Personality'
        
    def _load_or_create_config(self):
        """Load existing configuration or create new one"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                print(f" Loaded existing configuration from {self.config_path}")
        else:
            config = {
                'drift_cycle': 0,
                'drift_history': [],
                'current_drift_strength': self.base_drift_strength,
                'last_detected_drift': None,
                'failed_detections': 0,
                'strategy_adaptation_count': 0,
                'baseline_stats': None,
                'created_at': datetime.now().isoformat()
            }
            print("Created new adaptive drift configuration")
        
        return config
    
    def _save_config(self):
        """Save current configuration"""
        self.config['updated_at'] = datetime.now().isoformat()
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_data(self):
        """Load original and current data"""
        print(" Loading datasets...")
        
        # Load original data
        self.original_data = pd.read_csv(self.original_data_path)
        print(f" Original data loaded: {self.original_data.shape}")
        
        # Load current data if exists, otherwise use original
        if os.path.exists(self.output_path):
            self.current_data = pd.read_csv(self.output_path)
            print(f" Current synthetic data loaded: {self.current_data.shape}")
        else:
            self.current_data = self.original_data.copy()
            print(" No existing synthetic data, using original as baseline")
        
        # Store baseline statistics if not exists
        if self.config['baseline_stats'] is None:
            self.config['baseline_stats'] = self._calculate_baseline_stats()
            print(" Baseline statistics calculated and stored")
    
    def _calculate_baseline_stats(self):
        """Calculate baseline statistics for adaptive comparison"""
        stats_dict = {}
        
        for col in self.numerical_features:
            if col in self.original_data.columns:
                stats_dict[col] = {
                    'mean': float(self.original_data[col].mean()),
                    'std': float(self.original_data[col].std()),
                    'min': float(self.original_data[col].min()),
                    'max': float(self.original_data[col].max())
                }
        
        for col in self.categorical_features:
            if col in self.original_data.columns:
                stats_dict[col] = {
                    'value_counts': self.original_data[col].value_counts().to_dict()
                }
        
        # Target distribution
        if self.target_feature in self.original_data.columns:
            stats_dict[self.target_feature] = {
                'value_counts': self.original_data[self.target_feature].value_counts().to_dict()
            }
        
        return stats_dict
    
    def _detect_current_drift_level(self):
        """Analyze current data to determine if drift was successfully detected"""
        if len(self.config['drift_history']) < 2:
            return 'unknown'
        
        # Simple heuristic: if data size has grown significantly since last cycle,
        # assume drift was detected and new data was added
        last_size = self.config['drift_history'][-1].get('data_size', 0)
        current_size = len(self.current_data)
        
        growth_rate = (current_size - last_size) / last_size if last_size > 0 else 0
        
        if growth_rate > 0.1:  # 10% growth indicates successful drift detection
            return 'detected'
        else:
            return 'not_detected'
    
    def _adapt_drift_strategy(self):
        """Adapt drift generation strategy based on historical performance"""
        drift_level = self._detect_current_drift_level()
        
        if drift_level == 'not_detected':
            # Increase drift strength if previous drift wasn't detected
            self.config['failed_detections'] += 1
            old_strength = self.config['current_drift_strength']
            self.config['current_drift_strength'] = min(
                self.config['current_drift_strength'] * self.escalation_factor,
                self.max_drift_strength
            )
            print(f" Escalating drift strength: {old_strength:.3f} → {self.config['current_drift_strength']:.3f}")
            
        elif drift_level == 'detected':
            # Reset failed detections counter on successful detection
            self.config['failed_detections'] = 0
            self.config['last_detected_drift'] = datetime.now().isoformat()
            print(" Previous drift was detected, maintaining current strategy")
        
        self.config['strategy_adaptation_count'] += 1
    
    def generate_base_synthetic_data(self, n_samples=1000):
        """Generate high-quality synthetic data using CTGAN or statistical sampling"""
        print(f" Generating {n_samples} base synthetic samples...")
        
        if CTGAN_AVAILABLE:
            try:
                print("   Using CTGAN for high-quality synthesis...")
                # Prepare metadata
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(self.original_data)
                
                # Initialize and train CTGAN
                synthesizer = CTGANSynthesizer(metadata, epochs=100, verbose=False)
                synthesizer.fit(self.original_data)
                
                # Generate synthetic data
                synthetic_data = synthesizer.sample(num_rows=n_samples)
                
                # Post-process to ensure correct data types
                synthetic_data = self._post_process_synthetic_data(synthetic_data)
                
                print(" CTGAN synthetic data generated successfully")
                return synthetic_data
                
            except Exception as e:
                print(f" Error in CTGAN generation: {e}")
                print(" Falling back to statistical sampling...")
        
        # Fallback to statistical sampling
        return self._statistical_synthetic_generation(n_samples)
    
    def _statistical_synthetic_generation(self, n_samples):
        """Generate synthetic data using statistical sampling"""
        print(" Using statistical sampling...")
        
        # Bootstrap sampling with noise
        synthetic_data = self.original_data.sample(n=n_samples, replace=True).copy()
        
        # Add small amount of noise to numerical features
        for col in self.numerical_features:
            if col in synthetic_data.columns:
                noise_std = synthetic_data[col].std() * 0.1  # 10% noise
                noise = np.random.normal(0, noise_std, size=len(synthetic_data))
                synthetic_data[col] = synthetic_data[col] + noise
                
                # Keep within reasonable bounds
                if col in self.config['baseline_stats']:
                    min_val = max(0, self.config['baseline_stats'][col]['min'])
                    max_val = self.config['baseline_stats'][col]['max']
                    synthetic_data[col] = synthetic_data[col].clip(min_val, max_val).round()
        
        return synthetic_data
    
    def _post_process_synthetic_data(self, synthetic_data):
        """Post-process synthetic data to ensure correct data types and ranges"""
        processed_data = synthetic_data.copy()
        
        # Fix numerical features
        for col in self.numerical_features:
            if col in processed_data.columns:
                # Round to integers where appropriate
                processed_data[col] = processed_data[col].round()
                
                # Clip to reasonable ranges based on original data
                if col in self.config['baseline_stats']:
                    min_val = max(0, self.config['baseline_stats'][col]['min'])
                    max_val = self.config['baseline_stats'][col]['max']
                    processed_data[col] = processed_data[col].clip(min_val, max_val)
        
        # Fix categorical features
        for col in self.categorical_features:
            if col in processed_data.columns:
                # Ensure binary categorical values
                processed_data[col] = processed_data[col].round().clip(0, 1).astype(int)
        
        # Fix target feature
        if self.target_feature in processed_data.columns:
            # Ensure only valid personality types
            valid_personalities = ['Introvert', 'Extrovert']
            mask = processed_data[self.target_feature].isin(valid_personalities)
            if not mask.all():
                # Replace invalid values with random valid ones
                invalid_indices = processed_data[~mask].index
                processed_data.loc[invalid_indices, self.target_feature] = np.random.choice(
                    valid_personalities, size=len(invalid_indices)
                )
        
        return processed_data
    
    def apply_adaptive_drift(self, data):
        """Apply adaptive drift to data based on current configuration"""
        print(f" Applying adaptive drift (strength: {self.config['current_drift_strength']:.3f})")
        
        drifted_data = data.copy()
        cycle = self.config['drift_cycle']
        strength = self.config['current_drift_strength']
        
        # Determine drift type based on cycle
        drift_types = ['input_drift', 'label_drift', 'concept_drift', 'combined_drift']
        drift_type = drift_types[cycle % len(drift_types)]
        
        print(f" Applying {drift_type} (cycle {cycle})")
        
        if drift_type == 'input_drift':
            drifted_data = self._apply_input_drift(drifted_data, strength)
        elif drift_type == 'label_drift':
            drifted_data = self._apply_label_drift(drifted_data, strength)
        elif drift_type == 'concept_drift':
            drifted_data = self._apply_concept_drift(drifted_data, strength)
        else:  # combined_drift
            drifted_data = self._apply_combined_drift(drifted_data, strength)
        
        return drifted_data, drift_type
    
    def _apply_input_drift(self, data, strength):
        """Apply input feature distribution drift"""
        drifted_data = data.copy()
        
        # Apply shifts to numerical features
        shifts = {
            'Time_spent_Alone': strength * 1.5,
            'Social_event_attendance': -strength * 1.2,
            'Going_outside': strength * 0.8,
            'Friends_circle_size': strength * 1.0,
            'Post_frequency': strength * 0.6
        }
        
        for col, shift in shifts.items():
            if col in drifted_data.columns:
                # Add systematic shift with noise
                noise = np.random.normal(shift, strength * 0.3, len(drifted_data))
                drifted_data[col] = drifted_data[col] + noise
                
                # Keep within valid bounds
                if col in self.config['baseline_stats']:
                    min_val = 0
                    max_val = self.config['baseline_stats'][col]['max'] * 1.2
                    drifted_data[col] = drifted_data[col].clip(min_val, max_val).round()
        
        return drifted_data
    
    def _apply_label_drift(self, data, strength):
        """Apply label distribution drift"""
        drifted_data = data.copy()
        
        # Flip some labels based on strength
        introvert_mask = drifted_data[self.target_feature] == 'Introvert'
        n_introverts = introvert_mask.sum()
        
        if n_introverts > 0:
            flip_rate = min(strength * 0.3, 0.4)  # Up to 40% flip rate
            n_to_flip = int(n_introverts * flip_rate)
            
            if n_to_flip > 0:
                introvert_indices = drifted_data[introvert_mask].index
                flip_indices = np.random.choice(introvert_indices, size=n_to_flip, replace=False)
                drifted_data.loc[flip_indices, self.target_feature] = 'Extrovert'
                print(f"  Flipped {n_to_flip}/{n_introverts} Introvert labels to Extrovert")
        
        return drifted_data
    
    def _apply_concept_drift(self, data, strength):
        """Apply concept drift - change feature-label relationships"""
        drifted_data = data.copy()
        
        # Make introverts more socially active (contradicting typical patterns)
        introvert_mask = drifted_data[self.target_feature] == 'Introvert'
        
        if introvert_mask.any():
            # Increase social event attendance for introverts
            noise = np.random.normal(strength * 3, strength, introvert_mask.sum())
            drifted_data.loc[introvert_mask, 'Social_event_attendance'] += noise
            drifted_data.loc[introvert_mask, 'Social_event_attendance'] = \
                drifted_data.loc[introvert_mask, 'Social_event_attendance'].clip(0, 11).round()
            
            # Decrease time spent alone for introverts
            noise = np.random.normal(-strength * 2.5, strength * 0.5, introvert_mask.sum())
            drifted_data.loc[introvert_mask, 'Time_spent_Alone'] += noise
            drifted_data.loc[introvert_mask, 'Time_spent_Alone'] = \
                drifted_data.loc[introvert_mask, 'Time_spent_Alone'].clip(0, 11).round()
        
        return drifted_data
    
    def _apply_combined_drift(self, data, strength):
        """Apply multiple types of drift simultaneously for maximum impact"""
        drifted_data = data.copy()
        
        # Apply all drift types with reduced individual strength
        reduced_strength = strength * 0.6
        
        drifted_data = self._apply_input_drift(drifted_data, reduced_strength)
        drifted_data = self._apply_label_drift(drifted_data, reduced_strength)
        drifted_data = self._apply_concept_drift(drifted_data, reduced_strength)
        
        print(f"   Applied combined drift with reduced strength: {reduced_strength:.3f}")
        
        return drifted_data
    
    def generate_drifted_data(self, n_new_samples=200):
        """Main method to generate new data with adaptive drift"""
        print(f"\n Adaptive Drift Generator - Cycle {self.config['drift_cycle']}")
        print(f" Current drift strength: {self.config['current_drift_strength']:.3f}")
        print(f" Failed detections: {self.config['failed_detections']}")
        
        # Load data
        self.load_data()
        
        # Adapt strategy based on previous performance
        if self.config['drift_cycle'] > 0:
            self._adapt_drift_strategy()
        
        # Generate base synthetic data
        if self.config['drift_cycle'] == 0:
            # First cycle: generate more base data
            base_data = self.generate_base_synthetic_data()
        else:
            # Subsequent cycles: generate smaller amounts
            base_data = self.generate_base_synthetic_data(n_samples=n_new_samples)
        
        # Apply adaptive drift
        drifted_data, drift_type = self.apply_adaptive_drift(base_data)
        
        # Combine with existing data (if not first cycle)
        if self.config['drift_cycle'] > 0 and len(self.current_data) > 0:
            # Keep a rolling window to prevent data from growing too large
            max_data_size = 4000
            if len(self.current_data) > max_data_size - len(drifted_data):
                # Remove oldest data
                keep_size = max_data_size - len(drifted_data)
                self.current_data = self.current_data.tail(keep_size)
            
            combined_data = pd.concat([self.current_data, drifted_data], ignore_index=True)
        else:
            combined_data = drifted_data
        
        # Save the new dataset
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        combined_data.to_csv(self.output_path, index=False)
        
        # Update configuration
        drift_info = {
            'cycle': self.config['drift_cycle'],
            'timestamp': datetime.now().isoformat(),
            'drift_type': drift_type,
            'drift_strength': self.config['current_drift_strength'],
            'new_samples': len(drifted_data),
            'total_size': len(combined_data),
            'data_size': len(combined_data)
        }
        
        self.config['drift_history'].append(drift_info)
        self.config['drift_cycle'] += 1
        self._save_config()
        
        # Print summary
        print(f"\n Drift generation completed!")
        print(f" Generated {len(drifted_data)} new samples with {drift_type}")
        print(f" Total dataset size: {len(combined_data)} rows")
        print(f" Saved to: {self.output_path}")
        print(f" Configuration saved to: {self.config_path}")
        
        return combined_data, drift_info
    
    def reset_baseline(self):
        """Reset to original baseline data"""
        print(" Resetting to baseline data...")
        
        # Copy original data to output path
        baseline_data = pd.read_csv(self.original_data_path)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        baseline_data.to_csv(self.output_path, index=False)
        
        # Reset configuration
        self.config = {
            'drift_cycle': 0,
            'drift_history': [],
            'current_drift_strength': self.base_drift_strength,
            'last_detected_drift': None,
            'failed_detections': 0,
            'strategy_adaptation_count': 0,
            'baseline_stats': None,
            'created_at': datetime.now().isoformat()
        }
        
        # Recalculate baseline stats
        self.original_data = baseline_data
        self.config['baseline_stats'] = self._calculate_baseline_stats()
        self._save_config()
        
        print(" Reset complete - ready for new drift generation cycle")
    
    def get_status(self):
        """Get current status and statistics"""
        status = {
            'cycle': self.config['drift_cycle'],
            'drift_strength': self.config['current_drift_strength'],
            'failed_detections': self.config['failed_detections'],
            'total_adaptations': self.config['strategy_adaptation_count'],
            'last_drift_detected': self.config.get('last_detected_drift'),
            'data_exists': os.path.exists(self.output_path)
        }
        
        if os.path.exists(self.output_path):
            data = pd.read_csv(self.output_path)
            status['current_data_size'] = len(data)
            if self.target_feature in data.columns:
                status['personality_distribution'] = data[self.target_feature].value_counts().to_dict()
        
        return status


def run_continuous_generation(generator, interval_minutes=10, max_cycles=None):
    """Run drift generation continuously with specified interval"""
    print(" Starting Continuous Adaptive Drift Generation")
    print(f" Interval: {interval_minutes} minutes")
    print(f" Max cycles: {max_cycles if max_cycles else 'Infinite'}")
    print("Press Ctrl+C to stop gracefully\n")
    
    cycle_count = 0
    
    try:
        while True:
            cycle_count += 1
            print(f"\n{'='*60}")
            print(f" Starting Generation Cycle {cycle_count}")
            print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            
            try:
                data, drift_info = generator.generate_drifted_data()
                
                print(f"\n Cycle {cycle_count} completed successfully")
                print(f" Generated {drift_info['new_samples']} new samples")
                print(f" Total dataset size: {drift_info['total_size']}")
                print(f" Drift type: {drift_info['drift_type']}")
                print(f" Drift strength: {drift_info['drift_strength']:.3f}")
                
                # Check if we've reached max cycles
                if max_cycles and cycle_count >= max_cycles:
                    print(f"\n Reached maximum cycles ({max_cycles}). Stopping.")
                    break
                
                # Wait for next cycle
                print(f"\n Waiting {interval_minutes} minutes until next cycle...")
                print(f"   Next cycle at: {datetime.fromtimestamp(time.time() + interval_minutes * 60).strftime('%H:%M:%S')}")
                
                time.sleep(interval_minutes * 60)
                
            except Exception as e:
                print(f" Error in cycle {cycle_count}: {e}")
                print(" Continuing to next cycle in 1 minute...")
                time.sleep(60)
                
    except KeyboardInterrupt:
        print(f"\n\n Gracefully stopping after {cycle_count} cycles")
        print(" All generated data has been saved")
        print(" Drift generation stopped")


def show_status(generator):
    """Show current generator status"""
    try:
        status = generator.get_status()
        
        print("\n Unified Adaptive Drift Generator Status Report")
        print("="*60)
        print(f" Current Cycle: {status['cycle']}")
        print(f" Drift Strength: {status['drift_strength']:.3f}")
        print(f" Failed Detections: {status['failed_detections']}")
        print(f" Total Adaptations: {status['total_adaptations']}")
        print(f" Last Drift Detected: {status.get('last_drift_detected', 'Never')}")
        print(f" Current Data Size: {status.get('current_data_size', 'N/A')}")
        
        if 'personality_distribution' in status:
            print(f"\n Personality Distribution:")
            for personality, count in status['personality_distribution'].items():
                percentage = (count / status['current_data_size']) * 100
                print(f"   {personality}: {count} ({percentage:.1f}%)")
        
        print(f"\n Data File Exists: {'' if status['data_exists'] else 'not found'}")
        print(f" CTGAN Available: {'' if CTGAN_AVAILABLE else ' (using statistical sampling)'}")
        
        # Show drift history if available
        try:
            if os.path.exists(generator.config_path):
                with open(generator.config_path, 'r') as f:
                    config = json.load(f)
                
                if config.get('drift_history'):
                    print(f"\n Recent Drift History (last 5):")
                    recent_history = config['drift_history'][-5:]
                    for entry in recent_history:
                        timestamp = datetime.fromisoformat(entry['timestamp']).strftime('%m-%d %H:%M')
                        print(f"   Cycle {entry['cycle']}: {entry['drift_type']} "
                              f"(strength: {entry['drift_strength']:.3f}, "
                              f"samples: {entry['new_samples']}) - {timestamp}")
        except Exception as e:
            print(f" Could not load drift history: {e}")
        
        print("="*60)
        
    except Exception as e:
        print(f" Error getting status: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified Adaptive Drift Generator - Single File Solution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_synthetic_generator.py --single                  # Generate one cycle
  python data_synthetic_generator.py --continuous              # Run continuously every 10 minutes
  python data_synthetic_generator.py --continuous --interval 5 # Run every 5 minutes
  python data_synthetic_generator.py --status                  # Show current status
  python data_synthetic_generator.py --reset                   # Reset to baseline
  python data_synthetic_generator.py --continuous --max-cycles 5 # Run max 5 cycles
"""
    )
    
    parser.add_argument('--single', action='store_true',
                       help='Generate single drift cycle')
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuous drift generation')
    parser.add_argument('--status', action='store_true',
                       help='Show current status')
    parser.add_argument('--reset', action='store_true',
                       help='Reset generator to baseline')
    parser.add_argument('--interval', type=int, default=10,
                       help='Interval between generations in minutes (default: 10)')
    parser.add_argument('--max-cycles', type=int,
                       help='Maximum number of cycles for continuous mode')
    parser.add_argument('--samples', type=int, default=200,
                       help='Number of new samples to generate per cycle (default: 200)')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = adaptiveDriftGenerator()
    
    if args.single:
        print(" Generating single drift cycle...")
        try:
            data, drift_info = generator.generate_drifted_data(n_new_samples=args.samples)
            print(" Single cycle completed successfully")
            print(f" Generated {drift_info['new_samples']} new samples")
            print(f" Total dataset size: {drift_info['total_size']}")
            print(f" Drift type: {drift_info['drift_type']}")
        except Exception as e:
            print(f" Error generating single cycle: {e}")
            
    elif args.continuous:
        run_continuous_generation(generator, args.interval, args.max_cycles)
        print("\nPress Ctrl+C to stop monitoring...")
        
        
        
    elif args.reset:
        print(" Resetting Unified Adaptive Drift Generator...")
        try:
            generator.reset_baseline()
            print(" Generator reset to baseline successfully")
        except Exception as e:
            print(f" Error resetting generator: {e}")
            
    elif args.status:
        show_status(generator)
        
    else:
        # Default: show status and help
        show_status(generator)
        print(f"\n Use --help to see all available options")
        print(f" Quick start: python {os.path.basename(__file__)} --single")


if __name__ == "__main__":
    main()
