#!/usr/bin/env python3
"""
Silhouette Score Analyzer for Quantum Text Classification
Saves comprehensive silhouette analysis data for later analysis
"""

import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

class SilhouetteAnalyzer:
    def __init__(self, save_dir: str = "silhouette_analysis", 
                 dynamic_naming: bool = False, 
                 quantum_params: Optional[Dict[str, Any]] = None):
        """
        Initialize Silhouette Analyzer
        
        Args:
            save_dir: Directory to save silhouette analysis data
            dynamic_naming: Whether to use dynamic naming for files
            quantum_params: Dictionary with quantum parameters for dynamic naming
        """
        self.save_dir = save_dir
        self.dynamic_naming = dynamic_naming
        self.quantum_params = quantum_params or {}
        
        # Create directories
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "raw_data"), exist_ok=True)
        
        # Initialize data storage
        self.analysis_history = []
        
    def get_dynamic_filename(self, base_name: str) -> str:
        """Generate dynamic filename based on quantum parameters"""
        if not self.dynamic_naming or not self.quantum_params:
            return base_name
        
        # Extract file extension
        name_parts = base_name.split('.')
        if len(name_parts) > 1:
            extension = name_parts[-1]
            name_without_ext = '.'.join(name_parts[:-1])
        else:
            extension = ''
            name_without_ext = base_name
        
        # Get quantum parameters for dynamic naming
        feature_dim = self.quantum_params.get('feature_dim', 'unknown')
        shots = self.quantum_params.get('shots', 'unknown')
        train_samples = self.quantum_params.get('train_samples', 'unknown')
        val_samples = self.quantum_params.get('val_samples', 'unknown')
        
        # Generate dynamic filename: {name}_{feature_dim}-{shots}-{train_samples}-{val_samples}.{ext}
        dynamic_name = f"{name_without_ext}_{feature_dim}-{shots}-{train_samples}-{val_samples}"
        
        if extension:
            return f"{dynamic_name}.{extension}"
        else:
            return dynamic_name

    def calculate_silhouette_scores(self, X: np.ndarray, y_true: np.ndarray, 
                                  data_type: str = "unknown", 
                                  experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive silhouette scores for given data
        
        Args:
            X: Feature matrix (2D or t-SNE reduced)
            y_true: True labels
            data_type: Type of data (e.g., "original_features", "quantum_predictions")
            experiment_id: Unique experiment identifier
            
        Returns:
            Dictionary with comprehensive silhouette analysis
        """
        if len(np.unique(y_true)) < 2:
            print(f"⚠️ Silhouette score requires at least 2 classes. Skipping for {data_type}.")
            return {}

        if len(X) <= 1:
            print(f"⚠️ Silhouette score requires more than 1 sample. Skipping for {data_type}.")
            return {}

        # Calculate sklearn's silhouette score
        sklearn_score = silhouette_score(X, y_true)
        sklearn_samples = silhouette_samples(X, y_true)

        # Basic statistics for individual sample scores
        stats = {
            "mean": float(np.mean(sklearn_samples)),
            "std": float(np.std(sklearn_samples)),
            "min": float(np.min(sklearn_samples)),
            "max": float(np.max(sklearn_samples)),
            "median": float(np.median(sklearn_samples)),
            "q25": float(np.percentile(sklearn_samples, 25)),
            "q75": float(np.percentile(sklearn_samples, 75))
        }

        # Generate experiment ID if not provided
        if experiment_id is None:
            experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get dynamic filename if dynamic_naming is True
        if self.dynamic_naming:
            file_name = self.get_dynamic_filename(f"{experiment_id}_{data_type.replace(' ', '_')}_silhouette.json")
        else:
            file_name = f"{experiment_id}_{data_type.replace(' ', '_')}_silhouette.json"
            
        file_path = os.path.join(self.save_dir, "raw_data", file_name)

        analysis_data = {
            "experiment_id": experiment_id,
            "data_type": data_type,
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y_true)),
            "sklearn_silhouette_score": float(sklearn_score),
            "sklearn_silhouette_samples": sklearn_samples.tolist(), # Convert to list for JSON
            "silhouette_statistics": stats,
            # Enhanced raw data for analysis
            "raw_data": {
                "feature_matrix": X.tolist(),  # Original feature matrix (t-SNE reduced)
                "true_labels": y_true.tolist(),  # True class labels
                "unique_labels": np.unique(y_true).tolist(),  # Available classes
                "feature_names": [f"feature_{i}" for i in range(X.shape[1])],  # Feature names
                "data_summary": {
                    "feature_ranges": {
                        "min": X.min(axis=0).tolist(),
                        "max": X.max(axis=0).tolist(),
                        "mean": X.mean(axis=0).tolist(),
                        "std": X.std(axis=0).tolist()
                    },
                    "class_distribution": {
                        str(label): int(np.sum(y_true == label)) for label in np.unique(y_true)
                    }
                }
            }
        }

        with open(file_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        print(f"✅ Silhouette analysis saved: {file_path}")
        
        # Save raw data as CSV for easier analysis
        self._save_raw_data_csv(X, y_true, sklearn_samples, data_type, experiment_id)
        
        return analysis_data
    
    def _save_raw_data_csv(self, X: np.ndarray, y_true: np.ndarray, 
                          silhouette_samples: np.ndarray, data_type: str, 
                          experiment_id: str):
        """Save raw silhouette data as CSV for easier analysis"""
        try:
            import pandas as pd
            
            # Create DataFrame with all data
            df_data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            df_data['true_label'] = y_true
            df_data['silhouette_score'] = silhouette_samples
            
            # Generate filename
            if self.dynamic_naming:
                csv_filename = self.get_dynamic_filename(f"{experiment_id}_{data_type.replace(' ', '_')}_silhouette_raw.csv")
            else:
                csv_filename = f"{experiment_id}_{data_type.replace(' ', '_')}_silhouette_raw.csv"
            
            csv_path = os.path.join(self.save_dir, "raw_data", csv_filename)
            df_data.to_csv(csv_path, index=False)
            print(f"✅ Raw silhouette data saved as CSV: {csv_path}")
            
        except ImportError:
            print("⚠️ pandas not available, skipping CSV export")
        except Exception as e:
            print(f"⚠️ Failed to save CSV: {e}")
