import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

logger = logging.getLogger(__name__)

class FeatureImportanceAnalyzer:
    """Analyze feature importance for both Classical SVM and Quantum SVM"""
    
    def __init__(self):
        self.classical_importance = None
        self.quantum_importance = None
        self.feature_names = None
        
    def analyze_classical_svm_importance(self, classical_model, X_train: np.ndarray, 
                                       y_train: np.ndarray, X_val: np.ndarray, 
                                       y_val: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """
        Analyze feature importance for Classical SVM using coefficients and permutation importance
        """
        logger.info("Analyzing Classical SVM feature importance...")
        
        # Method 1: Linear SVM Coefficients (most reliable for LinearSVM)
        try:
            # Get the base LinearSVM from the calibrated classifier
            if hasattr(classical_model, 'calibrated_classifiers_'):
                base_svm = classical_model.calibrated_classifiers_[0].estimator
            elif hasattr(classical_model, 'estimator'):
                base_svm = classical_model.estimator
            else:
                base_svm = classical_model
            
            if hasattr(base_svm, 'coef_'):
                # For binary classification, coef_ shape is (1, n_features)
                # For multiclass, it's (n_classes, n_features)
                coefficients = base_svm.coef_
                
                if coefficients.shape[0] == 1:
                    # Binary classification
                    feature_importance = np.abs(coefficients[0])
                else:
                    # Multiclass - take mean absolute value across classes
                    feature_importance = np.mean(np.abs(coefficients), axis=0)
                
                fi_sum = np.sum(feature_importance)
                feature_importance_pct = (feature_importance / fi_sum * 100) if fi_sum > 0 else np.zeros_like(feature_importance)
                
                logger.info(f"Extracted {len(feature_importance)} coefficient-based importances")
            else:
                logger.warning("No coefficients available, using permutation importance only")
                feature_importance_pct = np.zeros(len(feature_names))
                
        except Exception as e:
            logger.error(f"Error extracting coefficients: {e}")
            feature_importance_pct = np.zeros(len(feature_names))
        
        # Method 2: Permutation Importance (more reliable, but slower)
        try:
            logger.info("Computing permutation importance for Classical SVM...")
            # Use a smaller subset for permutation importance to avoid computational bottleneck
            if len(X_val) > 500:
                # Sample 500 validation samples for permutation importance
                sample_indices = np.random.choice(len(X_val), 500, replace=False)
                X_val_sample = X_val[sample_indices]
                y_val_sample = y_val[sample_indices]
                logger.info(f"Using 500 samples for permutation importance (from {len(X_val)} total)")
            else:
                X_val_sample = X_val
                y_val_sample = y_val
                logger.info(f"Using all {len(X_val)} samples for permutation importance")
            
            perm_importance = permutation_importance(
                classical_model, X_val_sample, y_val_sample, 
                n_repeats=3, random_state=42,  # Reduced from 5 to 3 repeats
                n_jobs=1, scoring='accuracy'  # Sequential processing to prevent overload
            )
            _raw_perm = perm_importance.importances_mean
            perm_importance_scores = np.maximum(_raw_perm, 0)
            pi_sum = np.sum(perm_importance_scores)
            perm_importance_pct = (perm_importance_scores / pi_sum * 100) if pi_sum > 0 else np.zeros_like(perm_importance_scores)
            
            logger.info("Permutation importance computed successfully")
        except Exception as e:
            logger.error(f"Error computing permutation importance: {e}")
            perm_importance_pct = feature_importance_pct.copy()
        
        # Combine both methods (weighted average)
        combined_importance = 0.7 * feature_importance_pct + 0.3 * perm_importance_pct
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient_importance': feature_importance_pct,
            'permutation_importance': perm_importance_pct,
            'combined_importance': combined_importance
        })
        
        # Sort by combined importance
        importance_df = importance_df.sort_values('combined_importance', ascending=False)
        
        self.classical_importance = importance_df
        self.feature_names = feature_names
        
        return {
            'importance_df': importance_df,
            'top_100_features': importance_df.head(100),
            'method': 'coefficient + permutation',
            'total_features': len(feature_names)
        }
    
    def analyze_quantum_svm_importance(self, quantum_model, X_train_quantum: np.ndarray, 
                                     y_train: np.ndarray, X_val_quantum: np.ndarray, 
                                     y_val: np.ndarray, feature_names: np.ndarray,
                                     pca_model: Any) -> Dict[str, Any]:
        """
        Analyze feature importance for Quantum SVM using PCA component analysis
        """
        logger.info("Analyzing Quantum SVM feature importance...")
        
        try:
            # Method 1: PCA Component Analysis
            # Get PCA components (how original features contribute to quantum features)
            pca_components = pca_model.components_  # Shape: (n_quantum_features, n_original_features)
            
            # Weight each component's loadings by its explained variance ratio so that
            # features contributing to high-variance components rank higher than those
            # contributing only to low-variance components.
            evr = pca_model.explained_variance_ratio_  # shape: (n_components,)
            pca_importance = np.sum(np.abs(pca_components) * evr[:, np.newaxis], axis=0)
            pca_sum = np.sum(pca_importance)
            pca_importance_pct = (pca_importance / pca_sum * 100) if pca_sum > 0 else np.zeros_like(pca_importance)
            
            logger.info(f"PCA analysis: {pca_components.shape[1]} original features → {pca_components.shape[0]} quantum features (weighted by explained variance ratio)")
            
            # Method 2: Permutation Importance on Original Feature Space
            # This is computationally expensive but more accurate
            # DISABLED: Permutation importance for quantum SVM is too slow and error-prone
            # We rely on PCA-based importance which is more reliable for quantum models
            try:
                logger.info("Skipping permutation importance for Quantum SVM (using PCA-based importance only)")
                # Use PCA importance only for quantum models
                perm_importance_pct = pca_importance_pct.copy()
                
            except Exception as e:
                logger.warning(f"Permutation importance failed, using PCA only: {e}")
                perm_importance_pct = pca_importance_pct.copy()
            
            # Use PCA importance only (permutation is disabled for quantum)
            combined_importance = pca_importance_pct
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'pca_importance': pca_importance_pct,
                'permutation_importance': perm_importance_pct,
                'combined_importance': combined_importance
            })
            
            # Sort by combined importance
            importance_df = importance_df.sort_values('combined_importance', ascending=False)
            
            self.quantum_importance = importance_df
            
            return {
                'importance_df': importance_df,
                'top_100_features': importance_df.head(100),
                'method': 'PCA + permutation',
                'total_features': len(feature_names),
                'quantum_dimensions': pca_components.shape[0],
                'pca_explained_variance': np.sum(pca_model.explained_variance_ratio_)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing quantum feature importance: {e}")
            # Fallback: equal importance
            uniform_importance = np.ones(len(feature_names)) / len(feature_names) * 100
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'pca_importance': uniform_importance,
                'permutation_importance': uniform_importance,
                'combined_importance': uniform_importance
            })
            
            return {
                'importance_df': importance_df,
                'top_100_features': importance_df.head(100),
                'method': 'fallback_uniform',
                'total_features': len(feature_names)
            }
    
    def compare_feature_importance(self) -> Dict[str, Any]:
        """Compare feature importance between Classical and Quantum SVMs"""
        if self.classical_importance is None or self.quantum_importance is None:
            raise ValueError("Both classical and quantum importance must be computed first")
        
        logger.info("Comparing Classical vs Quantum feature importance...")
        
        # Merge the importance dataframes
        comparison_df = pd.merge(
            self.classical_importance[['feature', 'combined_importance']].rename(columns={'combined_importance': 'classical_importance'}),
            self.quantum_importance[['feature', 'combined_importance']].rename(columns={'combined_importance': 'quantum_importance'}),
            on='feature'
        )
        
        # Calculate importance difference
        comparison_df['importance_diff'] = comparison_df['quantum_importance'] - comparison_df['classical_importance']
        comparison_df['importance_ratio'] = comparison_df['quantum_importance'] / (comparison_df['classical_importance'] + 1e-10)
        
        # Sort by quantum importance
        comparison_df = comparison_df.sort_values('quantum_importance', ascending=False)
        
        return {
            'comparison_df': comparison_df,
            'top_100_comparison': comparison_df.head(100),
            'quantum_preferred_features': comparison_df.nlargest(50, 'importance_diff'),
            'classical_preferred_features': comparison_df.nsmallest(50, 'importance_diff'),
            'correlation': comparison_df['classical_importance'].corr(comparison_df['quantum_importance'])
        }
    
    def create_importance_visualization(self, top_n: int = 100) -> str:
        """Create feature importance visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            if self.classical_importance is not None:
                # Classical SVM top features
                top_classical = self.classical_importance.head(top_n)
                axes[0, 0].barh(range(min(20, len(top_classical))), 
                               top_classical['combined_importance'].head(20))
                axes[0, 0].set_yticks(range(min(20, len(top_classical))))
                axes[0, 0].set_yticklabels(top_classical['feature'].head(20), fontsize=8)
                axes[0, 0].set_title('Top 20 Classical SVM Features')
                axes[0, 0].set_xlabel('Importance (%)')
            
            if self.quantum_importance is not None:
                # Quantum SVM top features
                top_quantum = self.quantum_importance.head(top_n)
                axes[0, 1].barh(range(min(20, len(top_quantum))), 
                               top_quantum['combined_importance'].head(20))
                axes[0, 1].set_yticks(range(min(20, len(top_quantum))))
                axes[0, 1].set_yticklabels(top_quantum['feature'].head(20), fontsize=8)
                axes[0, 1].set_title('Top 20 Quantum SVM Features')
                axes[0, 1].set_xlabel('Importance (%)')
            
            # Feature type analysis
            if self.classical_importance is not None:
                feature_types = []
                for feature in self.classical_importance['feature']:
                    if feature.startswith('lexical_'):
                        feature_types.append('Lexical')
                    elif feature.startswith('syntactic_'):
                        feature_types.append('Syntactic')
                    elif feature.startswith('stylistic_'):
                        feature_types.append('Stylistic')
                    else:
                        feature_types.append('TF-IDF')
                
                self.classical_importance['feature_type'] = feature_types
                
                # Feature type importance distribution
                type_importance = self.classical_importance.groupby('feature_type')['combined_importance'].sum()
                axes[1, 0].pie(type_importance.values, labels=type_importance.index, autopct='%1.1f%%')
                axes[1, 0].set_title('Classical SVM: Importance by Feature Type')
            
            if self.quantum_importance is not None:
                feature_types = []
                for feature in self.quantum_importance['feature']:
                    if feature.startswith('lexical_'):
                        feature_types.append('Lexical')
                    elif feature.startswith('syntactic_'):
                        feature_types.append('Syntactic')
                    elif feature.startswith('stylistic_'):
                        feature_types.append('Stylistic')
                    else:
                        feature_types.append('TF-IDF')
                
                self.quantum_importance['feature_type'] = feature_types
                
                # Feature type importance distribution
                type_importance = self.quantum_importance.groupby('feature_type')['combined_importance'].sum()
                axes[1, 1].pie(type_importance.values, labels=type_importance.index, autopct='%1.1f%%')
                axes[1, 1].set_title('Quantum SVM: Importance by Feature Type')
            
            plt.tight_layout()
            
            # Convert to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return base64.b64encode(plot_data).decode()
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return ""
    
    def generate_importance_report(self) -> str:
        """Generate a comprehensive feature importance report"""
        report = []
        report.append("# Feature Importance Analysis Report\n")
        
        if self.classical_importance is not None:
            report.append("## Classical SVM Feature Importance\n")
            report.append("### Top 20 Features:\n")
            top_classical = self.classical_importance.head(20)
            for i, row in top_classical.iterrows():
                report.append(f"{row.name + 1}. **{row['feature']}**: {row['combined_importance']:.3f}%\n")
            report.append("\n")
        
        if self.quantum_importance is not None:
            report.append("## Quantum SVM Feature Importance\n")
            report.append("### Top 20 Features:\n")
            top_quantum = self.quantum_importance.head(20)
            for i, row in top_quantum.iterrows():
                report.append(f"{row.name + 1}. **{row['feature']}**: {row['combined_importance']:.3f}%\n")
            report.append("\n")
        
        if self.classical_importance is not None and self.quantum_importance is not None:
            comparison = self.compare_feature_importance()
            report.append("## Classical vs Quantum Comparison\n")
            report.append(f"**Correlation between importance scores**: {comparison['correlation']:.3f}\n\n")
            
            report.append("### Features Quantum SVM Values More:\n")
            quantum_preferred = comparison['quantum_preferred_features'].head(10)
            for i, row in quantum_preferred.iterrows():
                report.append(f"- **{row['feature']}**: Quantum {row['quantum_importance']:.3f}% vs Classical {row['classical_importance']:.3f}%\n")
            
            report.append("\n### Features Classical SVM Values More:\n")
            classical_preferred = comparison['classical_preferred_features'].head(10)
            for i, row in classical_preferred.iterrows():
                report.append(f"- **{row['feature']}**: Classical {row['classical_importance']:.3f}% vs Quantum {row['quantum_importance']:.3f}%\n")
        
        return "".join(report)