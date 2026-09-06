#!/usr/bin/env python3
"""
t-SNE Visualization System for QSVM Project
Generates 2D visualizations to analyze feature space separation and model performance
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Must be set before pyplot import for headless environments
import matplotlib.pyplot as plt
plt.ioff()
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
from silhouette_analyzer import SilhouetteAnalyzer
import joblib
import json

class TSNEVisualizer:
    """Generate t-SNE visualizations for quantum and classical SVM analysis"""
    
    def __init__(self, output_dir: str = '.', dynamic_naming: bool = True, 
                 quantum_params: Dict[str, Any] = None):
        """
        Initialize t-SNE Visualizer
        
        Args:
            output_dir: Directory to save plots and analysis
            dynamic_naming: Whether to use dynamic file naming
            quantum_params: Quantum experiment parameters for naming
        """
        self.output_dir = output_dir
        self.dynamic_naming = dynamic_naming
        self.quantum_params = quantum_params or {}
        
        # Initialize raw data storage for reproducibility
        self._raw_tsne_data = {}
        
        # Default t-SNE parameters
        self.tsne_params = {
            'n_components': 2,
            'perplexity': 30,
            'learning_rate': 200,
            'max_iter': 1000,
            'random_state': 42,
            'init': 'pca',
            'metric': 'euclidean'
        }
        
        # Initialize silhouette analyzer with dynamic naming support
        self.silhouette_analyzer = SilhouetteAnalyzer(
            save_dir=os.path.join(output_dir, "silhouette_analysis"),
            dynamic_naming=dynamic_naming,
            quantum_params=quantum_params
        )
        # Plot settings — palette supports up to 10 classes
        self.plot_settings = {
            'figsize': (12, 10),
            'dpi': 300,
            'style': 'whitegrid',
            'palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                        '#bcbd22', '#17becf'],
            'alpha': 0.7,
            'point_size': 50
        }
        
        # Track generated plots
        self.plots_generated = []
    
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
        
        # Generate dynamic filename
        feature_dim = self.quantum_params.get('feature_dim', 4)
        shots = self.quantum_params.get('shots', 1024)
        train_samples = self.quantum_params.get('train_samples', 200)
        val_samples = self.quantum_params.get('val_samples', 100)
        
        shots_str = "exact" if shots in ("deterministic", "exact") else str(shots)
        
        dynamic_name = f"{name_without_ext}_{feature_dim}-{shots_str}-{train_samples}-{val_samples}"
        
        if extension:
            return f"{dynamic_name}.{extension}"
        else:
            return dynamic_name
    
    def _get_feature_label(self, data_type: str, n_features: int) -> str:
        """Get context-aware feature label based on data type"""
        if data_type == 'quantum_predictions':
            return 'Dimensions'
        elif data_type == 'pca_reduced':
            return 'Dimensions'
        elif data_type == 'original_features':
            return 'Features'
        elif data_type == 'classical_predictions':
            return 'Features'
        else:
            return 'Features'
    
    def _get_display_name(self, data_type: str) -> str:
        """Get proper display name for data type without underscores"""
        display_names = {
            'original_features': 'Original Features',
            'pca_reduced': 'PCA Reduced',
            'classical_predictions': 'Classical SVM Predictions',
            'quantum_predictions': 'QSVM Predictions'
        }
        return display_names.get(data_type, data_type.replace('_', ' ').title())
    
    def _format_samples_string(self, viz_data: Dict[str, Any]) -> str:
        """Format samples string with training/validation breakdown if available"""
        n_samples = viz_data['n_samples']
        n_train = viz_data.get('n_train_samples')
        n_val = viz_data.get('n_val_samples')
        
        # Special case: hold-out only (no training points)
        if n_train == 0 and n_val is not None:
            return f"Hold-out: {n_val} samples"
        if n_train is not None and n_val is not None:
            return f"Samples: {n_samples} ({n_train} training, {n_val} validation)"
        else:
            return f"Samples: {n_samples}"
    
    def prepare_data_for_tsne(self, X_data: np.ndarray, y_data: np.ndarray, 
                             data_type: str, model_predictions: np.ndarray = None,
                             class_names: List[str] = None, 
                             n_train_samples: int = None, n_val_samples: int = None) -> Dict[str, Any]:
        """
        Prepare data for t-SNE visualization
        
        Args:
            X_data: Feature data
            y_data: True labels
            data_type: Type of data ('original', 'pca_reduced', etc.)
            model_predictions: Model predictions (optional)
            class_names: Class names for labels
            n_train_samples: Number of training samples (optional)
            n_val_samples: Number of validation samples (optional)
            
        Returns:
            Dictionary with prepared data
        """
        if class_names is None:
            class_names = ['Gemma 3', 'Qwen 2.5']
        
        # Standardize data for t-SNE
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_data)
        
        # Apply t-SNE — clamp perplexity so it never exceeds n_samples - 1
        print(f"🔄 Computing t-SNE for {data_type} data ({X_scaled.shape[0]} samples, {X_scaled.shape[1]} features)...")
        tsne_start = time.time()

        tsne_params = dict(self.tsne_params)
        max_perp = max(1, X_scaled.shape[0] - 1)
        if tsne_params.get('perplexity', 30) > max_perp:
            tsne_params['perplexity'] = max_perp
            print(f"⚠️  Perplexity clamped to {max_perp} (n_samples={X_scaled.shape[0]})")

        tsne = TSNE(**tsne_params)
        X_tsne = tsne.fit_transform(X_scaled)
        
        tsne_time = time.time() - tsne_start
        print(f"✅ t-SNE completed in {tsne_time:.2f}s")
        
        # Prepare visualization data
        viz_data = {
            'X_tsne': X_tsne,
            'y_true': y_data,
            'y_pred': model_predictions,
            'data_type': data_type,
            'class_names': class_names,
            'n_samples': len(X_data),
            'n_features': X_data.shape[1],
            'n_train_samples': n_train_samples,
            'n_val_samples': n_val_samples,
            'tsne_time': tsne_time,
            'tsne_params': self.tsne_params.copy(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Store raw data for reproducibility
        self._raw_tsne_data[data_type] = {
            'tsne_coords': X_tsne,
            'true_labels': y_data,
            'class_names': class_names,
            'n_features_original': X_data.shape[1],
            'computation_time': tsne_time,
            'preprocessing': 'StandardScaler',
            'tsne_params': self.tsne_params.copy()
        }
        
        # Add predictions if available
        if model_predictions is not None:
            self._raw_tsne_data[data_type]['predictions'] = model_predictions
            self._raw_tsne_data[data_type]['prediction_source'] = f'{data_type}_model'
        
        # Store original features if reasonable size (for small feature sets)
        if X_data.shape[1] <= 100:  # Only store if manageable size
            self._raw_tsne_data[data_type]['original_features'] = X_data
            self._raw_tsne_data[data_type]['include_features'] = True
        else:
            self._raw_tsne_data[data_type]['include_features'] = False
        
        return viz_data
    
    def create_tsne_plot(self, viz_data: Dict[str, Any], plot_type: str = 'true_labels',
                        save_plot: bool = True) -> str:
        """
        Create t-SNE visualization plot
        
        Args:
            viz_data: Visualization data from prepare_data_for_tsne
            plot_type: Type of plot ('true_labels', 'predictions', 'comparison')
            save_plot: Whether to save the plot
            
        Returns:
            Path to saved plot file
        """
        plt.style.use('default')
        sns.set_style(self.plot_settings['style'])
        
        X_tsne = viz_data['X_tsne']
        y_true = viz_data['y_true']
        y_pred = viz_data.get('y_pred')
        class_names = viz_data['class_names']
        data_type = viz_data['data_type']
        
        if plot_type == 'comparison' and y_pred is not None:
            # Create comparison plot with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Build label set from actual values in y_true and y_pred
            unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

            # True labels plot
            for i, class_name in enumerate(class_names):
                lbl = unique_labels[i] if i < len(unique_labels) else i
                mask = y_true == lbl
                color = self.plot_settings['palette'][i % len(self.plot_settings['palette'])]
                ax1.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                           c=color,
                           label=f'{class_name} (True)',
                           alpha=self.plot_settings['alpha'],
                           s=self.plot_settings['point_size'])

            ax1.set_title(f'True Labels', fontsize=16, fontweight='bold')
            ax1.set_xlabel('t-SNE Component 1', fontsize=14)
            ax1.set_ylabel('t-SNE Component 2', fontsize=14)
            ax1.tick_params(axis='both', which='major', labelsize=14)
            ax1.legend(fontsize=14)
            ax1.grid(True, alpha=0.3)
            # Auto-scale axes for hold-out DBI plot (no fixed limits)

            # Predicted labels plot
            for i, class_name in enumerate(class_names):
                lbl = unique_labels[i] if i < len(unique_labels) else i
                mask = y_pred == lbl
                color = self.plot_settings['palette'][i % len(self.plot_settings['palette'])]
                ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                           c=color,
                           label=f'{class_name} (Predicted)',
                           alpha=self.plot_settings['alpha'],
                           s=self.plot_settings['point_size'])
            
            ax2.set_title(f'Model Predictions', fontsize=16, fontweight='bold')
            ax2.set_xlabel('t-SNE Component 1', fontsize=14)
            ax2.set_ylabel('t-SNE Component 2', fontsize=14)
            ax2.tick_params(axis='both', which='major', labelsize=14)
            ax2.legend(fontsize=14)
            ax2.grid(True, alpha=0.3)
            # Auto-scale axes for hold-out DBI plot (no fixed limits)
            
            # Add metadata
            samples_str = self._format_samples_string(viz_data)
            if data_type == 'classical_predictions':
                # For classical SVM, don't include shots info
                fig.suptitle(f't-SNE Analysis: {self._get_display_name(data_type)} Feature Space\n'
                            f'{samples_str}, {self._get_feature_label(data_type, viz_data["n_features"])}: {viz_data["n_features"]}, '
                            f'Perplexity: {self.tsne_params["perplexity"]}', 
                            fontsize=18, fontweight='bold', y=0.98)
            elif data_type == 'quantum_predictions':
                # For quantum SVM, include shots and reorder: Samples, Qubits, Shots, Perplexity
                shots_value = self.quantum_params.get('shots', 'N/A') if self.quantum_params else 'N/A'
                shots_str = "exact" if shots_value in ("deterministic", "exact") else str(shots_value)
                shots_info = f", Shots: {shots_str}" if self.quantum_params and 'shots' in self.quantum_params else ""
                fig.suptitle(f't-SNE Analysis: {self._get_display_name(data_type)} Feature Space\n'
                            f'{samples_str}, {self._get_feature_label(data_type, viz_data["n_features"])}: {viz_data["n_features"]}{shots_info}, '
                            f'Perplexity: {self.tsne_params["perplexity"]}', 
                            fontsize=18, fontweight='bold', y=0.98)
            else:
                # For other data types, use proper display names
                shots_value = self.quantum_params.get('shots', 'N/A') if self.quantum_params else 'N/A'
                shots_str = "exact" if shots_value in ("deterministic", "exact") else str(shots_value)
                shots_info = f", Shots: {shots_str}" if self.quantum_params and 'shots' in self.quantum_params else ""
                fig.suptitle(f't-SNE Analysis: {self._get_display_name(data_type)} Feature Space\n'
                            f'{samples_str}{shots_info}, {self._get_feature_label(data_type, viz_data["n_features"])}: {viz_data["n_features"]}, '
                            f'Perplexity: {self.tsne_params["perplexity"]}', 
                            fontsize=18, fontweight='bold', y=0.98)
            
            plot_suffix = 'comparison'
            
        else:
            # Single plot
            fig, ax = plt.subplots(figsize=self.plot_settings['figsize'])
            
            # Choose labels to plot
            labels_to_plot = y_pred if plot_type == 'predictions' and y_pred is not None else y_true
            label_type = 'Predicted' if plot_type == 'predictions' and y_pred is not None else 'True'
            
            # Build label→index mapping from actual values present
            unique_labels_single = sorted(np.unique(labels_to_plot))

            # Create scatter plot
            for i, class_name in enumerate(class_names):
                lbl = unique_labels_single[i] if i < len(unique_labels_single) else i
                mask = labels_to_plot == lbl
                color = self.plot_settings['palette'][i % len(self.plot_settings['palette'])]
                ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                          c=color,
                          label=f'{class_name} ({label_type})',
                          alpha=self.plot_settings['alpha'],
                          s=self.plot_settings['point_size'])
            
            ax.set_title(f'{label_type} Labels', 
                        fontsize=18, fontweight='bold')
            ax.set_xlabel('t-SNE Component 1', fontsize=16)
            ax.set_ylabel('t-SNE Component 2', fontsize=16)
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.3)
            # Auto-scale axes for hold-out DBI plot (no fixed limits)
            
            # Add metadata text box
            samples_str = self._format_samples_string(viz_data)
            if data_type == 'classical_predictions':
                # For classical SVM, don't include shots info
                textstr = f'{samples_str}\n{self._get_feature_label(data_type, viz_data["n_features"])}: {viz_data["n_features"]}\n' \
                         f'Perplexity: {self.tsne_params["perplexity"]}'
            elif data_type == 'quantum_predictions':
                # For quantum SVM, include shots; special-case hold-out DBI plot for a cleaner label box
                shots_value = self.quantum_params.get('shots', 'N/A') if self.quantum_params else 'N/A'
                shots_str = "exact" if shots_value in ("deterministic", "exact") else str(shots_value)
                shots_info = f"\nShots: {shots_str}" if self.quantum_params and 'shots' in self.quantum_params else ""
                if viz_data.get('is_holdout', False):
                    # Hold-out DBI plot: include samples again for scientific clarity
                    textstr = f'{samples_str}\n{self._get_feature_label(data_type, viz_data["n_features"])}: {viz_data["n_features"]}{shots_info}\n' \
                             f'Perplexity: {self.tsne_params["perplexity"]}'
                else:
                    textstr = f'{samples_str}\n{self._get_feature_label(data_type, viz_data["n_features"])}: {viz_data["n_features"]}{shots_info}\n' \
                             f'Perplexity: {self.tsne_params["perplexity"]}'
            else:
                # For other data types, use proper display names
                shots_value = self.quantum_params.get('shots', 'N/A') if self.quantum_params else 'N/A'
                shots_str = "exact" if shots_value in ("deterministic", "exact") else str(shots_value)
                shots_info = f"\nShots: {shots_str}" if self.quantum_params and 'shots' in self.quantum_params else ""
                textstr = f'{samples_str}{shots_info}\n{self._get_feature_label(data_type, viz_data["n_features"])}: {viz_data["n_features"]}\n' \
                         f'Perplexity: {self.tsne_params["perplexity"]}'

            # Append DBI to metadata box if available
            if 'dbi' in viz_data and viz_data['dbi'] is not None:
                textstr = f"{textstr}\nDBI: {viz_data['dbi']:.3f}"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=14,
                   verticalalignment='top', bbox=props)
            
            plot_suffix = plot_type
        
        plt.tight_layout()
        
        if save_plot:
            # Generate filename
            base_filename = f'tsne_{data_type}_{plot_suffix}.png'
            filename = self.get_dynamic_filename(base_filename)
            filepath = os.path.join(self.output_dir, filename)
            
            # Save plot
            plt.savefig(filepath, dpi=self.plot_settings['dpi'], bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"✅ t-SNE plot saved: {filename}")
            
            # Store plot info
            plot_info = {
                'filename': filename,
                'filepath': filepath,
                'plot_type': plot_type,
                'data_type': data_type,
                'timestamp': datetime.now().isoformat(),
                'file_size_mb': os.path.getsize(filepath) / (1024 * 1024) if os.path.exists(filepath) else 0
            }
            self.plots_generated.append(plot_info)
            
            plt.close()
            return filepath
        else:
            plt.show()
            return None
    
    def analyze_separation_quality(self, viz_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the quality of class separation in t-SNE space
        
        Args:
            viz_data: Visualization data from prepare_data_for_tsne
            
        Returns:
            Dictionary with separation quality metrics
        """
        X_tsne = viz_data['X_tsne']
        y_true = viz_data['y_true']
        class_names = viz_data['class_names']
        
        # Calculate class centroids — use actual label values, not enumerate index
        centroids = {}
        class_stats = {}
        unique_labels_sep = sorted(np.unique(y_true))

        for i, class_name in enumerate(class_names):
            lbl = unique_labels_sep[i] if i < len(unique_labels_sep) else i
            mask = y_true == lbl
            class_points = X_tsne[mask]
            
            if len(class_points) > 0:
                centroid = np.mean(class_points, axis=0)
                centroids[class_name] = centroid
                
                # Calculate intra-class statistics
                distances_to_centroid = np.linalg.norm(class_points - centroid, axis=1)
                
                class_stats[class_name] = {
                    'n_samples': len(class_points),
                    'centroid': centroid.tolist(),
                    'mean_distance_to_centroid': float(np.mean(distances_to_centroid)),
                    'std_distance_to_centroid': float(np.std(distances_to_centroid)),
                    'max_distance_to_centroid': float(np.max(distances_to_centroid)),
                    'compactness_score': float(np.mean(distances_to_centroid))  # Lower is more compact
                }
        
        # Calculate inter-class separation using whichever classes have centroids
        centroid_keys = list(centroids.keys())
        if len(centroid_keys) >= 2:
            centroid_distance = np.linalg.norm(
                centroids[centroid_keys[0]] - centroids[centroid_keys[1]]
            )
            
            # Calculate separation ratio (higher is better separated)
            avg_compactness = np.mean([stats['compactness_score'] for stats in class_stats.values()])
            separation_ratio = centroid_distance / avg_compactness if avg_compactness > 0 else 0
        else:
            centroid_distance = 0
            separation_ratio = 0
        
        # Silhouette analysis via dedicated analyzer
        shots_value = self.quantum_params.get("shots", "unknown")
        shots_str = "exact" if shots_value in ("deterministic", "exact") else f"{shots_value}shots"
        experiment_id = f"{self.quantum_params.get('feature_dim', 'unknown')}D_{shots_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        silhouette_analysis = self.silhouette_analyzer.calculate_silhouette_scores(
            X_tsne, y_true, viz_data["data_type"], experiment_id
        )
        avg_silhouette = silhouette_analysis.get("sklearn_silhouette_score", None)
        
        analysis_result = {
            'data_type': viz_data['data_type'],
            'n_samples': viz_data['n_samples'],
            'n_features': viz_data['n_features'],
            'class_statistics': class_stats,
            'inter_class_distance': float(centroid_distance),
            'separation_ratio': float(separation_ratio),
            'average_silhouette_score': avg_silhouette,
            'separation_quality': 'Excellent' if separation_ratio > 3 else 
                                 'Good' if separation_ratio > 2 else 
                                 'Fair' if separation_ratio > 1 else 'Poor',
            "analysis_timestamp": datetime.now().isoformat(),
            "comprehensive_silhouette_analysis": silhouette_analysis
        }
        
        return analysis_result
    
    def generate_comprehensive_analysis(self, X_train: np.ndarray, X_val: np.ndarray,
                                      y_train: np.ndarray, y_val: np.ndarray,
                                      classical_model=None, quantum_model=None,
                                      quantum_pca=None, class_names: List[str] = None,
                                      generate_plots: bool = True,
                                      precomputed_classical_predictions: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                      precomputed_quantum_predictions: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                      precomputed_quantum_features: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive t-SNE analysis for all data types
        
        Args:
            X_train: Training features (original space)
            X_val: Validation features (original space)
            y_train: Training labels
            y_val: Validation labels
            classical_model: Trained classical model
            quantum_model: Trained quantum model
            quantum_pca: PCA model used for quantum features
            class_names: Class names
            generate_plots: Whether to generate PNG visualizations (default: True)
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        if class_names is None:
            class_names = ['Gemma 3', 'Qwen 2.5']
        
        print("🎨 Starting Comprehensive t-SNE Analysis")
        print("=" * 50)
        
        analysis_start_time = time.time()
        comprehensive_results = {
            'analysis_config': {
                'tsne_params': self.tsne_params,
                'plot_settings': self.plot_settings,
                'class_names': class_names,
                'timestamp': datetime.now().isoformat()
            },
            'visualizations': [],
            'separation_analyses': [],
            'model_comparisons': []
        }
        
        # 1. Original Feature Space Analysis
        print("\n📊 Analyzing Original Feature Space...")
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.hstack([y_train, y_val])
        
        original_viz_data = self.prepare_data_for_tsne(
            X_combined, y_combined, 'original_features', 
            class_names=class_names,
            n_train_samples=len(X_train), n_val_samples=len(X_val)
        )
        
        # Create plots for original space (if enabled)
        if generate_plots:
            self.create_tsne_plot(original_viz_data, 'true_labels')
        
        # Analyze separation quality
        original_separation = self.analyze_separation_quality(original_viz_data)
        comprehensive_results['separation_analyses'].append(original_separation)
        comprehensive_results['visualizations'].append({
            'data_type': 'original_features',
            'plots_created': ['true_labels'] if generate_plots else [],
            'viz_data_summary': {
                'n_samples': original_viz_data['n_samples'],
                'n_features': original_viz_data['n_features'],
                'tsne_time': original_viz_data['tsne_time']
            }
        })
        
        # 2. PCA-Reduced Space Analysis (if quantum model exists)
        if quantum_pca is not None:
            print("\n📊 Analyzing PCA-Reduced Feature Space...")
            scaler = getattr(quantum_model, 'feature_scaler', getattr(quantum_model, 'scaler', None)) if quantum_model is not None else None
            if scaler is not None:
                X_train_pca = quantum_pca.transform(scaler.transform(X_train))
                X_val_pca = quantum_pca.transform(scaler.transform(X_val))
            else:
                X_train_pca = quantum_pca.transform(X_train)
                X_val_pca = quantum_pca.transform(X_val)
            X_combined_pca = np.vstack([X_train_pca, X_val_pca])
            
            pca_viz_data = self.prepare_data_for_tsne(
                X_combined_pca, y_combined, 'pca_reduced', 
                class_names=class_names,
                n_train_samples=len(X_train), n_val_samples=len(X_val)
            )
            
            # Create plots for PCA space (if enabled)
            if generate_plots:
                self.create_tsne_plot(pca_viz_data, 'true_labels')
            
            # Analyze separation quality
            pca_separation = self.analyze_separation_quality(pca_viz_data)
            comprehensive_results['separation_analyses'].append(pca_separation)
            comprehensive_results['visualizations'].append({
                'data_type': 'pca_reduced',
                'plots_created': ['true_labels'] if generate_plots else [],
                'viz_data_summary': {
                    'n_samples': pca_viz_data['n_samples'],
                    'n_features': pca_viz_data['n_features'],
                    'tsne_time': pca_viz_data['tsne_time']
                }
            })
        
        # 3. Model Predictions Analysis
        if classical_model is not None:
            print("\n📊 Analyzing Classical Model Predictions...")
            try:
                if precomputed_classical_predictions is not None:
                    classical_train_pred, classical_val_pred = precomputed_classical_predictions
                else:
                    classical_train_pred = classical_model.predict(X_train)
                    classical_val_pred = classical_model.predict(X_val)
                classical_combined_pred = np.hstack([classical_train_pred, classical_val_pred])
                
                classical_pred_viz_data = self.prepare_data_for_tsne(
                    X_combined, y_combined, 'classical_predictions', 
                    model_predictions=classical_combined_pred, 
                    class_names=class_names,
                    n_train_samples=len(X_train), n_val_samples=len(X_val)
                )
                
                # Create comparison plot (if enabled)
                if generate_plots:
                    self.create_tsne_plot(classical_pred_viz_data, 'comparison')
                
                # Analyze separation quality for classical predictions
                classical_separation = self.analyze_separation_quality(classical_pred_viz_data)
                comprehensive_results['separation_analyses'].append(classical_separation)
                
                comprehensive_results['visualizations'].append({
                    'data_type': 'classical_predictions',
                    'plots_created': ['comparison'] if generate_plots else [],
                    'viz_data_summary': {
                        'n_samples': classical_pred_viz_data['n_samples'],
                        'n_features': classical_pred_viz_data['n_features'],
                        'tsne_time': classical_pred_viz_data['tsne_time']
                    }
                })
                
            except Exception as e:
                print(f"⚠️  Classical model prediction analysis failed: {e}")
        
        if quantum_model is not None and (quantum_pca is not None or precomputed_quantum_features is not None):
            print("\n📊 Analyzing Quantum Model Predictions...")
            try:
                # Use precomputed quantum features if provided (always available for skip_pca modes)
                if precomputed_quantum_features is not None:
                    X_train_quantum, X_val_quantum = precomputed_quantum_features
                elif quantum_pca is not None:
                    # Current mode: scale → PCA → [0, 2π]
                    scaler = getattr(quantum_model, 'feature_scaler', getattr(quantum_model, 'scaler', None))
                    if scaler is not None:
                        X_train_scaled = scaler.transform(X_train)
                        X_val_scaled = scaler.transform(X_val)
                    else:
                        X_train_scaled = X_train
                        X_val_scaled = X_val
                    X_train_pca = quantum_pca.transform(X_train_scaled)
                    X_val_pca = quantum_pca.transform(X_val_scaled)

                    # Normalize to [0, 2π] using training-fitted stats
                    if hasattr(quantum_model, '_normalize_to_2pi'):
                        X_train_quantum = quantum_model._normalize_to_2pi(X_train_pca)
                        X_val_quantum = quantum_model._normalize_to_2pi(X_val_pca)
                    else:
                        q_min = getattr(quantum_model, 'quant_norm_min', X_train_pca.min(axis=0))
                        q_max = getattr(quantum_model, 'quant_norm_max', X_train_pca.max(axis=0))
                        q_range = q_max - q_min
                        q_range[q_range == 0] = 1
                        X_train_quantum = np.clip(2 * np.pi * (X_train_pca - q_min) / q_range, 0.0, 2 * np.pi)
                        X_val_quantum = np.clip(2 * np.pi * (X_val_pca - q_min) / q_range, 0.0, 2 * np.pi)
                else:
                    raise RuntimeError("No quantum features available (quantum_pca=None and precomputed_quantum_features=None)")

                # Predict with the trained QSVM; prefer original features (cuTensorNet), fallback to normalized PCA (QSVC)
                if precomputed_quantum_predictions is not None:
                    quantum_train_pred, quantum_val_pred = precomputed_quantum_predictions
                else:
                    def _try_predict(model, X):
                        return model.predict(X) if hasattr(model, 'predict') else None

                    # Try original feature space first (expected by cuTensorNet wrapper)
                    quantum_train_pred = None
                    quantum_val_pred = None
                    try:
                        quantum_train_pred = _try_predict(quantum_model, X_train)
                        quantum_val_pred = _try_predict(quantum_model, X_val)
                    except Exception:
                        quantum_train_pred = None
                        quantum_val_pred = None

                    # If that failed or returned None, try normalized PCA space (expected by QSVC)
                    if quantum_train_pred is None or quantum_val_pred is None:
                        try:
                            quantum_train_pred = _try_predict(quantum_model, X_train_quantum)
                            quantum_val_pred = _try_predict(quantum_model, X_val_quantum)
                        except Exception:
                            pass

                    # As a last resort, attempt inner estimator if present (only valid when input already correct type)
                    if (quantum_train_pred is None or quantum_val_pred is None) and hasattr(quantum_model, 'qsvm_model') and hasattr(quantum_model.qsvm_model, 'predict'):
                        try:
                            quantum_train_pred = quantum_model.qsvm_model.predict(X_train_quantum)
                            quantum_val_pred = quantum_model.qsvm_model.predict(X_val_quantum)
                        except Exception:
                            pass

                    if quantum_train_pred is None or quantum_val_pred is None:
                        raise RuntimeError('Quantum prediction failed on both original and normalized feature spaces')
                quantum_combined_pred = np.hstack([quantum_train_pred, quantum_val_pred])
                
                # Use the same quantum feature space that predictions were made on
                X_combined_quantum = np.vstack([X_train_quantum, X_val_quantum])

                quantum_pred_viz_data = self.prepare_data_for_tsne(
                    X_combined_quantum, y_combined, 'quantum_predictions', 
                    model_predictions=quantum_combined_pred, 
                    class_names=class_names,
                    n_train_samples=len(X_train), n_val_samples=len(X_val)
                )
                
                # Create comparison plot (if enabled)
                if generate_plots:
                    self.create_tsne_plot(quantum_pred_viz_data, 'comparison')
                
                # Analyze separation quality for quantum predictions
                quantum_separation = self.analyze_separation_quality(quantum_pred_viz_data)
                comprehensive_results['separation_analyses'].append(quantum_separation)
                
                comprehensive_results['visualizations'].append({
                    'data_type': 'quantum_predictions',
                    'plots_created': ['comparison'] if generate_plots else [],
                    'viz_data_summary': {
                        'n_samples': quantum_pred_viz_data['n_samples'],
                        'n_features': quantum_pred_viz_data['n_features'],
                        'tsne_time': quantum_pred_viz_data['tsne_time']
                    }
                })

                # Validation-only DBI in exact QSVM input space (normalization uses model's stored training stats)
                # For skip_pca modes the precomputed features are already in quantum-ready form.
                try:
                    if quantum_pca is None and precomputed_quantum_features is not None:
                        # skip_pca=True path: use the val quantum features directly
                        _, X_val_quantum_norm = precomputed_quantum_features
                        if len(np.unique(y_val)) >= 2:
                            dbi_val = davies_bouldin_score(X_val_quantum_norm, y_val)
                            comprehensive_results.setdefault('metrics', {})['quantum_validation_dbi'] = float(dbi_val)
                            print(f"   Quantum DBI (skip_pca): {dbi_val:.4f}")
                        # Skip PCA-dependent DBI plot/JSON sections (quantum_pca attrs unavailable)
                    if quantum_pca is None:
                        raise ValueError("quantum_pca=None: skipping PCA-dependent DBI sections")
                    dbi_scaler = getattr(quantum_model, 'feature_scaler', getattr(quantum_model, 'scaler', None)) if quantum_model else None
                    X_val_for_dbi = dbi_scaler.transform(X_val) if dbi_scaler is not None else X_val
                    X_val_pca_dbi = quantum_pca.transform(X_val_for_dbi)

                    norm_min = norm_max = None
                    if quantum_model is not None and hasattr(quantum_model, '_normalize_to_2pi'):
                        X_val_quantum_norm = quantum_model._normalize_to_2pi(X_val_pca_dbi)
                        if hasattr(quantum_model, 'quant_norm_min') and hasattr(quantum_model, 'quant_norm_max'):
                            norm_min = np.asarray(quantum_model.quant_norm_min)
                            norm_max = np.asarray(quantum_model.quant_norm_max)
                    elif quantum_model is not None and hasattr(quantum_model, 'quant_norm_min') and hasattr(quantum_model, 'quant_norm_max'):
                        q_min = quantum_model.quant_norm_min
                        q_max = quantum_model.quant_norm_max
                        q_range = q_max - q_min
                        q_range[q_range == 0] = 1
                        X_val_quantum_norm = np.clip(2 * np.pi * (X_val_pca_dbi - q_min) / q_range, 0.0, 2 * np.pi)
                        norm_min = np.asarray(q_min)
                        norm_max = np.asarray(q_max)
                    elif quantum_model is not None and hasattr(quantum_model, '_norm_min'):
                        _safe_range = np.where(quantum_model._norm_range == 0, 1.0, quantum_model._norm_range)
                        X_val_quantum_norm = np.clip(2 * np.pi * (X_val_pca_dbi - quantum_model._norm_min) / _safe_range, 0.0, 2 * np.pi)
                        norm_min = np.asarray(quantum_model._norm_min)
                        norm_max = np.asarray(quantum_model._norm_min) + np.asarray(quantum_model._norm_range)
                    else:
                        X_train_for_dbi = dbi_scaler.transform(X_train) if dbi_scaler is not None else X_train
                        X_train_pca_dbi = quantum_pca.transform(X_train_for_dbi)
                        norm_min = X_train_pca_dbi.min(axis=0)
                        norm_max = X_train_pca_dbi.max(axis=0)
                        X_train_range = norm_max - norm_min
                        X_train_range = np.where(X_train_range == 0, 1, X_train_range)
                        X_val_quantum_norm = np.clip(2 * np.pi * (X_val_pca_dbi - norm_min) / X_train_range, 0.0, 2 * np.pi)

                    if len(np.unique(y_val)) < 2:
                        raise ValueError("DBI requires at least 2 classes; only 1 class present in y_val")
                    dbi_val = davies_bouldin_score(X_val_quantum_norm, y_val)
                    comprehensive_results.setdefault('metrics', {})['quantum_validation_dbi'] = float(dbi_val)

                    # Create a validation-only t-SNE plot with DBI annotated
                    if generate_plots:
                        # Ensure t-SNE perplexity < n_samples for hold-out-only plot
                        n_val_samples = len(y_val)
                        old_perp = self.tsne_params.get('perplexity', 30)
                        if old_perp >= n_val_samples:
                            self.tsne_params['perplexity'] = max(2, min(n_val_samples - 1, n_val_samples // 3))
                        try:
                            # Use a distinct data_type to avoid overwriting the combined dataset
                            val_viz = self.prepare_data_for_tsne(
                                X_val_quantum_norm, y_val, 'quantum_predictions_holdout',
                                class_names=class_names,
                                n_train_samples=0, n_val_samples=n_val_samples
                            )
                            val_viz['dbi'] = float(dbi_val)
                            val_viz['is_holdout'] = True
                            self.create_tsne_plot(val_viz, 'true_labels')
                        finally:
                            # Restore original perplexity
                            self.tsne_params['perplexity'] = old_perp
                        comprehensive_results['visualizations'].append({
                            'data_type': 'quantum_predictions_holdout',
                            'plots_created': ['true_labels'],
                            'viz_data_summary': {
                                'n_samples': val_viz['n_samples'],
                                'n_features': val_viz['n_features'],
                                'tsne_time': val_viz['tsne_time'],
                                'dbi': float(dbi_val)
                            }
                        })

                    # Save DBI raw data (JSON + CSV) with dynamic naming
                    try:
                        # Compute cumulative explained variance
                        try:
                            k = self.quantum_params.get('feature_dim')
                            cum_explained = float(np.sum(quantum_pca.explained_variance_ratio_[:k]))
                        except Exception:
                            cum_explained = None

                        # Clipping diagnostics
                        clipped_zero = (X_val_quantum_norm <= 0.0 + 1e-12)
                        clipped_2pi = (X_val_quantum_norm >= 2 * np.pi - 1e-12)
                        zero_frac_per_comp = clipped_zero.mean(axis=0).tolist()
                        twopi_frac_per_comp = clipped_2pi.mean(axis=0).tolist()
                        zero_frac_overall = float(clipped_zero.mean())
                        twopi_frac_overall = float(clipped_2pi.mean())

                        # Normalize backend label to canonical {cutensornet,statevector,aer}
                        raw_backend = self.quantum_params.get('backend', 'unknown') or 'unknown'
                        backend_norm = 'cutensornet' if 'cutensornet' in str(raw_backend).lower() else (
                            'statevector' if 'statevector' in str(raw_backend).lower() else (
                            'aer' if 'aer' in str(raw_backend).lower() else str(raw_backend)))

                        # Normalize shots for statevector (shot-insensitive)
                        raw_shots = self.quantum_params.get('shots')
                        shots_norm = "N/A (exact statevector)" if backend_norm == 'statevector' else raw_shots

                        # PCA provenance
                        pca_n_components = getattr(quantum_pca, 'n_components', None)
                        pca_n_components_fit = getattr(quantum_pca, 'n_components_', None)
                        pca_solver = getattr(quantum_pca, 'svd_solver', None)
                        pca_random_state = str(getattr(quantum_pca, 'random_state', None))

                        dbi_record = {
                            'dbi': float(dbi_val),
                            'k': int(self.quantum_params.get('feature_dim')) if self.quantum_params.get('feature_dim') is not None else None,
                            'feature_dim': int(self.quantum_params.get('feature_dim')) if self.quantum_params.get('feature_dim') is not None else None,
                            'shots': shots_norm,
                            'train_samples': int(self.quantum_params.get('train_samples')) if self.quantum_params.get('train_samples') is not None else None,
                            'holdout_samples': int(self.quantum_params.get('val_samples')) if self.quantum_params.get('val_samples') is not None else None,
                            'val_samples': int(self.quantum_params.get('val_samples')) if self.quantum_params.get('val_samples') is not None else None,
                            'backend': backend_norm,
                            'split': 'holdout',
                            'seed': 42,
                            'pca_cumulative_explained_variance': cum_explained,
                            'pca_n_components_config': pca_n_components,
                            'pca_n_components_fit': pca_n_components_fit,
                            'pca_solver': pca_solver,
                            'pca_random_state': pca_random_state,
                            # Use configured train_samples if provided; otherwise fall back to actual length
                            'pca_training_samples': int(self.quantum_params.get('train_samples') or len(X_train)),
                            'scale_min': norm_min.tolist() if norm_min is not None else None,
                            'scale_max': norm_max.tolist() if norm_max is not None else None,
                            'clipping_stats': {
                                'per_component_zero_fraction': zero_frac_per_comp,
                                'per_component_2pi_fraction': twopi_frac_per_comp,
                                'overall_zero_fraction': zero_frac_overall,
                                'overall_2pi_fraction': twopi_frac_overall
                            },
                            'frac_clipped_at_0': zero_frac_overall,
                            'frac_clipped_at_2pi': twopi_frac_overall,
                            'note': 'DBI computed on PCA-k input space (pre-kernel); independent of shots',
                            'timestamp': datetime.now().isoformat()
                        }
                        # JSON
                        # Use a timestamped filename to avoid collisions across runs with same dynamic naming
                        timestamp_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
                        dbi_json_name = self.get_dynamic_filename(f'Davies-Bouldin_Index_raw_data_{timestamp_tag}.json')
                        dbi_json_path = os.path.join(self.output_dir, dbi_json_name)
                        def _json_safe(obj):
                            if isinstance(obj, (np.integer,)): return int(obj)
                            if isinstance(obj, (np.floating,)): return float(obj)
                            if isinstance(obj, np.ndarray): return obj.tolist()
                            return str(obj)
                        with open(dbi_json_path, 'w') as f:
                            json.dump(dbi_record, f, indent=2, default=_json_safe)
                        # CSV
                        dbi_csv_name = self.get_dynamic_filename(f'Davies-Bouldin_Index_raw_data_{timestamp_tag}.csv')
                        dbi_csv_path = os.path.join(self.output_dir, dbi_csv_name)
                        pd.DataFrame([dbi_record]).to_csv(dbi_csv_path, index=False)
                        # Track for moving
                        self.plots_generated.append({'filename': dbi_json_name, 'filepath': dbi_json_path})
                        self.plots_generated.append({'filename': dbi_csv_name, 'filepath': dbi_csv_path})
                        print(f"✅ DBI raw data saved: {dbi_json_name}, {dbi_csv_name}")
                    except Exception as e:
                        print(f"⚠️  Failed to save DBI raw data: {e}")
                except Exception as e:
                    print(f"⚠️  Validation DBI computation failed: {e}")
                
            except Exception as e:
                print(f"⚠️  Quantum model prediction analysis failed: {e}")
        
        # 4. Generate Summary Report
        total_analysis_time = time.time() - analysis_start_time
        
        comprehensive_results['summary'] = {
            'total_analysis_time': total_analysis_time,
            'total_plots_generated': len(self.plots_generated),
            'separation_quality_comparison': self._compare_separation_qualities(
                comprehensive_results['separation_analyses']
            ),
            'plots_generated': self.plots_generated
        }
        
        print(f"\n✅ Comprehensive t-SNE Analysis Complete!")
        print(f"   Total Time: {total_analysis_time:.2f}s")
        print(f"   Plots Generated: {len(self.plots_generated)}")
        print(f"   Separation Analyses: {len(comprehensive_results['separation_analyses'])}")
        
        # Export raw data for reproducibility
        print(f"\n💾 Exporting raw t-SNE data for reproducibility...")
        try:
            raw_data_path, csv_files = self.save_raw_tsne_data(comprehensive_results)
            comprehensive_results['summary']['raw_data_exported'] = {
                'json_file': os.path.basename(raw_data_path),
                'csv_files': csv_files,
                'total_datasets': len(self._raw_tsne_data),
                'export_timestamp': datetime.now().isoformat()
            }
            print(f"✅ Raw data export complete: {len(csv_files)} CSV files + 1 JSON file")
        except Exception as e:
            print(f"⚠️  Raw data export failed: {e}")
            comprehensive_results['summary']['raw_data_exported'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        return comprehensive_results
    
    def _compare_separation_qualities(self, separation_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare separation qualities across different feature spaces"""
        if not separation_analyses:
            return {}
        
        comparison = {
            'best_separation': None,
            'worst_separation': None,
            'separation_ranking': [],
            'insights': []
        }
        
        # Rank by separation ratio
        ranked_analyses = sorted(separation_analyses, 
                               key=lambda x: x['separation_ratio'], reverse=True)
        
        comparison['separation_ranking'] = [
            {
                'data_type': analysis['data_type'],
                'separation_ratio': analysis['separation_ratio'],
                'separation_quality': analysis['separation_quality'],
                'silhouette_score': analysis['average_silhouette_score']
            }
            for analysis in ranked_analyses
        ]
        
        if ranked_analyses:
            comparison['best_separation'] = ranked_analyses[0]['data_type']
            comparison['worst_separation'] = ranked_analyses[-1]['data_type']
            
            # Generate insights
            best = ranked_analyses[0]
            if len(ranked_analyses) > 1:
                worst = ranked_analyses[-1]
                improvement = best['separation_ratio'] - worst['separation_ratio']
                comparison['insights'].append(
                    f"Best separation in {best['data_type']} space "
                    f"(ratio: {best['separation_ratio']:.2f}) vs "
                    f"{worst['data_type']} space (ratio: {worst['separation_ratio']:.2f}). "
                    f"Improvement: {improvement:.2f}x"
                )
            
            if best['separation_ratio'] > 2:
                comparison['insights'].append(
                    f"Excellent class separation achieved in {best['data_type']} space"
                )
            elif best['separation_ratio'] > 1:
                comparison['insights'].append(
                    f"Good class separation in {best['data_type']} space"
                )
            else:
                comparison['insights'].append(
                    "Poor class separation across all feature spaces - consider feature engineering"
                )
        
        return comparison
    
    def save_analysis_report(self, comprehensive_results: Dict[str, Any]) -> str:
        """Save comprehensive analysis report to JSON"""
        report_filename = self.get_dynamic_filename('tsne_analysis_report.json')
        report_path = os.path.join(self.output_dir, report_filename)
        
        with open(report_path, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        print(f"✅ t-SNE analysis report saved: {report_filename}")
        return report_path
    
    def save_raw_tsne_data(self, comprehensive_results: Dict[str, Any]) -> str:
        """Save raw t-SNE data for complete reproducibility"""
        import pandas as pd
        
        # Create comprehensive raw data structure
        raw_data = {
            'metadata': {
                'experiment_config': comprehensive_results.get('analysis_config', {}),
                'timestamp': comprehensive_results.get('analysis_config', {}).get('timestamp', ''),
                'total_samples': 0,
                'data_types_included': []
            },
            'datasets': {}
        }
        
        # Extract raw data from each visualization
        for viz in comprehensive_results.get('visualizations', []):
            data_type = viz['data_type']
            raw_data['metadata']['data_types_included'].append(data_type)
            
            # Get the stored t-SNE coordinates and related data
            if hasattr(self, '_raw_tsne_data') and data_type in self._raw_tsne_data:
                tsne_data = self._raw_tsne_data[data_type]
                
                # Create DataFrame with all relevant information
                dataset = {
                    'tsne_coordinates': {
                        'x': tsne_data['tsne_coords'][:, 0].tolist(),
                        'y': tsne_data['tsne_coords'][:, 1].tolist()
                    },
                    'labels': {
                        'true_labels': tsne_data['true_labels'].tolist(),
                        'class_names': tsne_data.get('class_names', [])
                    },
                    'sample_info': {
                        'sample_indices': list(range(len(tsne_data['tsne_coords']))),
                        'n_samples': len(tsne_data['tsne_coords']),
                        'n_features_original': tsne_data.get('n_features_original', 'unknown')
                    },
                    'tsne_parameters': tsne_data.get('tsne_params', {}),
                    'processing_info': {
                        'data_type': data_type,
                        'tsne_computation_time': tsne_data.get('computation_time', 0),
                        'preprocessing_applied': tsne_data.get('preprocessing', 'none')
                    }
                }
                
                # Add model predictions if available
                if 'predictions' in tsne_data:
                    dataset['predictions'] = {
                        'predicted_labels': tsne_data['predictions'].tolist(),
                        'prediction_source': tsne_data.get('prediction_source', 'unknown')
                    }
                
                # Add original feature data if requested and available
                if tsne_data.get('include_features', False) and 'original_features' in tsne_data:
                    # For large feature sets, we might want to limit this
                    n_features = tsne_data['original_features'].shape[1]
                    if n_features <= 100:  # Only include if reasonable size
                        dataset['original_features'] = tsne_data['original_features'].tolist()
                    else:
                        dataset['original_features_info'] = {
                            'note': 'Original features not included due to size',
                            'n_features': n_features,
                            'shape': list(tsne_data['original_features'].shape)
                        }
                
                raw_data['datasets'][data_type] = dataset
                raw_data['metadata']['total_samples'] += dataset['sample_info']['n_samples']
        
        # Save as JSON
        raw_data_filename = self.get_dynamic_filename('tsne_raw_data.json')
        raw_data_path = os.path.join(self.output_dir, raw_data_filename)
        
        with open(raw_data_path, 'w') as f:
            json.dump(raw_data, f, indent=2)
        
        print(f"✅ Raw t-SNE data saved: {raw_data_filename}")
        
        # Also save as CSV for easy analysis
        csv_files_created = []
        for data_type, dataset in raw_data['datasets'].items():
            # Create DataFrame with essential data
            df_data = {
                'sample_index': dataset['sample_info']['sample_indices'],
                'tsne_x': dataset['tsne_coordinates']['x'],
                'tsne_y': dataset['tsne_coordinates']['y'],
                'true_label': dataset['labels']['true_labels'],
                'true_label_name': [dataset['labels']['class_names'][i] if 0 <= i < len(dataset['labels']['class_names']) else f'class_{i}'
                                   for i in dataset['labels']['true_labels']]
            }

            # Add predictions if available
            if 'predictions' in dataset:
                df_data['predicted_label'] = dataset['predictions']['predicted_labels']
                df_data['predicted_label_name'] = [dataset['labels']['class_names'][i] if 0 <= i < len(dataset['labels']['class_names']) else f'class_{i}'
                                                  for i in dataset['predictions']['predicted_labels']]
                df_data['prediction_correct'] = [true == pred for true, pred in zip(dataset['labels']['true_labels'], dataset['predictions']['predicted_labels'])]
            
            df = pd.DataFrame(df_data)
            
            # Save CSV
            csv_filename = self.get_dynamic_filename(f'tsne_raw_data_{data_type}.csv')
            csv_path = os.path.join(self.output_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            csv_files_created.append(csv_filename)
            
            print(f"✅ Raw t-SNE CSV saved: {csv_filename}")
        
        return raw_data_path, csv_files_created 