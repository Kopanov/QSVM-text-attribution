#!/usr/bin/env python3
"""
Direct QSVM Training Script
Simple and reliable quantum SVM training using original files
"""

import os
import sys
import time
import random
import threading
from datetime import datetime
from pathlib import Path
import psutil
import numpy as np
from typing import Dict, Any, Optional

# Global RNG seeds for full reproducibility
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# Set up headless environment for supercomputer
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''

# =============================================================================
# CONFIGURATION PARAMETERS - Enhanced hybrid system
# =============================================================================

# Data Configuration. Override with DATA_FILE=/path/to/file.csv when needed.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = os.environ.get(
    'DATA_FILE',
    str(PROJECT_ROOT / 'data' / 'main_corpus.csv'),
)

# Training Sample Sizes (Environment-based for experiment flexibility)
TRAINING_SAMPLES = int(os.environ.get('TRAINING_SAMPLES', 4640))      # Number of training samples to use - max 4640
VALIDATION_SAMPLES = int(os.environ.get('VALIDATION_SAMPLES', 1160))    # Number of validation samples to use - max 1160

# SVM Regularization Parameters (separate for fair independent tuning)
SVM_C = float(os.environ.get('SVM_C', 1.0))                            # Classical SVM C (typical values: 0.1, 1, 10, 100)
QSVM_C = float(os.environ.get('QSVM_C', 1.0))                         # Quantum SVM C (typical values: 0.1, 1, 10, 100)

# Quantum SVM Parameters (Enhanced environment-based configuration)
QUANTUM_FEATURE_DIM = int(os.environ.get('QUANTUM_FEATURE_DIM', 4))    # Quantum feature dimension (2, 4, 6, 8, 10, 12)
QUANTUM_SHOTS = int(os.environ.get('QUANTUM_SHOTS', 1024))            # Number of quantum circuit shots (cuTensorNet and AER backends)
QUANTUM_BACKEND = os.environ.get('QUANTUM_BACKEND', 'qiskit_statevector_deterministic')  # Available backends:
                                                                        #   • 'cutensornet_gpu' - NVIDIA cuTensorNet GPU acceleration (fastest, recommended)
                                                                        #   • 'cutensornet_cpu' - NVIDIA cuTensorNet CPU simulation (fast tensor networks)
                                                                        #   • 'aer_simulator_gpu' - Qiskit AER GPU simulator (limited GPU support)
                                                                        #   • 'aer_simulator_cpu' - Qiskit AER CPU simulator (traditional, reliable)
                                                                        #   • 'aer_simulator' - Default Qiskit AER (auto-selects CPU/GPU)
                                                                        #   • 'qiskit_statevector_deterministic' - Exact statevector simulation (see STATEVECTOR_SHOTS)

# Statevector Kernel Shot-Noise Control (only for qiskit_statevector_deterministic backend)
# "None" / "exact" → truly noiseless: exact |<φ(x)|φ(y)>|² via Statevector inner product
# An integer (e.g. 1024)  → emulates shot noise via binomial sampling on exact fidelity
#   Use this to predict how results will degrade on real IBM hardware at a given shot budget.
_sv_shots_raw = os.environ.get('STATEVECTOR_SHOTS', 'None')
STATEVECTOR_SHOTS = None if _sv_shots_raw.lower() in ('none', 'exact', '') else int(_sv_shots_raw)

FEATURE_MAP_REPS = int(os.environ.get('FEATURE_MAP_REPS', 2))         # Feature map repetitions (1 or 2 recommended)
FEATURE_MAP_TYPE = os.environ.get('FEATURE_MAP_TYPE', 'ZZ')           # Feature map type: 'ZZ' (ZZFeatureMap), 'Pauli_XX' (ZX+XX entanglement), 'Pauli_ZZ_XX' (ZZ+XX), 'Pauli_full' (Z+ZX+XX)
FEATURE_MODE = os.environ.get('FEATURE_MODE', 'current')              # Feature pipeline: 'current' (baseline TF-IDF+18ling→PCA), 'hybrid' (char-ngram PCA(8)+top-6 MI stylometric), 'stylometric_direct' (24 features → MI top-N, no PCA)

# GPU Configuration (Static - system behavior)
# Auto-Slicing Configuration (Enhanced environment-based)
AUTO_SLICE = os.environ.get("AUTO_SLICE", "False").lower() == "true"    # Automatically determine if slicing is needed
SLICING_THRESHOLD_GB = int(os.environ.get("SLICING_THRESHOLD_GB", 40)) # Memory threshold in GB for triggering slicing

AUTO_DETECT_GPU = True      # Automatically detect and use GPU if available
FORCE_GPU = False           # Force GPU usage (will fail if GPU not available)

# Output Configuration (Static - system behavior)
OUTPUT_DIR = 'QSVM_model'   # Directory to save results (when DYNAMIC_NAMING=False)
SAVE_MODELS = True          # Whether to save trained models
SAVE_RESULTS = True         # Whether to save comparison results
DYNAMIC_NAMING = True       # Dynamic folder naming based on parameters

# System Monitoring Configuration (Static - system behavior)
MONITOR_RESOURCES = True    # Enable CPU/Memory monitoring during training
# Recommended intervals:  5s for short debug runs (<30 min, captures fast spikes)
#                         10s good default (balances granularity vs overhead)
#                         30s for long production runs (hours; spikes last minutes anyway)
#                         60s for very long runs on shared HPC nodes (minimal footprint)
MONITORING_INTERVAL = float(os.environ.get('MONITORING_INTERVAL', '60'))
INCLUDE_GPU_INFO = True     # Include GPU information (if available)

# Quantum Training Monitoring Configuration (Enhanced environment-based)
ENABLE_QUANTUM_TRAINING_LOG = os.environ.get('ENABLE_QUANTUM_TRAINING_LOG', 'True').lower() == 'true'  # Enable detailed quantum training monitoring
QUANTUM_LOG_INTERVAL = float(os.environ.get('QUANTUM_LOG_INTERVAL', 30.0))           # Seconds between quantum training checkpoints
QUANTUM_CV_FOLDS = int(os.environ.get('QUANTUM_CV_FOLDS', 5))                       # Number of cross-validation folds for more data points
QUANTUM_LOG_FILENAME = 'quantum_model_training_time_log.csv'  # CSV log filename

# Batch Training Configuration (Ensemble)
QSVM_BATCH_TRAINING = os.environ.get('QSVM_BATCH_TRAINING', 'False').lower() == 'true'  # Enable batch (ensemble) training
QSVM_BATCH_SIZE = int(os.environ.get('QSVM_BATCH_SIZE', 5))                             # Number of batches (splits) for training

# t-SNE Configuration (Enhanced environment-based)
TSNE_REPORT = os.environ.get('TSNE_REPORT', 'True').lower() == 'true'               # Generate t-SNE data for later analysis (JSON report)
ENABLE_TSNE_VISUALIZATION = os.environ.get('ENABLE_TSNE_VISUALIZATION', 'True').lower() == 'true'  # Auto-generate PNG visualizations during training
TSNE_PERPLEXITY = int(os.environ.get('TSNE_PERPLEXITY', 30))                        # t-SNE perplexity parameter (5-50 recommended)
TSNE_LEARNING_RATE = int(os.environ.get('TSNE_LEARNING_RATE', 200))                 # t-SNE learning rate (10-1000 recommended)
TSNE_ITERATIONS = int(os.environ.get('TSNE_ITERATIONS', 1000))                      # Number of t-SNE iterations (250-1000 recommended)
TSNE_ANALYZE_SEPARATION = os.environ.get('TSNE_ANALYZE_SEPARATION', 'True').lower() == 'true'  # Enable separation quality analysis

# Memory Profiling
SAVE_VMHWM = os.environ.get('SAVE_VMHWM', 'True').lower() == 'true'  # Save kernel peak memory (VmHWM) to a per-run CSV for cross-configuration comparison

# =============================================================================
# PROVENANCE HELPERS
# =============================================================================
def get_software_versions():
    """Collect library versions and Git hash for reproducibility."""
    versions = {'python': sys.version}
    for pkg in ('numpy', 'scipy', 'sklearn', 'qiskit', 'qiskit_machine_learning',
                'qiskit_aer', 'pandas', 'psutil', 'matplotlib', 'nltk', 'joblib'):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, '__version__', 'installed')
        except ImportError:
            versions[pkg] = 'not installed'
    import subprocess as _sp
    try:
        versions['git_hash'] = _sp.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=_sp.DEVNULL, cwd=os.path.dirname(__file__) or '.'
        ).decode().strip()
    except Exception:
        versions['git_hash'] = 'unavailable'
    return versions

def get_threading_env():
    """Capture thread-control environment variables for numerical reproducibility."""
    keys = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'PYTHONUNBUFFERED']
    return {k: os.environ.get(k) for k in keys}

def get_vmhwm_mb():
    """Read the kernel high-water-mark (VmHWM) for the current process from /proc.
    This captures the true peak RSS including spikes between sampling intervals.
    Returns None on non-Linux platforms.
    """
    try:
        with open(f'/proc/{os.getpid()}/status') as f:
            for line in f:
                if line.startswith('VmHWM:'):
                    return round(int(line.split()[1]) / 1024, 1)  # kB → MB
    except Exception:
        pass
    return None

# =============================================================================
# PARAMETER VALIDATION
# =============================================================================
def determine_slicing_requirements(feature_dim, shots, training_samples, validation_samples, backend="cutensornet_cpu", auto_slice=True):
    """Determine if slicing is needed based on dimension, shots, and dataset size"""
    
    # If auto-slicing is disabled, return no slicing needed
    if not auto_slice:
        return {
            "slicing_needed": False,
            "optimal_slices": 0,
            "total_memory_gb": 0,
            "available_memory_gb": 0,
            "memory_usage_percent": 0,
            "server_type": "unknown",
            "server_name": "Auto-slicing disabled",
            "slicing_threshold_gb": 0,
            "memory_per_sample_mb": 0
        }
    
    # Special handling for StatevectorSimulator - use parallelism instead of memory management
    if "statevector" in backend.lower() and auto_slice:
        # Get CV folds from environment variable
        cv_folds = int(os.environ.get('QUANTUM_CV_FOLDS', 5))
        parallel_processes = cv_folds + 1  # CV folds + final model
        
        return {
            "slicing_needed": True,  # Repurposed as "parallelism needed"
            "optimal_slices": parallel_processes,  # Number of parallel processes
            "parallelism_mode": "statevector_cv_parallel",
            "cv_folds": cv_folds,
            "total_processes": parallel_processes,
            "memory_per_process_gb": 2.0,  # StatevectorSimulator is memory-efficient
            "total_memory_gb": parallel_processes * 2.0,
            "server_type": "cpu_parallel",
            "server_name": f"StatevectorSimulator with {parallel_processes} parallel processes",
            "slicing_threshold_gb": 0,  # Not applicable for parallelism mode
            "memory_per_sample_mb": 0,  # Not applicable for parallelism mode
            "memory_usage_percent": 0,  # Not applicable for parallelism mode
            "available_memory_gb": 0    # Not applicable for parallelism mode
        }
    
    # Original cuTensorNet and AER logic continues unchanged...
    # Backend-specific memory estimation
    if "aer" in backend.lower():
        # AER simulator memory estimation (based on actual observed usage)
        # AER uses significantly more memory than cuTensorNet
        base_8d_memory_mb = 8.0  # AER 8D/1024 uses ~8GB vs cuTensorNet's ~1GB
        memory_multiplier = 8.0  # AER uses ~8x more memory than cuTensorNet
    else:
        # cuTensorNet memory estimation (optimized)
        base_8d_memory_mb = 1.089  # Calibrated from actual 8D/1024 experiment with 5800 samples
        memory_multiplier = 1.0
    
    # Calculate memory per sample with exponential scaling (2^dimension)
    scaling_factor = 2 ** (feature_dim - 8)  # Relative to 8D
    memory_per_sample_mb = base_8d_memory_mb * scaling_factor * memory_multiplier
    
    # Calculate total memory needed
    total_samples = training_samples + validation_samples
    total_memory_gb = (memory_per_sample_mb * total_samples * shots) / (1024 * 1024)
    
    # Server-specific memory limits and recommendations
    server_configs = {
        "base": {"memory_gb": 112.8, "name": "Router/PC Server", "slicing_threshold_gb": 15},
        "cpu": {"memory_gb": 244, "name": "CPU Node (cn###)", "slicing_threshold_gb": 40},  # Increased to 40GB
        "gpu": {"memory_gb": 320, "name": "GPU Node (gn##)", "slicing_threshold_gb": 32}
    }
    
    # Detect current server (simplified detection)
    import psutil
    current_memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Detect current server based on hostname
    import socket
    hostname = socket.gethostname().lower()
    
    # Determine server type based on hostname pattern
    if hostname.startswith('gn'):
        server_type = "gpu"  # GPU servers (gn##) - high performance
    elif hostname.startswith('cn'):
        server_type = "cpu"  # CPU servers (cn###) - compute nodes
    else:
        server_type = "base"  # Router/lightweight servers (sun01) or personal PC
    
    server_config = server_configs.get(server_type, server_configs["base"])
    available_memory_gb = server_config["memory_gb"]
    
    # Determine if slicing is needed based on optimized logic
    slicing_needed = total_memory_gb > server_config["slicing_threshold_gb"]
    
    # Apply specific rules based on user requirements (these override the threshold)
    if "aer" in backend.lower():
        # AER simulator: More aggressive slicing due to higher memory usage
        if feature_dim >= 6 and shots >= 1024:
            slicing_needed = True  # AER 6D/1024+ requires slicing
        elif feature_dim >= 8 and shots >= 512:
            slicing_needed = True  # AER 8D/512+ requires slicing
    else:
        # cuTensorNet: Original rules (only when auto-slicing is enabled)
        if auto_slice:
            if feature_dim == 8 and shots >= 2048:
                slicing_needed = True  # 8D/2048+ requires slicing (regardless of threshold)
            elif feature_dim == 12 and shots >= 2048:
                slicing_needed = True  # 12D/2048+ requires slicing (regardless of threshold)
    
    # Calculate optimal slice count based on optimized strategy
    optimal_slices = None
    if slicing_needed:
        if "aer" in backend.lower():
            # AER simulator: More aggressive slicing
            if feature_dim <= 8:
                if shots <= 1024:
                    optimal_slices = 8
                else:  # shots >= 2048
                    optimal_slices = 16
            else:  # feature_dim >= 10
                optimal_slices = 16
        else:
            # cuTensorNet: Original strategy
            if feature_dim <= 12:
                if shots <= 2048:
                    optimal_slices = 8
                else:  # shots >= 4096
                    optimal_slices = 16
            else:  # feature_dim >= 14
                optimal_slices = 16
    
    # Special handling for very high dimensions (18D+) - only when auto-slicing is enabled
    if auto_slice and feature_dim >= 18:
        slicing_needed = True
        optimal_slices = 16
    
    return {
        "slicing_needed": slicing_needed,
        "optimal_slices": optimal_slices,
        "total_memory_gb": round(total_memory_gb, 2),
        "available_memory_gb": available_memory_gb,
        "memory_usage_percent": round((total_memory_gb / available_memory_gb) * 100, 1),
        "server_type": server_type,
        "server_name": server_config["name"],
        "slicing_threshold_gb": server_config["slicing_threshold_gb"],
        "memory_per_sample_mb": round(memory_per_sample_mb, 3)
    }



def validate_parameters():
    """Validate configuration parameters and provide warnings"""
    warnings = []
    
    # Validate training samples
    if TRAINING_SAMPLES > 4640:
        warnings.append(f"⚠️  TRAINING_SAMPLES ({TRAINING_SAMPLES}) exceeds maximum (4640)")
    if VALIDATION_SAMPLES > 1160:
        warnings.append(f"⚠️  VALIDATION_SAMPLES ({VALIDATION_SAMPLES}) exceeds maximum (1160)")
    
    # Validate quantum parameters
    if QUANTUM_FEATURE_DIM not in [2, 4, 6, 8, 10, 12, 14, 16]:
        warnings.append(f"⚠️  QUANTUM_FEATURE_DIM ({QUANTUM_FEATURE_DIM}) not in recommended range [2, 4, 6, 8, 10, 12, 14, 16]")
    if QUANTUM_SHOTS < 100 or QUANTUM_SHOTS > 10000:
        warnings.append(f"⚠️  QUANTUM_SHOTS ({QUANTUM_SHOTS}) not in recommended range [100, 10000]")
    if FEATURE_MAP_REPS not in [1, 2, 3]:
        warnings.append(f"⚠️  FEATURE_MAP_REPS ({FEATURE_MAP_REPS}) outside recommended range [1, 2, 3]")
    if SLICING_THRESHOLD_GB < 1 or SLICING_THRESHOLD_GB > 100:
        warnings.append(f"⚠️  SLICING_THRESHOLD_GB ({SLICING_THRESHOLD_GB}) not in recommended range [1, 100]")
    
    # Validate t-SNE parameters
    if TSNE_PERPLEXITY < 5 or TSNE_PERPLEXITY > 50:
        warnings.append(f"⚠️  TSNE_PERPLEXITY ({TSNE_PERPLEXITY}) not in recommended range [5, 50]")
    if TSNE_ITERATIONS < 250 or TSNE_ITERATIONS > 2000:
        warnings.append(f"⚠️  TSNE_ITERATIONS ({TSNE_ITERATIONS}) not in recommended range [250, 2000]")
    
    return warnings

def print_parameter_summary():
    """Print a summary of current parameter configuration"""
    print("📋 PARAMETER CONFIGURATION SUMMARY")
    print("=" * 50)
    print(f"🔢 Training: {TRAINING_SAMPLES} samples, {VALIDATION_SAMPLES} validation")
    print(f"⚛️  Quantum: {QUANTUM_FEATURE_DIM}D, {QUANTUM_SHOTS} shots, {FEATURE_MAP_REPS} reps")
    print(f"⚖️  Classical SVM C: {SVM_C} | Quantum SVM C: {QSVM_C}")
    print(f"🖥️  Backend: {QUANTUM_BACKEND}")
    if QUANTUM_BACKEND in ['qiskit_statevector_deterministic', 'statevector_deterministic']:
        if STATEVECTOR_SHOTS is None:
            print(f"🎯 Statevector kernel: EXACT (noiseless, no shot noise)")
        else:
            print(f"🎯 Statevector kernel: shot-noise emulation ({STATEVECTOR_SHOTS} shots)")
    print(f"📊 Monitoring: {'Enabled' if MONITOR_RESOURCES else 'Disabled'} ({MONITORING_INTERVAL}s)")
    print(f"🔬 CV Folds: {QUANTUM_CV_FOLDS}")
    print(f"🔄 Batch Training: {'Enabled' if QSVM_BATCH_TRAINING else 'Disabled'} (Splits: {QSVM_BATCH_SIZE})")
    print(f"🔧 Auto-Slicing: {'Enabled' if AUTO_SLICE else 'Disabled'} (threshold: {SLICING_THRESHOLD_GB}GB)")
    print(f"📐 Quantum normalization: [0, 2π] fitted on training data only")
    print(f"🎨 t-SNE: {'Enabled' if TSNE_REPORT else 'Disabled'} (perplexity: {TSNE_PERPLEXITY})")
    print(f"💾 Output: {'Dynamic' if DYNAMIC_NAMING else 'Static'} naming")
    print(f"🧠 VmHWM profile: {'Enabled' if SAVE_VMHWM else 'Disabled'}")
    print()

# =============================================================================

def _shots_label():
    """Consistent shots label for dynamic naming and logs."""
    if QUANTUM_BACKEND in ['qiskit_statevector_deterministic', 'statevector_deterministic']:
        return 'exact' if STATEVECTOR_SHOTS is None else f'sv{STATEVECTOR_SHOTS}'
    return str(QUANTUM_SHOTS)

def get_output_directory():
    """Generate output directory name based on configuration"""
    if not DYNAMIC_NAMING:
        return OUTPUT_DIR
    
    c_str = f"Cs{SVM_C:g}_Cq{QSVM_C:g}"
    fm_suffix   = f"-{FEATURE_MAP_TYPE}" if FEATURE_MAP_TYPE != 'ZZ' else ""
    reps_suffix = f"-r{FEATURE_MAP_REPS}" if FEATURE_MAP_REPS != 2 else ""
    mode_suffix = f"-{FEATURE_MODE}" if FEATURE_MODE != 'current' else ""
    dynamic_name = f"QSVM_model_{QUANTUM_FEATURE_DIM}-{_shots_label()}-{TRAINING_SAMPLES}-{VALIDATION_SAMPLES}-{c_str}{fm_suffix}{reps_suffix}{mode_suffix}"
    return dynamic_name

def get_dynamic_filename(base_name):
    """Generate dynamic filename based on configuration"""
    if not DYNAMIC_NAMING:
        return base_name
    
    # Extract file extension
    name_parts = base_name.split('.')
    if len(name_parts) > 1:
        extension = name_parts[-1]
        name_without_ext = '.'.join(name_parts[:-1])
    else:
        extension = ''
    c_str = f"Cs{SVM_C:g}_Cq{QSVM_C:g}"
    dynamic_name = f"{name_without_ext}_{QUANTUM_FEATURE_DIM}-{_shots_label()}-{TRAINING_SAMPLES}-{VALIDATION_SAMPLES}-{c_str}"
    if extension:
        return f"{dynamic_name}.{extension}"
    else:
        return dynamic_name
def calculate_model_association(feature_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                              X_val: np.ndarray, y_val: np.ndarray, feature_idx: int) -> Dict[str, Any]:
    """
    Calculate which model (Gemma vs Qwen) is associated with a specific feature
    
    Args:
        feature_name: Name of the feature
        X_train: Training features
        y_train: Training labels (0=Gemma, 1=Qwen)
        X_val: Validation features
        y_val: Validation labels (0=Gemma, 1=Qwen)
        feature_idx: Index of the feature in the feature matrix
        
    Returns:
        Dictionary with model association information
    """
    # Combine training and validation data
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.hstack([y_train, y_val])
    
    # Get feature values for this specific feature
    feature_values = X_combined[:, feature_idx]
    
    # Calculate statistics for each class
    gemma_mask = y_combined == 0
    qwen_mask = y_combined == 1
    
    gemma_values = feature_values[gemma_mask]
    qwen_values = feature_values[qwen_mask]
    
    # Calculate mean values for each model
    gemma_mean = np.mean(gemma_values) if len(gemma_values) > 0 else 0
    qwen_mean = np.mean(qwen_values) if len(qwen_values) > 0 else 0
    
    # Calculate frequency of non-zero values (for binary features)
    gemma_freq = np.mean(gemma_values > 0) if len(gemma_values) > 0 else 0
    qwen_freq = np.mean(qwen_values > 0) if len(qwen_values) > 0 else 0
    
    # Determine which model has higher values
    if gemma_mean > qwen_mean:
        preferred_model = "Gemma 3"
        confidence = abs(gemma_mean - qwen_mean) / max(gemma_mean + qwen_mean, 1e-8)
    else:
        preferred_model = "Qwen 2.5"
        confidence = abs(qwen_mean - gemma_mean) / max(gemma_mean + qwen_mean, 1e-8)
    
    # Normalize confidence to [0, 1]
    confidence = min(confidence, 1.0)
    
    return {
        "gemma_mean_value": float(gemma_mean),
        "qwen_mean_value": float(qwen_mean),
        "gemma_frequency": float(gemma_freq),
        "qwen_frequency": float(qwen_freq),
        "preferred_model": preferred_model,
        "confidence": float(confidence),
        "difference": float(abs(gemma_mean - qwen_mean))
    }

def extract_feature_importance(model, data_dict: Dict[str, Any], model_type: str, 
                             X_train: np.ndarray, y_train: np.ndarray, 
                             X_val: np.ndarray, y_val: np.ndarray, 
                             pca_model: Optional[Any] = None,
                             quantum_feature_scaler: Optional[Any] = None) -> Dict[str, Any]:
    """Extract comprehensive feature importance analysis with model associations"""
    try:
        from feature_importance import FeatureImportanceAnalyzer
        
        # Get feature names — skip_pca modes have pre-selected features
        skip_pca = data_dict.get('skip_pca', False)
        if skip_pca:
            mi_names = data_dict.get('mi_selected_feature_names')
            fn = data_dict.get('feature_names')
            if mi_names and len(mi_names) == X_train.shape[1]:
                all_feature_names = np.array(mi_names)
            elif fn is not None and len(fn) == X_train.shape[1]:
                all_feature_names = np.array(fn)
            else:
                all_feature_names = np.array([f'feature_{i}' for i in range(X_train.shape[1])])
            linguistic_features = list(all_feature_names)
        else:
            processor = data_dict.get('processor', None)
            if processor and hasattr(processor, 'vectorizer') and processor.vectorizer:
                tfidf_features = processor.vectorizer.get_feature_names_out()
                linguistic_features = processor.get_linguistic_feature_names()
                all_feature_names = np.concatenate([tfidf_features, linguistic_features])
            else:
                all_feature_names = np.array([f'feature_{i}' for i in range(X_train.shape[1])])
                linguistic_features = []
        
        analyzer = FeatureImportanceAnalyzer()
        
        if model_type == 'classical':
            # Classical SVM analysis
            importance_result = analyzer.analyze_classical_svm_importance(
                model, X_train, y_train, X_val, y_val, all_feature_names
            )
        else:
            # Quantum SVM analysis
            if pca_model is None:
                # skip_pca=True: features are already in final named form (MI-selected / hybrid)
                # Override all_feature_names with the direct MI-selected names so that
                # calculate_model_association uses the correct X_train column indices.
                mi_names = data_dict.get('mi_selected_feature_names')
                if mi_names and len(mi_names) == X_train.shape[1]:
                    all_feature_names = np.array(mi_names)
                else:
                    all_feature_names = np.array([f'feature_{i}' for i in range(X_train.shape[1])])
                n = len(all_feature_names)
                uniform = np.ones(n) / n * 100
                import pandas as pd
                importance_df = pd.DataFrame({
                    'feature': all_feature_names,
                    'pca_importance': uniform,
                    'permutation_importance': uniform,
                    'combined_importance': uniform,
                }).sort_values('combined_importance', ascending=False).reset_index(drop=True)
                analyzer.quantum_importance = importance_df
                importance_result = {
                    'importance_df': importance_df,
                    'top_100_features': importance_df,
                    'method': 'direct_mi_selection (no PCA)',
                    'total_features': n,
                    'quantum_dimensions': X_train.shape[1],
                }
            else:
                # Scale data before PCA if the quantum model used a StandardScaler
                X_train_for_pca = quantum_feature_scaler.transform(X_train) if quantum_feature_scaler is not None else X_train
                X_val_for_pca = quantum_feature_scaler.transform(X_val) if quantum_feature_scaler is not None else X_val
                X_train_quantum = pca_model.transform(X_train_for_pca)
                X_val_quantum = pca_model.transform(X_val_for_pca)
                
                importance_result = analyzer.analyze_quantum_svm_importance(
                    model, X_train_quantum, y_train, X_val_quantum, y_val, 
                    all_feature_names, pca_model
                )
        
        # Extract top 100 overall features with model associations
        top_100_df = importance_result.get('top_100_features', None)
        if top_100_df is not None and len(top_100_df) > 0:
            top_100_features = []
            for _, row in top_100_df.iterrows():
                feature_name = row['feature']
                importance_score = float(row.get('combined_importance', 0))
                
                # Find the feature index
                feature_idx = np.where(all_feature_names == feature_name)[0]
                if len(feature_idx) > 0:
                    feature_idx = feature_idx[0]
                    
                    # Calculate model association
                    model_association = calculate_model_association(
                        feature_name, X_train, y_train, X_val, y_val, feature_idx
                    )
                else:
                    # Fallback if feature not found
                    model_association = {
                        "gemma_mean_value": 0.0,
                        "qwen_mean_value": 0.0,
                        "gemma_frequency": 0.0,
                        "qwen_frequency": 0.0,
                        "preferred_model": "Unknown",
                        "confidence": 0.0,
                        "difference": 0.0
                    }
                
                feature_info = {
                    'feature': feature_name,
                    'importance_score': importance_score,
                    'importance_percentage': importance_score,
                    'model_association': model_association
                }
                top_100_features.append(feature_info)
        else:
            top_100_features = []
        
        # Separate linguistic features by category with model associations
        lexical_features = []
        syntactic_features = []
        stylistic_features = []
        
        if top_100_df is not None and len(linguistic_features) > 0:
            for _, row in top_100_df.iterrows():
                feature_name = row['feature']
                importance_score = float(row.get('combined_importance', 0))
                
                # Find the feature index
                feature_idx = np.where(all_feature_names == feature_name)[0]
                if len(feature_idx) > 0:
                    feature_idx = feature_idx[0]
                    
                    # Calculate model association
                    model_association = calculate_model_association(
                        feature_name, X_train, y_train, X_val, y_val, feature_idx
                    )
                else:
                    model_association = {
                        "gemma_mean_value": 0.0,
                        "qwen_mean_value": 0.0,
                        "gemma_frequency": 0.0,
                        "qwen_frequency": 0.0,
                        "preferred_model": "Unknown",
                        "confidence": 0.0,
                        "difference": 0.0
                    }
                
                feature_info = {
                    'feature': feature_name,
                    'importance_score': importance_score,
                    'model_association': model_association
                }
                
                if feature_name.startswith('lexical_'):
                    lexical_features.append(feature_info)
                elif feature_name.startswith('syntactic_'):
                    syntactic_features.append(feature_info)
                elif feature_name.startswith('stylistic_'):
                    stylistic_features.append(feature_info)
        
        return {
            'top_100_features': top_100_features[:100],
            'all_6_lexical_features': lexical_features[:6],
            'all_6_syntactic_features': syntactic_features[:6],
            'all_6_stylistic_features': stylistic_features[:6],
            'total_features_analyzed': len(all_feature_names),
            'analysis_method': importance_result.get('method', 'unknown')
        }
        
    except Exception as e:
        print(f"⚠️  Feature importance analysis failed: {e}")
        return {
            'top_100_features': [],
            'all_6_lexical_features': [],
            'all_6_syntactic_features': [],
            'all_6_stylistic_features': [],
            'total_features_analyzed': 0,
            'analysis_method': 'failed',
            'error': str(e)
        }

def detect_gpu_support():
    """Detect if GPU support is available for quantum simulation"""
    try:
        # Try to import CUDA support
        import cupy
        gpu_available = True
        gpu_info = f"CuPy detected - GPU acceleration available"
    except ImportError:
        gpu_available = False
        gpu_info = "CuPy not available - GPU acceleration not supported"
    
    # Also check for NVIDIA GPU using nvidia-ml-py or nvidia-smi
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            gpu_details = []
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 2:
                        gpu_details.append(f"{parts[0].strip()} ({parts[1].strip()}MB)")
            if gpu_details:
                gpu_info += f" - Detected: {', '.join(gpu_details)}"
    except Exception:
        pass

    return gpu_available, gpu_info

def get_optimal_backend():
    """Determine the optimal quantum backend based on configuration and hardware"""
    # For cuTensorNet backends, let create_optimal_quantum_svm handle the selection
    if 'cutensornet' in QUANTUM_BACKEND.lower():
        return QUANTUM_BACKEND, f"cuTensorNet backend requested: {QUANTUM_BACKEND}"
    
    # Legacy Qiskit AER backend selection
    if QUANTUM_BACKEND == 'aer_simulator_gpu' or FORCE_GPU:
        gpu_available, gpu_info = detect_gpu_support()
        if gpu_available or FORCE_GPU:
            return 'aer_simulator_gpu', gpu_info
        else:
            print(f"⚠️  GPU requested but not available: {gpu_info}")
            if FORCE_GPU:
                print("❌ FORCE_GPU=True but GPU not available - will attempt anyway")
                return 'aer_simulator_gpu', "GPU forced but may not be available"
            else:
                print("🔄 Falling back to CPU backend")
                return 'aer_simulator_cpu', "Fallback to CPU due to GPU unavailability"
    
    elif QUANTUM_BACKEND == 'aer_simulator_cpu':
        return 'aer_simulator_cpu', "CPU backend explicitly selected"
    

    elif QUANTUM_BACKEND == 'qiskit_statevector_deterministic':
        if STATEVECTOR_SHOTS is None:
            return 'qiskit_statevector_deterministic', "Exact statevector kernel (FidelityStatevectorKernel, noiseless)"
        else:
            return 'qiskit_statevector_deterministic', f"Statevector kernel with shot-noise emulation ({STATEVECTOR_SHOTS} shots)"
    elif QUANTUM_BACKEND == 'aer_simulator':
        # Default AER simulator - auto-detect GPU if enabled
        if AUTO_DETECT_GPU:
            gpu_available, gpu_info = detect_gpu_support()
            if gpu_available:
                print(f"🚀 Auto-detected GPU support: {gpu_info}")
                return 'aer_simulator_gpu', f"Auto-detected GPU: {gpu_info}"
            else:
                return 'aer_simulator_cpu', f"Auto-detection: {gpu_info}, using CPU"
        else:
            return 'aer_simulator_cpu', "Default AER simulator (CPU)"
    
    else:
        # Unknown backend - fallback to CPU
        print(f"⚠️  Unknown backend '{QUANTUM_BACKEND}', falling back to CPU")
        return 'aer_simulator_cpu', f"Unknown backend '{QUANTUM_BACKEND}', using CPU fallback"

class _RunningStats:
    """O(1) memory running min/max/sum/count accumulator."""
    __slots__ = ('count', 'total', 'min_val', 'max_val', 'first_ts', 'last_ts')
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.first_ts = None
        self.last_ts = None
    def update(self, value, ts=None):
        self.count += 1
        self.total += value
        if value < self.min_val:
            self.min_val = value
        if value > self.max_val:
            self.max_val = value
        if ts is not None:
            if self.first_ts is None:
                self.first_ts = ts
            self.last_ts = ts
    @property
    def avg(self):
        return self.total / self.count if self.count else 0.0
    @property
    def duration(self):
        if self.first_ts is not None and self.last_ts is not None:
            return self.last_ts - self.first_ts
        return 0.0


class SystemMonitor:
    """Monitor system resources continuously with per-stage tracking.
    
    Streams every sample to a CSV on disk as it is collected (crash-safe,
    O(1) memory).  Running min/max/avg statistics are kept in memory so
    get_statistics() and get_per_stage_stats() never need to re-read the file.
    
    Sampling interval is configurable via MONITORING_INTERVAL (default 10 s).
    """
    
    _CSV_COLUMNS = ['timestamp', 'elapsed_seconds', 'stage',
                    'cpu_percent', 'memory_percent', 'memory_used_gb',
                    'memory_available_gb', 'process_cpu_percent',
                    'process_memory_mb', 'process_tree_memory_mb',
                    'num_child_processes']
    
    _METRIC_KEYS = ('cpu_percent', 'memory_percent', 'memory_used_gb',
                    'process_cpu_percent', 'process_memory_mb',
                    'process_tree_memory_mb')
    
    def __init__(self, interval=1.0):
        self.interval = interval
        self.monitoring = False
        self.process = psutil.Process()
        self.monitor_thread = None
        self._current_stage = 'init'
        self._lock = threading.Lock()
        self.stage_snapshots = []
        
        # Latest sample values (for QuantumTrainingMonitor.log_checkpoint)
        self.data = {k: [] for k in ['cpu_percent', 'memory_percent', 'memory_used_gb',
                                      'memory_available_gb', 'timestamps',
                                      'process_cpu_percent', 'process_memory_mb',
                                      'process_tree_memory_mb', 'stage_labels']}
        self._latest = {}
        
        # Running statistics (global + per-stage) -- O(1) memory
        self._global = {k: _RunningStats() for k in self._METRIC_KEYS}
        self._per_stage = {}   # stage_name -> {metric -> _RunningStats}
        self._sample_count = 0
        self._first_ts = None
        self._last_ts = None
        self._max_parallel_procs = 1   # track peak parallelism across pipeline
        
        # Child-process CPU tracking (psutil needs a prior call to prime)
        self._known_children = {}  # pid -> psutil.Process

        # Cache static system properties and prime CPU meter
        self._n_logical_cpus = psutil.cpu_count(logical=True) or 1
        self.process.cpu_percent()  # prime so first real sample is non-zero

        # Streaming CSV state
        self._csv_file = None
        self._csv_path = None
    
    # ------------------------------------------------------------------
    # Static system / GPU info (unchanged)
    # ------------------------------------------------------------------
    def get_system_info(self):
        """Get static system information"""
        cpu_info = {
            'cpu_cores_physical': psutil.cpu_count(logical=False),
            'cpu_cores_logical': psutil.cpu_count(logical=True),
            'cpu_freq_max': psutil.cpu_freq().max if psutil.cpu_freq() else 'N/A',
            'cpu_freq_current': psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A'
        }
        memory = psutil.virtual_memory()
        memory_info = {
            'total_memory_gb': round(memory.total / (1024**3), 2),
            'available_memory_gb': round(memory.available / (1024**3), 2)
        }
        gpu_info = self.get_gpu_info() if INCLUDE_GPU_INFO else {}
        return {
            'cpu': cpu_info,
            'memory': memory_info,
            'gpu': gpu_info,
            'platform': f"{psutil.os.uname().sysname} {psutil.os.uname().release}"
        }
    
    def get_gpu_info(self):
        """Try to get GPU information"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpus = []
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 2:
                            gpus.append({
                                'name': parts[0].strip(),
                                'memory_mb': int(parts[1].strip())
                            })
                return {'gpus': gpus, 'gpu_available': True}
        except Exception:
            pass
        return {'gpu_available': False, 'note': 'No GPU detected or nvidia-smi not available'}
    
    def _get_process_tree_stats(self):
        """Collect memory, CPU, and child count for the whole process tree in one pass.

        Returns (main_mem_mb, tree_memory_mb, tree_cpu_pct, n_children).
        tree_cpu_pct sums cpu_percent() of main + every child whose CPU was
        already primed on a prior call (first-seen children are primed but
        contribute 0 to avoid psutil's meaningless first-call value).
        """
        try:
            main_mem = self.process.memory_info().rss
            main_cpu = self.process.cpu_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return (0.0, 0.0, 0.0, 0)

        total_mem = main_mem
        total_cpu = main_cpu
        current_children = {}

        for child in self.process.children(recursive=True):
            try:
                pid = child.pid
                current_children[pid] = child
                total_mem += child.memory_info().rss
                if pid in self._known_children:
                    total_cpu += child.cpu_percent()
                else:
                    child.cpu_percent()   # prime; don't count yet
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        self._known_children = current_children
        main_mem_mb = round(main_mem / (1024**2), 1)
        tree_mem_mb = round(total_mem / (1024**2), 1)
        return (main_mem_mb, tree_mem_mb, round(total_cpu, 1), len(current_children))
    
    # ------------------------------------------------------------------
    # Streaming CSV helpers
    # ------------------------------------------------------------------
    def _open_csv(self, output_dir, *, append=False):
        """Open (or reopen) the streaming CSV for incremental writes."""
        import csv
        os.makedirs(output_dir, exist_ok=True)
        if append and self._csv_path:
            path = self._csv_path
        else:
            fname = get_dynamic_filename('resource_timeseries.csv') if DYNAMIC_NAMING else 'resource_timeseries.csv'
            path = os.path.join(output_dir, fname)
        self._csv_path = path
        mode = 'a' if append else 'w'
        self._csv_file = open(self._csv_path, mode, newline='', buffering=1)
        self._csv_writer = csv.writer(self._csv_file)
        if not append:
            self._csv_writer.writerow(self._CSV_COLUMNS)
            self._csv_file.flush()
    
    def _write_csv_row(self, row):
        if self._csv_file and not self._csv_file.closed:
            self._csv_writer.writerow(row)
            # Flush every row -- negligible cost at 10 s intervals, guarantees crash safety
            self._csv_file.flush()
    
    def _close_csv(self):
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.close()
    
    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start_monitoring(self, output_dir=None):
        """Start resource monitoring in background thread.
        
        If output_dir is provided, the CSV opens there.  Otherwise it opens
        in a temporary location (cwd) so samples are never lost.  Call
        open_csv_to() later to move it to the real output directory.
        """
        if not MONITOR_RESOURCES:
            return
        csv_dir = output_dir if output_dir else os.getcwd()
        self._open_csv(csv_dir)
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def open_csv_to(self, output_dir):
        """Move the streaming CSV to a specific directory.
        
        Call this once the output directory is known (e.g. after temp dir
        creation).  The existing CSV is closed, moved, and reopened in
        append mode so no samples are lost.
        """
        import shutil
        self._close_csv()
        if self._csv_path and os.path.exists(self._csv_path):
            os.makedirs(output_dir, exist_ok=True)
            new_path = os.path.join(output_dir, os.path.basename(self._csv_path))
            shutil.move(self._csv_path, new_path)
            self._csv_path = new_path
            self._open_csv(output_dir, append=True)
        else:
            self._open_csv(output_dir)
    
    def relocate_csv(self, new_dir):
        """Move the streaming CSV to a new directory and reopen for continued writing."""
        import shutil
        self._close_csv()
        if self._csv_path and os.path.exists(self._csv_path):
            os.makedirs(new_dir, exist_ok=True)
            new_path = os.path.join(new_dir, os.path.basename(self._csv_path))
            shutil.move(self._csv_path, new_path)
            self._csv_path = new_path
            self._open_csv(new_dir, append=True)
    
    def stop_monitoring(self):
        """Stop resource monitoring and close the CSV."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        self._close_csv()
    
    # ------------------------------------------------------------------
    # Stage tracking
    # ------------------------------------------------------------------
    def mark_stage(self, stage_name: str):
        """Record a stage boundary with an instant resource snapshot."""
        with self._lock:
            prev_stage = self._current_stage
            self._current_stage = stage_name
        
        proc_mem, tree_mem, tree_cpu, n_children = self._get_process_tree_stats()
        snapshot = {
            'stage': stage_name,
            'prev_stage': prev_stage,
            'timestamp': time.time(),
            'timestamp_iso': datetime.now().isoformat(),
            'process_memory_mb': proc_mem,
            'process_tree_memory_mb': tree_mem,
            'system_memory_percent': psutil.virtual_memory().percent,
            'system_memory_used_gb': round((psutil.virtual_memory().total - psutil.virtual_memory().available) / (1024**3), 2),
            'cpu_percent': psutil.cpu_percent(interval=None),
            'num_child_processes': n_children,
            'vmhwm_mb': get_vmhwm_mb()
        }
        self.stage_snapshots.append(snapshot)
        return snapshot
    
    # ------------------------------------------------------------------
    # Background sampling loop
    # ------------------------------------------------------------------
    def _monitor_loop(self):
        """Background monitoring loop -- streams to CSV and updates running stats."""
        while self.monitoring:
            try:
                ts = time.time()
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                process_memory, tree_memory, tree_cpu, n_children = self._get_process_tree_stats()
                mem_used_gb = round((memory.total - memory.available) / (1024**3), 2)
                mem_avail_gb = round(memory.available / (1024**3), 2)
                
                n_procs = max(1, 1 + n_children)
                if n_procs > self._max_parallel_procs:
                    self._max_parallel_procs = n_procs
                
                # CPU: node-level utilization from the OS (0-100%, real, no math)
                # Memory: real total stacked across all parallel processes
                workload_cpu_pct = round(tree_cpu / self._n_logical_cpus, 2)
                
                with self._lock:
                    stage = self._current_stage
                
                # Keep latest values for QuantumTrainingMonitor.log_checkpoint
                self._latest = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': mem_used_gb,
                    'memory_available_gb': mem_avail_gb,
                    'process_cpu_percent': workload_cpu_pct,
                    'process_memory_mb': process_memory,
                    'process_tree_memory_mb': tree_memory
                }
                # Keep backward-compatible `.data` with only the latest element
                for k, v in self._latest.items():
                    self.data[k] = [v]
                self.data['timestamps'] = [ts]
                self.data['stage_labels'] = [stage]
                
                # CPU: workload % of total node capacity; memory: real totals
                sample = {'cpu_percent': cpu_percent, 'memory_percent': memory.percent,
                          'memory_used_gb': mem_used_gb,
                          'process_cpu_percent': workload_cpu_pct,
                          'process_memory_mb': process_memory,
                          'process_tree_memory_mb': tree_memory}
                for k, v in sample.items():
                    self._global[k].update(v, ts)
                
                # Update per-stage running statistics
                if stage not in self._per_stage:
                    self._per_stage[stage] = {k: _RunningStats() for k in self._METRIC_KEYS}
                for k, v in sample.items():
                    self._per_stage[stage][k].update(v, ts)
                
                self._sample_count += 1
                if self._first_ts is None:
                    self._first_ts = ts
                self._last_ts = ts
                
                # CSV: node-level CPU, raw memory totals, child count
                elapsed = ts - self._first_ts if self._first_ts else 0
                self._write_csv_row([
                    ts, round(elapsed, 2), stage,
                    cpu_percent, memory.percent, mem_used_gb,
                    mem_avail_gb, workload_cpu_pct, process_memory, tree_memory,
                    n_children
                ])
                
                time.sleep(self.interval)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Transient: a child process vanished mid-sample; keep monitoring
                time.sleep(self.interval)
            except Exception:
                break
    
    # ------------------------------------------------------------------
    # Statistics (computed from running accumulators, NOT from stored lists)
    # ------------------------------------------------------------------
    def get_statistics(self):
        """Get monitoring statistics (global summary) -- O(1), no list traversal."""
        if self._sample_count == 0:
            return {}
        g = self._global
        return {
            'cpu_usage': {
                'avg_percent': round(g['cpu_percent'].avg, 2),
                'max_percent': round(g['cpu_percent'].max_val, 2),
                'min_percent': round(g['cpu_percent'].min_val, 2)
            },
            'memory_usage': {
                'avg_percent': round(g['memory_percent'].avg, 2),
                'max_percent': round(g['memory_percent'].max_val, 2),
                'peak_used_gb': round(g['memory_used_gb'].max_val, 2)
            },
            'process_usage': {
                'avg_cpu_percent': round(g['process_cpu_percent'].avg, 2),
                'max_cpu_percent': round(g['process_cpu_percent'].max_val, 2),
                'peak_memory_mb': round(g['process_memory_mb'].max_val, 1),
                'peak_tree_memory_mb': round(g['process_tree_memory_mb'].max_val, 1),
                'max_parallel_processes': self._max_parallel_procs,
                'note': 'CPU is workload fraction of total node capacity (0-100%); tree memory is real total stacked across all parallel processes'
            },
            'monitoring_duration_seconds': round((self._last_ts - self._first_ts) if self._first_ts and self._last_ts else 0, 2),
            'samples_collected': self._sample_count
        }
    
    def get_per_stage_stats(self):
        """Compute peak/avg memory and CPU per pipeline stage -- O(stages), no list traversal."""
        if not self._per_stage:
            return {}
        per_stage = {}
        for stage, metrics in self._per_stage.items():
            per_stage[stage] = {
                'samples': metrics['cpu_percent'].count,
                'duration_seconds': round(metrics['cpu_percent'].duration, 2),
                'process_memory_peak_mb': round(metrics['process_memory_mb'].max_val, 1),
                'process_memory_avg_mb': round(metrics['process_memory_mb'].avg, 1),
                'tree_memory_peak_mb': round(metrics['process_tree_memory_mb'].max_val, 1),
                'system_memory_peak_gb': round(metrics['memory_used_gb'].max_val, 2),
                'cpu_avg_percent': round(metrics['cpu_percent'].avg, 2),
                'cpu_max_percent': round(metrics['cpu_percent'].max_val, 2)
            }
        return per_stage

class QuantumTrainingMonitor:
    """Monitor quantum training progress with detailed logging.
    
    Accepts an external SystemMonitor reference so it reuses the pipeline-wide
    sampling thread instead of creating a redundant one.
    """
    
    def __init__(self, log_interval=10.0, cv_folds=5, filename='quantum_model_training_time_log.csv',
                 system_monitor=None):
        self.log_interval = log_interval
        self.cv_folds = cv_folds
        self.filename = filename
        self.log_data = []
        self.start_time = None
        self.system_monitor = system_monitor
        self._owns_monitor = system_monitor is None
        if self._owns_monitor:
            self.system_monitor = SystemMonitor(interval=MONITORING_INTERVAL)
        self.last_checkpoint_time = None
        self.fold_start_times = {}
        
    def start_monitoring(self):
        """Start monitoring quantum training"""
        self.start_time = time.time()
        if self._owns_monitor:
            self.system_monitor.start_monitoring()
        self.log_data = []
        self.last_checkpoint_time = self.start_time
        self.fold_start_times = {}
        
    def stop_monitoring(self):
        """Stop monitoring quantum training"""
        if self._owns_monitor:
            self.system_monitor.stop_monitoring()
        
    def log_checkpoint(self, step, fold, train_acc, val_acc, circuit_time=None, prediction_time=None, additional_metrics=None):
        """Log a training checkpoint with enhanced timing information"""
        if self.start_time is None:
            return
            
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        timestamp = datetime.now().isoformat()
        
        # Calculate per-step timing
        step_duration = current_time - self.last_checkpoint_time
        
        # Track fold timing
        fold_key = f"fold_{fold}"
        if fold_key not in self.fold_start_times:
            self.fold_start_times[fold_key] = current_time
        
        fold_duration = current_time - self.fold_start_times[fold_key]
        
        # Get current resource usage from the shared pipeline-wide monitor
        resource_stats = {}
        if self.system_monitor.data['cpu_percent']:
            resource_stats = {
                'cpu_percent': self.system_monitor.data['cpu_percent'][-1],
                'memory_percent': self.system_monitor.data['memory_percent'][-1],
                'memory_used_gb': self.system_monitor.data['memory_used_gb'][-1],
                'process_cpu_percent': self.system_monitor.data['process_cpu_percent'][-1],
                'process_memory_mb': self.system_monitor.data['process_memory_mb'][-1],
                'process_tree_memory_mb': self.system_monitor.data['process_tree_memory_mb'][-1]
            }
        
        # Determine step type and calculate relevant timings
        is_in_progress = additional_metrics and additional_metrics.get('training_status') == 'in_progress'
        
        # Create log entry with enhanced timing
        log_entry = {
            'timestamp': timestamp,
            'elapsed_time_seconds': round(elapsed_time, 2),
            'step_duration_seconds': round(step_duration, 2),  # NEW: Time for this specific step
            'fold_duration_seconds': round(fold_duration, 2) if not is_in_progress else None,  # NEW: Total time for this fold
            'step': step,
            'cv_fold': fold,
            'train_accuracy': round(train_acc, 6) if train_acc is not None else None,
            'val_accuracy': round(val_acc, 6) if val_acc is not None else None,
            'circuit_time_seconds': round(circuit_time, 4) if circuit_time is not None else None,
            'prediction_time_seconds': round(prediction_time, 4) if prediction_time is not None else None,
            **resource_stats
        }
        
        # Add additional metrics if provided
        if additional_metrics:
            log_entry.update(additional_metrics)
            
            # Enhanced timing analysis for in-progress steps
            if is_in_progress:
                log_entry['monitoring_interval_actual'] = round(step_duration, 2)  # NEW: Actual monitoring interval
                log_entry['fold_duration_seconds'] = round(fold_duration, 2)  # Show current fold progress
            else:
                # For completed steps, calculate training efficiency
                training_time_this_fold = fold_duration
                if 'fold_training_elapsed' in additional_metrics:
                    monitoring_time = additional_metrics['fold_training_elapsed']
                    pure_training_time = training_time_this_fold - monitoring_time
                    log_entry['training_time_this_fold'] = round(pure_training_time, 2)  # NEW: Pure training time
                    log_entry['monitoring_overhead_seconds'] = round(monitoring_time, 2)  # NEW: Monitoring overhead
                    log_entry['training_efficiency'] = round(pure_training_time / training_time_this_fold * 100, 1) if training_time_this_fold > 0 else 0  # NEW: Efficiency percentage
            
        self.log_data.append(log_entry)
        
        # Update last checkpoint time
        self.last_checkpoint_time = current_time
        
    def save_log(self, output_dir):
        """Save the training log to CSV"""
        if not self.log_data:
            return (None, None)
            
        import pandas as pd
        
        # Create DataFrame
        df = pd.DataFrame(self.log_data)
        
        # Apply dynamic naming if enabled
        if DYNAMIC_NAMING:
            filename = get_dynamic_filename(self.filename)
        else:
            filename = self.filename
            
        # Save to output directory
        csv_path = os.path.join(output_dir, filename)
        df.to_csv(csv_path, index=False)
        
        return csv_path, filename
        
    def get_summary_stats(self):
        """Get summary statistics from the training log"""
        if not self.log_data:
            return {}
            
        import pandas as pd
        df = pd.DataFrame(self.log_data)
        
        summary = {
            'total_checkpoints': len(df),
            'total_folds': df['cv_fold'].nunique() if 'cv_fold' in df else 0,
            'avg_train_accuracy': df['train_accuracy'].mean() if 'train_accuracy' in df else None,
            'avg_val_accuracy': df['val_accuracy'].mean() if 'val_accuracy' in df else None,
            'best_val_accuracy': df['val_accuracy'].max() if 'val_accuracy' in df else None,
            'total_training_time': df['elapsed_time_seconds'].max() if 'elapsed_time_seconds' in df else None,
            'avg_cpu_usage': df['cpu_percent'].mean() if 'cpu_percent' in df else None,
            'peak_memory_usage': df['process_memory_mb'].max() if 'process_memory_mb' in df else None
        }
        
        return summary

def main():
    """Direct QSVM training - simple and reliable"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)  # Ensure we're in the correct directory
    
    # Validate parameters
    warnings = validate_parameters()
    if warnings:
        print("⚠️  Parameter warnings:")
        for warning in warnings:
            print(f"   - {warning}")
        print("Please check your environment variables or command line arguments.")
        print()
    
    # Print parameter summary
    print_parameter_summary()
    
    # Determine optimal backend
    optimal_backend, backend_info = get_optimal_backend()
    
    print("🚀 Direct QSVM Training")
    print("="*50)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Working Directory: {script_dir}")
    print(f"📊 Data File: {DATA_FILE}")
    print(f"🔢 Training Samples: {TRAINING_SAMPLES}")
    print(f"🔢 Validation Samples: {VALIDATION_SAMPLES}")
    print(f"⚛️  Quantum Feature Dim: {QUANTUM_FEATURE_DIM}")
    print(f"⚛️  Quantum Shots: {QUANTUM_SHOTS}")
    print(f"⚛️  Feature Map Reps: {FEATURE_MAP_REPS}")
    print(f"⚖️  Classical SVM C: {SVM_C}")
    print(f"⚖️  Quantum SVM C: {QSVM_C}")
    print(f"🖥️  Quantum Backend: {optimal_backend}")
    print(f"📈 Resource Monitoring: {'Enabled' if MONITOR_RESOURCES else 'Disabled'}")
    print(f"📁 Output Directory: {get_output_directory()} ({'Dynamic' if DYNAMIC_NAMING else 'Static'})")
    print(f"🎮 Backend Info: {backend_info}")
    print()
    
    # Pipeline-wide resource monitor (runs from start to finish)
    monitor = SystemMonitor(interval=MONITORING_INTERVAL)
    system_info = monitor.get_system_info()
    monitor.start_monitoring()
    monitor.mark_stage('system_init')
    
    # Display system information
    print("🖥️  System Information:")
    print(f"   Platform: {system_info['platform']}")
    print(f"   CPU Cores: {system_info['cpu']['cpu_cores_physical']} physical, {system_info['cpu']['cpu_cores_logical']} logical")
    print(f"   Total Memory: {system_info['memory']['total_memory_gb']} GB")
    if system_info['gpu'].get('gpu_available'):
        for i, gpu in enumerate(system_info['gpu']['gpus']):
            print(f"   GPU {i}: {gpu['name']} ({gpu['memory_mb']} MB)")
    else:
        print(f"   GPU: {system_info['gpu'].get('note', 'Not available')}")
    print()
    
    # Track all stage timings for the JSON
    stage_timings = {}
    pipeline_start_time = time.time()
    
    try:
        # Load and process data using original files
        monitor.mark_stage('data_loading')
        data_load_start = time.time()
        print("📊 Loading data...")
        from data_processor import DataProcessor
        processor = DataProcessor()
        
        # Check if data file exists
        if not os.path.exists(DATA_FILE):
            print(f"❌ Error: Data file '{DATA_FILE}' not found in {script_dir}")
            sys.exit(1)
        
        df = processor.load_csv_with_flexible_delimiter(DATA_FILE)
        print(f"✅ Loaded {len(df)} samples from dataset")

        # Subset early to requested total to avoid heavy full-dataset preprocessing during smoke tests
        total_needed = TRAINING_SAMPLES + VALIDATION_SAMPLES
        if total_needed > 0 and len(df) > total_needed:
            # Try to preserve class balance if label column exists
            try:
                import pandas as pd
                if 'label' in df.columns:
                    from sklearn.model_selection import train_test_split
                    df_small, _ = train_test_split(
                        df,
                        train_size=total_needed,
                        stratify=df['label'] if df['label'].nunique() > 1 else None,
                        random_state=42
                    )
                    df = df_small
                else:
                    df = df.head(total_needed)
                print(f"✂️  Subsetting to {len(df)} samples for requested TRAIN/VAL total ({TRAINING_SAMPLES}+{VALIDATION_SAMPLES})")
            except Exception as _subset_err:
                df = df.head(total_needed)
                print(f"✂️  Subsetting to first {len(df)} samples (stratified split unavailable)")

        stage_timings['data_loading'] = time.time() - data_load_start
        
        # Process data
        print("🔄 Processing data...")
        monitor.mark_stage('preprocessing')
        preprocess_start = time.time()
        data_dict = processor.load_and_preprocess(df)
        stage_timings['preprocessing'] = time.time() - preprocess_start
        print(f"✅ Preprocessing completed in {stage_timings['preprocessing']:.2f}s")
        # Add processor to data_dict for feature importance analysis
        data_dict['processor'] = processor
        print(f"✅ Processed: {len(data_dict['X_train'])} train, {len(data_dict['X_val'])} validation samples")
        
        # Use configured sample sizes
        X_train = data_dict['X_train'][:TRAINING_SAMPLES]
        y_train = data_dict['y_train'][:TRAINING_SAMPLES]
        X_val = data_dict['X_val'][:VALIDATION_SAMPLES]
        y_val = data_dict['y_val'][:VALIDATION_SAMPLES]
        
        print(f"📈 Using {len(X_train)} training samples, {len(X_val)} validation samples")
        print()
        
        # Train Classical SVM first (fast baseline)
        monitor.mark_stage('classical_svm_training')
        print("🔬 Training Classical SVM...")
        
        from classical_svm import ClassicalSVM
        csvm = ClassicalSVM(C=SVM_C)
        classical_start_time = time.time()
        classical_results = csvm.train_and_evaluate(X_train, X_val, y_train, y_val)
        classical_end_time = time.time()
        classical_total_time = classical_end_time - classical_start_time
        stage_timings['classical_svm_training'] = classical_total_time
        
        classical_monitoring_stats = monitor.get_statistics()
        
        # Add the actual model object to results
        classical_results['model'] = csvm.model
        
        monitor.mark_stage('classical_feature_importance')
        _fi_start = time.time()
        classical_feature_importance = extract_feature_importance(
            csvm.model, data_dict, 'classical', X_train, y_train, X_val, y_val
        )
        stage_timings['classical_feature_importance'] = time.time() - _fi_start
        
        # Extract per-class metrics for Gemma and Qwen
        classical_class_metrics = {}
        if 'classification_report' in classical_results and classical_results['classification_report']:
            class_report = classical_results['classification_report']
            # Get label encoder classes for mapping
            label_encoder_classes = data_dict.get('label_encoder', None)
            if label_encoder_classes and hasattr(label_encoder_classes, 'classes_'):
                class_names = label_encoder_classes.classes_
                # Map numeric keys to class names
                for i, class_name in enumerate(class_names):
                    class_key = str(i)  # Classification report uses string numeric keys
                    if class_key in class_report:
                        classical_class_metrics[f'{class_name.lower()}_precision'] = class_report[class_key]['precision']
                        classical_class_metrics[f'{class_name.lower()}_recall'] = class_report[class_key]['recall']
                        classical_class_metrics[f'{class_name.lower()}_f1_score'] = class_report[class_key]['f1-score']
                        classical_class_metrics[f'{class_name.lower()}_support'] = class_report[class_key]['support']
            else:
                # Fallback: try direct string matching
                for class_name in ['Gemma 3', 'Qwen 2.5']:
                    class_key = None
                    if class_name in class_report:
                        class_key = class_name
                    else:
                        # Look for string variations
                        for key in class_report.keys():
                            if str(key) == class_name:
                                class_key = key
                                break
                    
                    if class_key is not None:
                        classical_class_metrics[f'{class_name.lower()}_precision'] = class_report[class_key]['precision']
                        classical_class_metrics[f'{class_name.lower()}_recall'] = class_report[class_key]['recall']
                        classical_class_metrics[f'{class_name.lower()}_f1_score'] = class_report[class_key]['f1-score']
                        classical_class_metrics[f'{class_name.lower()}_support'] = class_report[class_key]['support']
        
        print("✅ Classical SVM Results:")
        classical_acc = classical_results.get('accuracy', None)
        classical_f1 = classical_results.get('f1_score', None)
        classical_time = classical_results.get('training_time', None)
        print(f"   Accuracy: {classical_acc:.4f}" if classical_acc is not None else "   Accuracy: N/A")
        print(f"   F1-Score: {classical_f1:.4f}" if classical_f1 is not None else "   F1-Score: N/A")
        print(f"   Training Time: {classical_time:.2f}s" if classical_time is not None else "   Training Time: N/A")
        if MONITOR_RESOURCES and classical_monitoring_stats:
            print(f"   Peak CPU (pipeline so far): {classical_monitoring_stats['cpu_usage']['max_percent']}%")
            print(f"   Peak Memory (pipeline so far): {classical_monitoring_stats['process_usage']['peak_memory_mb']} MB (process), {classical_monitoring_stats['process_usage'].get('peak_tree_memory_mb', 'N/A')} MB (tree/proc)")
        print()
        
        # Train Quantum SVM
        print("⚛️  Training Quantum SVM...")
        
        # Initialize quantum training monitor
        quantum_monitor = None
        if ENABLE_QUANTUM_TRAINING_LOG:
            quantum_monitor = QuantumTrainingMonitor(
                log_interval=QUANTUM_LOG_INTERVAL,
                cv_folds=QUANTUM_CV_FOLDS,
                filename=QUANTUM_LOG_FILENAME,
                system_monitor=monitor
            )
            quantum_monitor.start_monitoring()
            print(f"📊 Quantum training monitoring enabled: {QUANTUM_CV_FOLDS} CV folds, {QUANTUM_LOG_INTERVAL}s intervals")
        
        monitor.mark_stage('quantum_svm_training')
        
        from quantum_svm import create_optimal_quantum_svm
        # Determine slicing requirements if auto-slice is enabled
        slicing_info = None
        if AUTO_SLICE:
            slicing_info = determine_slicing_requirements(
                feature_dim=QUANTUM_FEATURE_DIM,
                shots=QUANTUM_SHOTS,
                training_samples=TRAINING_SAMPLES,
                validation_samples=VALIDATION_SAMPLES,
                backend=optimal_backend,
                auto_slice=AUTO_SLICE
            )
            
            print("🔧 Auto-Slicing Analysis:")
            print(f"   Server: {slicing_info['server_name']} ({slicing_info['server_type']})")
            
            # Handle StatevectorSimulator parallelism mode
            if slicing_info.get('parallelism_mode') == 'statevector_cv_parallel':
                print(f"   Parallelism Mode: {slicing_info['parallelism_mode']}")
                print(f"   CV Folds: {slicing_info['cv_folds']}")
                print(f"   Total Processes: {slicing_info['total_processes']}")
                print(f"   Memory per process: {slicing_info['memory_per_process_gb']} GB")
                print(f"   Total memory needed: {slicing_info['total_memory_gb']} GB")
                
                if slicing_info["slicing_needed"]:
                    print(f"   ⚡ Parallelism enabled: {slicing_info['optimal_slices']} parallel processes")
                else:
                    print("   ✅ No parallelism needed - single process processing")
            else:
                # Original cuTensorNet/AER memory management display
                print(f"   Memory per sample: {slicing_info['memory_per_sample_mb']} MB")
                print(f"   Total memory needed: {slicing_info['total_memory_gb']} GB")
                print(f"   Available memory: {slicing_info['available_memory_gb']} GB")
                print(f"   Memory usage: {slicing_info['memory_usage_percent']}%")
                
                if slicing_info["slicing_needed"]:
                    print(f"   ⚠️  Slicing required: {slicing_info['optimal_slices']} slices")
                else:
                    print("   ✅ No slicing needed - single batch processing")
            print()
        else:
            print("🔧 Auto-Slicing: Disabled - No slicing analysis performed")
            print()

        qsvm = create_optimal_quantum_svm(
            feature_dim=QUANTUM_FEATURE_DIM,
            shots=QUANTUM_SHOTS,
            quantum_backend=optimal_backend,
            feature_map_reps=FEATURE_MAP_REPS,
            feature_map_type=FEATURE_MAP_TYPE,
            slicing_info=slicing_info if AUTO_SLICE else None,
            sample_size='full',
            batch_training=QSVM_BATCH_TRAINING,
            batch_size=QSVM_BATCH_SIZE,
            C=QSVM_C,
            statevector_shots=STATEVECTOR_SHOTS,
            skip_pca=data_dict.get('skip_pca', False),
        )
        
        # Capture the actual backend used (important for cuTensorNet which may fallback)
        actual_backend_used = getattr(qsvm, 'actual_backend', optimal_backend)
        
        # Update backend info if cuTensorNet was used
        if hasattr(qsvm, 'actual_backend') and 'cutensornet' in actual_backend_used.lower():
            gpu_available = getattr(qsvm, 'gpu_available', False)
            cutensornet_available = getattr(qsvm, 'cutensornet_available', False)
            backend_info = f"cuTensorNet backend: {actual_backend_used} (GPU: {gpu_available}, cuTensorNet: {cutensornet_available})"
        
        quantum_start_time = time.time()
        
        # Use CV if cv_folds > 1, otherwise direct training
        if QUANTUM_CV_FOLDS > 1:
            cv_monitor = quantum_monitor if (ENABLE_QUANTUM_TRAINING_LOG and quantum_monitor) else None
            print(f"🔬 Running {QUANTUM_CV_FOLDS}-fold cross-validation" +
                  (" with detailed monitoring..." if cv_monitor else "..."))
            quantum_results = qsvm.train_with_cv_monitoring(
                X_train, X_val, y_train, y_val, 
                cv_folds=QUANTUM_CV_FOLDS,
                monitor=cv_monitor
            )
        else:
            print(f"🚀 Direct training (no cross-validation, cv_folds={QUANTUM_CV_FOLDS})...")
            quantum_results = qsvm.train_and_evaluate(X_train, X_val, y_train, y_val)
        
        quantum_end_time = time.time()
        quantum_core_time = quantum_end_time - quantum_start_time
        stage_timings['quantum_svm_training'] = quantum_core_time
        
        # Mark start of post-processing
        post_processing_start_time = time.time()
        post_processing_time = 0.0  # Initialize - will be calculated at the end
        quantum_total_time = 0.0  # Initialize - will be calculated at the end
        
        quantum_monitoring_stats = monitor.get_statistics()
        
        if quantum_monitor:
            quantum_monitor.stop_monitoring()
        
        # Add the actual model object to results
        quantum_results['model'] = qsvm.qsvm_model
        
        monitor.mark_stage('quantum_feature_importance')
        _qfi_start = time.time()
        quantum_feature_importance = extract_feature_importance(
            qsvm.qsvm_model, data_dict, 'quantum', X_train, y_train, X_val, y_val, qsvm.pca,
            quantum_feature_scaler=getattr(qsvm, 'feature_scaler', None)
        )
        stage_timings['quantum_feature_importance'] = time.time() - _qfi_start
        
        # Extract per-class metrics for Gemma and Qwen
        quantum_class_metrics = {}
        if 'classification_report' in quantum_results and quantum_results['classification_report']:
            class_report = quantum_results['classification_report']
            # Get label encoder classes for mapping
            label_encoder_classes = data_dict.get('label_encoder', None)
            if label_encoder_classes and hasattr(label_encoder_classes, 'classes_'):
                class_names = label_encoder_classes.classes_
                # Map numeric keys to class names
                for i, class_name in enumerate(class_names):
                    class_key = str(i)  # Classification report uses string numeric keys
                    if class_key in class_report:
                        quantum_class_metrics[f'{class_name.lower()}_precision'] = class_report[class_key]['precision']
                        quantum_class_metrics[f'{class_name.lower()}_recall'] = class_report[class_key]['recall']
                        quantum_class_metrics[f'{class_name.lower()}_f1_score'] = class_report[class_key]['f1-score']
                        quantum_class_metrics[f'{class_name.lower()}_support'] = class_report[class_key]['support']
            else:
                # Fallback: try direct string matching
                for class_name in ['Gemma 3', 'Qwen 2.5']:
                    class_key = None
                    if class_name in class_report:
                        class_key = class_name
                    else:
                        # Look for string variations
                        for key in class_report.keys():
                            if str(key) == class_name:
                                class_key = key
                                break
                    
                    if class_key is not None:
                        quantum_class_metrics[f'{class_name.lower()}_precision'] = class_report[class_key]['precision']
                        quantum_class_metrics[f'{class_name.lower()}_recall'] = class_report[class_key]['recall']
                        quantum_class_metrics[f'{class_name.lower()}_f1_score'] = class_report[class_key]['f1-score']
                        quantum_class_metrics[f'{class_name.lower()}_support'] = class_report[class_key]['support']
        
        print("✅ Quantum SVM Results:")
        q_acc = quantum_results.get('accuracy', None)
        q_f1 = quantum_results.get('f1_score', None)
        q_time = quantum_results.get('training_time', None)
        q_circuit = quantum_results.get('circuit_training_time', None)
        print(f"   Accuracy: {q_acc:.4f}" if q_acc is not None else "   Accuracy: N/A")
        print(f"   F1-Score: {q_f1:.4f}" if q_f1 is not None else "   F1-Score: N/A")
        print(f"   Training Time: {q_time:.2f}s" if q_time is not None else "   Training Time: N/A")
        print(f"   Circuit Training: {q_circuit:.2f}s" if q_circuit is not None else "   Circuit Training: N/A")
        
        if QUANTUM_CV_FOLDS > 1 and 'final_model_time' in quantum_results:
            final_model_time = quantum_results.get('final_model_time', 0)
            total_time = quantum_results.get('training_time', 0)
            cv_overhead = total_time - final_model_time
            cv_factor = total_time / final_model_time if final_model_time > 0 else 0
            print(f"   Final Model Only: {final_model_time:.2f}s")
            print(f"   CV Overhead ({QUANTUM_CV_FOLDS} folds): {cv_overhead:.2f}s ({cv_factor:.1f}x total vs single model)")
        
        if MONITOR_RESOURCES and quantum_monitoring_stats:
            print(f"   Peak CPU (pipeline so far): {quantum_monitoring_stats['cpu_usage']['max_percent']}%")
            print(f"   Peak Memory (pipeline so far): {quantum_monitoring_stats['process_usage']['peak_memory_mb']} MB (process), {quantum_monitoring_stats['process_usage'].get('peak_tree_memory_mb', 'N/A')} MB (tree/proc)")
        print()
        
        # Comparison
        print("📊 Comparison:")
        print("-" * 30)
        classical_acc = classical_results.get('accuracy', 0)
        quantum_acc = quantum_results.get('accuracy', 0)
        classical_time = classical_results.get('training_time', 0)
        quantum_time = quantum_results.get('training_time', 0)
        
        print(f"Classical Accuracy: {classical_acc:.4f}")
        print(f"Quantum Accuracy:   {quantum_acc:.4f}")
        print(f"Accuracy Difference: {quantum_acc - classical_acc:+.4f}")
        print()
        
        # Show both total time and fair comparison
        print(f"Classical Time: {classical_time:.2f}s")
        print(f"Quantum Time (Total): {quantum_time:.2f}s")
        
        # Fair comparison using final model time if available (excludes CV fold overhead)
        if QUANTUM_CV_FOLDS > 1 and 'final_model_time' in quantum_results:
            quantum_fair_time = quantum_results.get('final_model_time', quantum_time)
            print(f"Quantum Time (Final model only): {quantum_fair_time:.2f}s")
            print(f"Fair Time Ratio: {quantum_fair_time/classical_time:.2f}x" if classical_time > 0 else "Fair Time Ratio: N/A")
            print(f"Total Time Ratio: {quantum_time/classical_time:.2f}x (incl. {QUANTUM_CV_FOLDS} CV folds)" if classical_time > 0 else "Total Time Ratio: N/A")
        else:
            print(f"Time Ratio: {quantum_time/classical_time:.2f}x" if classical_time > 0 else "Time Ratio: N/A")
        print()
        
        # Save results
        if SAVE_MODELS or SAVE_RESULTS:
            print("💾 Saving results...")
            # Always save to a unique temp directory first (safe for concurrent jobs)
            unique_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            proc_tag = f"pid{os.getpid()}"
            temp_output_dirname = f"{OUTPUT_DIR}_tmp_{unique_ts}_{proc_tag}"
            temp_output_path = os.path.join(script_dir, temp_output_dirname)
            os.makedirs(temp_output_path, exist_ok=True)
            
            # Start streaming resource CSV to temp dir (crash-safe from this point on)
            monitor.open_csv_to(temp_output_path)
            
            # Save models with dynamic naming
            saved_files = []

            # Deterministic-only caching (predictions and quantum features)
            precomp_classical_preds = None
            precomp_quantum_preds = None
            precomp_quantum_features = None
            classical_cache_fn = None
            quantum_cache_fn = None
            quantum_feats_fn = None
            is_deterministic_backend = optimal_backend in ['qiskit_statevector_deterministic', 'statevector_deterministic']
            if is_deterministic_backend:
                try:
                    # Classical predictions cache
                    if 'model' in classical_results:
                        classical_train_pred = classical_results['model'].predict(X_train)
                        classical_val_pred = classical_results['model'].predict(X_val)
                        # Optional decision scores or probabilities
                        classical_scores_train = None
                        classical_scores_val = None
                        try:
                            if hasattr(classical_results['model'], 'decision_function'):
                                classical_scores_train = classical_results['model'].decision_function(X_train)
                                classical_scores_val = classical_results['model'].decision_function(X_val)
                            elif hasattr(classical_results['model'], 'predict_proba'):
                                classical_scores_train = classical_results['model'].predict_proba(X_train)
                                classical_scores_val = classical_results['model'].predict_proba(X_val)
                        except Exception:
                            classical_scores_train = None
                            classical_scores_val = None

                        classical_cache_fn = get_dynamic_filename('cached_predictions_classical.npz')
                        np.savez_compressed(
                            os.path.join(temp_output_path, classical_cache_fn),
                            train_pred=classical_train_pred,
                            val_pred=classical_val_pred,
                            train_scores=classical_scores_train if classical_scores_train is not None else np.array([]),
                            val_scores=classical_scores_val if classical_scores_val is not None else np.array([])
                        )
                        saved_files.append(classical_cache_fn)
                        precomp_classical_preds = (classical_train_pred, classical_val_pred)

                    # Quantum features and predictions cache
                    _scaler_for_cache = getattr(qsvm, 'feature_scaler', getattr(qsvm, 'scaler', None))
                    _cache_ready = False
                    if hasattr(qsvm, 'pca') and qsvm.pca is not None and hasattr(qsvm, 'qsvm_model'):
                        # Current mode: scale → PCA → [0, 2π]
                        X_train_scaled = _scaler_for_cache.transform(X_train) if _scaler_for_cache is not None else X_train
                        X_val_scaled   = _scaler_for_cache.transform(X_val)   if _scaler_for_cache is not None else X_val
                        X_train_pca = qsvm.pca.transform(X_train_scaled)
                        X_val_pca   = qsvm.pca.transform(X_val_scaled)

                        if hasattr(qsvm, '_normalize_to_2pi'):
                            X_train_quantum = qsvm._normalize_to_2pi(X_train_pca)
                            X_val_quantum = qsvm._normalize_to_2pi(X_val_pca)
                        else:
                            q_min = getattr(qsvm, 'quant_norm_min', X_train_pca.min(axis=0))
                            q_max = getattr(qsvm, 'quant_norm_max', X_train_pca.max(axis=0))
                            q_range = q_max - q_min
                            q_range[q_range == 0] = 1
                            X_train_quantum = np.clip(2 * np.pi * (X_train_pca - q_min) / q_range, 0.0, 2 * np.pi)
                            X_val_quantum = np.clip(2 * np.pi * (X_val_pca - q_min) / q_range, 0.0, 2 * np.pi)
                        _cache_ready = True
                    elif getattr(qsvm, 'skip_pca', False) and hasattr(qsvm, 'qsvm_model'):
                        # skip_pca modes (hybrid / stylometric_direct): scale → [0, 2π] (no PCA layer)
                        X_train_scaled = _scaler_for_cache.transform(X_train) if _scaler_for_cache is not None else X_train
                        X_val_scaled   = _scaler_for_cache.transform(X_val)   if _scaler_for_cache is not None else X_val
                        if hasattr(qsvm, '_normalize_to_2pi'):
                            X_train_quantum = qsvm._normalize_to_2pi(X_train_scaled)
                            X_val_quantum   = qsvm._normalize_to_2pi(X_val_scaled)
                        else:
                            q_min = getattr(qsvm, 'quant_norm_min', X_train_scaled.min(axis=0))
                            q_max = getattr(qsvm, 'quant_norm_max', X_train_scaled.max(axis=0))
                            q_range = q_max - q_min
                            q_range[q_range == 0] = 1
                            X_train_quantum = np.clip(2 * np.pi * (X_train_scaled - q_min) / q_range, 0.0, 2 * np.pi)
                            X_val_quantum   = np.clip(2 * np.pi * (X_val_scaled   - q_min) / q_range, 0.0, 2 * np.pi)
                        _cache_ready = True

                    if _cache_ready:
                        # Save quantum features (shared by both PCA and skip_pca paths)
                        quantum_feats_fn = get_dynamic_filename('cached_quantum_features.npz')
                        np.savez_compressed(
                            os.path.join(temp_output_path, quantum_feats_fn),
                            X_train_quantum=X_train_quantum,
                            X_val_quantum=X_val_quantum
                        )
                        saved_files.append(quantum_feats_fn)
                        precomp_quantum_features = (X_train_quantum, X_val_quantum)

                        # Predict using trained QSVM model
                        try:
                            quantum_train_pred = qsvm.qsvm_model.predict(X_train_quantum)
                            quantum_val_pred = qsvm.qsvm_model.predict(X_val_quantum)
                            quantum_cache_fn = get_dynamic_filename('cached_predictions_quantum.npz')
                            np.savez_compressed(
                                os.path.join(temp_output_path, quantum_cache_fn),
                                train_pred=quantum_train_pred,
                                val_pred=quantum_val_pred
                            )
                            saved_files.append(quantum_cache_fn)
                            precomp_quantum_preds = (quantum_train_pred, quantum_val_pred)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"⚠️  Deterministic caching failed: {e}")
            if SAVE_MODELS:
                monitor.mark_stage('model_serialization')
                _serial_start = time.time()
                import joblib
                if 'model' in classical_results:
                    classical_filename = get_dynamic_filename('classical_model.joblib')
                    classical_model_path = os.path.join(temp_output_path, classical_filename)
                    joblib.dump(classical_results['model'], classical_model_path)
                    saved_files.append(classical_filename)
                    print(f"✅ Classical model saved as {classical_filename}")
                
                if 'model' in quantum_results:
                    quantum_filename = get_dynamic_filename('quantum_model.joblib')
                    quantum_model_path = os.path.join(temp_output_path, quantum_filename)
                    joblib.dump(quantum_results['model'], quantum_model_path)
                    saved_files.append(quantum_filename)
                    print(f"✅ Quantum model saved as {quantum_filename}")
            
            if SAVE_MODELS:
                stage_timings['model_serialization'] = time.time() - _serial_start
            
            # Save comparison results
            if SAVE_RESULTS:
                monitor.mark_stage('json_results_writing')
                _json_start = time.time()
                import json
                _actual_tfidf = 0
                _actual_ling = 0
                try:
                    if processor and getattr(processor, 'vectorizer', None) is not None:
                        _actual_tfidf = len(processor.vectorizer.get_feature_names_out())
                    if processor and hasattr(processor, 'get_linguistic_feature_names'):
                        _actual_ling = len(processor.get_linguistic_feature_names())
                except Exception:
                    pass
                _skip_pca     = data_dict.get('skip_pca', False)
                _mi_features  = data_dict.get('mi_selected_feature_names')
                _n_features   = data_dict['X_train'].shape[1]
                # Build mode-aware descriptions for dataset and preprocessing sections
                if FEATURE_MODE == 'hybrid':
                    _char_k = round(_n_features * 8 / 14)
                    _feat_breakdown = (f"{_n_features} total: char 3-5gram TF-IDF → PCA({_char_k}) + "
                                       f"MI top-{_n_features - _char_k} stylometric (skip_pca=True)")
                    _feat_method_short = 'Hybrid: char n-gram TF-IDF PCA + MI-selected stylometric'
                    _dim_reduction     = f'Internal char-PCA({_char_k}) + MI selection (no QSVM-level PCA)'
                    _quantum_preproc   = 'StandardScaler → [0, 2π] normalization (hybrid pipeline pre-reduces; no QSVM PCA layer)'
                elif FEATURE_MODE == 'stylometric_direct':
                    _feat_breakdown = (f"{_n_features} total: MI top-{_n_features} from 24 extended "
                                       f"stylometric features (skip_pca=True, 1 qubit = 1 named feature)")
                    _feat_method_short = 'Stylometric direct: 24 extended stylometric → MI top-N (no PCA)'
                    _dim_reduction     = f'MI feature selection → {_n_features} dims (no PCA, direct encoding)'
                    _quantum_preproc   = 'StandardScaler → [0, 2π] normalization (no PCA, direct stylometric encoding)'
                else:  # 'current' baseline
                    _feat_breakdown    = f"{_n_features} total ({_actual_tfidf} TF-IDF + {_actual_ling} linguistic)"
                    _feat_method_short = 'TF-IDF + Linguistic Features (Enhanced)'
                    _dim_reduction     = 'PCA (for quantum only)'
                    _quantum_preproc   = 'StandardScaler → PCA → [0, 2π] normalization (all fitted on training data only)'
                comparison_data = {
                    'experiment_configuration': {
                        'training_samples': len(X_train),
                        'validation_samples': len(X_val),
                        'training_samples_requested': TRAINING_SAMPLES,
                        'validation_samples_requested': VALIDATION_SAMPLES,
                        'feature_mode': FEATURE_MODE,
                        'data_file': DATA_FILE,
                        'random_state': 42,
                        'global_rng_seed': GLOBAL_SEED,
                        'monitoring_enabled': MONITOR_RESOURCES,
                        'monitoring_interval': MONITORING_INTERVAL,
                        'dynamic_naming': DYNAMIC_NAMING,
                        'output_directory': get_output_directory(),
                        'timestamp': datetime.now().isoformat(),
                    },
                    'software_versions': get_software_versions(),
                    'threading_environment': get_threading_env(),
                    'dataset_statistics': {
                        'total_samples_loaded': len(data_dict['X_train']) + len(data_dict['X_val']),
                        'total_features': _n_features,
                        'feature_mode': FEATURE_MODE,
                        'tfidf_max_features': getattr(processor, 'tfidf_params', {}).get('max_features', 'N/A (non-current mode)') if processor else 'N/A',
                        'tfidf_features_actual': _actual_tfidf if _actual_tfidf > 0 else ('N/A' if FEATURE_MODE != 'current' else 'unknown'),
                        'linguistic_features': _actual_ling if FEATURE_MODE == 'current' else len(_mi_features or []),
                        'feature_breakdown': _feat_breakdown,
                        'mi_selected_feature_names': _mi_features,
                        'train_samples_full': len(data_dict['X_train']),
                        'validation_samples_full': len(data_dict['X_val']),
                        'train_samples_used': len(X_train),
                        'validation_samples_used': len(X_val),
                        'feature_extraction_method': _feat_method_short,
                        'dimensionality_reduction': _dim_reduction,
                    },
                    'preprocessing_parameters': {
                        'feature_mode': FEATURE_MODE,
                        'feature_extraction_method': _feat_method_short,
                        'skip_pca': _skip_pca,
                        'mi_selected_features': _mi_features,
                        'tfidf_params': getattr(processor, 'tfidf_params', {}) if (processor and FEATURE_MODE == 'current') else {},
                        'linguistic_features_enabled': True,
                        'lexical_features': 'word length, vocabulary richness, text statistics (6 features)',
                        'syntactic_features': 'POS ratios, punctuation density, complexity scores (6 features)', 
                        'stylistic_features': 'formality patterns, contractions, punctuation usage (6 features)',
                        'total_linguistic_features': _actual_ling,
                        'feature_scaling': 'StandardScaler applied to all feature representations',
                        'quantum_preprocessing': _quantum_preproc,
                        'stratified_sampling': True
                    },
                    'classical_svm_hyperparameters': {
                        'algorithm': 'LinearSVM with Isotonic Calibration',
                        'C': SVM_C,
                        'C_env_var': 'SVM_C',
                        'svm_params': csvm.svm_params if hasattr(csvm, 'svm_params') else {},
                        'calibration_params': csvm.calibration_params if hasattr(csvm, 'calibration_params') else {}
                    },
                    'quantum_svm_hyperparameters': {
                        'algorithm': ('QSVC with FidelityStatevectorKernel' if is_deterministic_backend else 'QSVC with FidelityQuantumKernel') + (' (Ensemble)' if QSVM_BATCH_TRAINING else ''),
                        'backend_requested': QUANTUM_BACKEND,
                        'backend_used': actual_backend_used,
                        'backend_info': backend_info,
                        'gpu_acceleration': 'Enabled' if 'gpu' in actual_backend_used.lower() else 'Disabled',
                        'feature_mode': FEATURE_MODE,
                        'feature_map': FEATURE_MAP_TYPE,
                        'feature_map_reps': FEATURE_MAP_REPS,
                        'feature_map_entanglement': 'linear',
                        'skip_pca': data_dict.get('skip_pca', False),
                        'mi_selected_features': data_dict.get('mi_selected_feature_names'),
                        'kernel': 'FidelityStatevectorKernel' if is_deterministic_backend else 'FidelityQuantumKernel',
                        'kernel_mode': ('exact_noiseless' if STATEVECTOR_SHOTS is None else f'shot_noise_emulation_{STATEVECTOR_SHOTS}') if is_deterministic_backend else f'sampler_{QUANTUM_SHOTS}_shots',
                        'quantum_feature_dim': QUANTUM_FEATURE_DIM,
                        'quantum_shots': ('exact' if STATEVECTOR_SHOTS is None else STATEVECTOR_SHOTS) if is_deterministic_backend else QUANTUM_SHOTS,
                        'statevector_shots': STATEVECTOR_SHOTS if is_deterministic_backend else None,
                        'C': QSVM_C,
                        'C_env_var': 'QSVM_C',
                        'class_weight': 'balanced',
                        'pca_components': QUANTUM_FEATURE_DIM,
                        
                        # Batch Training Configuration (Ensemble)
                        'batch_training_enabled': QSVM_BATCH_TRAINING,
                        'batch_size': QSVM_BATCH_SIZE if QSVM_BATCH_TRAINING else None,
                        'ensemble_method': 'Majority Voting (classification) / Probability Averaging (predict_proba)' if QSVM_BATCH_TRAINING else None,
                        'batch_strategy': 'StratifiedKFold with balanced class distribution' if QSVM_BATCH_TRAINING else None,
                        'models_per_ensemble': QSVM_BATCH_SIZE if QSVM_BATCH_TRAINING else 1,
                        
                        # Slicing and Parallelism Configuration
                        "auto_slicing_enabled": AUTO_SLICE,
                        "slicing_threshold_gb": slicing_info["slicing_threshold_gb"] if slicing_info else SLICING_THRESHOLD_GB,
                        "slicing_used": slicing_info["slicing_needed"] if slicing_info else False,
                        "slicing_slices": slicing_info["optimal_slices"] if slicing_info and slicing_info["slicing_needed"] else 0,
                        "slicing_memory_gb": slicing_info["total_memory_gb"] if slicing_info else 0,
                        "slicing_memory_percent": slicing_info["memory_usage_percent"] if slicing_info else 0,
                        "slicing_server": slicing_info["server_name"] if slicing_info else "Unknown",
                        "parallelism_mode": slicing_info.get("parallelism_mode") if slicing_info else None,
                        "parallel_cv_folds": slicing_info.get("cv_folds") if slicing_info else None,
                        "total_processes": slicing_info.get("total_processes") if slicing_info else None,
                        "cv_folds": QUANTUM_CV_FOLDS if QUANTUM_CV_FOLDS > 1 else None,

                        'pca_random_state': 42,
                        'feature_normalization': '[0, 2π] range (fitted on training data, applied to both splits)'
                    },
                    'system_info': system_info,
                    'classical': {
                        'description': 'Classical SVM Results: LinearSVM with Isotonic Calibration. Overall metrics represent validation set performance (generalization). Training metrics help detect overfitting. Per-class metrics show individual Gemma and Qwen performance. Timing metrics provide computational analysis.',
                        
                        # Overall Performance (Validation-based) - Main scientific metrics
                        'accuracy': float(classical_acc),                    # Overall accuracy on validation set
                        'precision': float(classical_results.get('precision', 0)),        # Overall precision on validation set  
                        'recall': float(classical_results.get('recall', 0)),              # Overall recall on validation set
                        'f1_score': float(classical_results.get('f1_score', 0)),          # Overall F1-score on validation set
                        
                        # Training Performance - For overfitting analysis
                        'training_accuracy': float(classical_results.get('training_accuracy', 0)),      # Accuracy on training set
                        'training_precision': float(classical_results.get('training_precision', 0)),    # Precision on training set
                        'training_recall': float(classical_results.get('training_recall', 0)),          # Recall on training set
                        'training_f1_score': float(classical_results.get('training_f1', 0)),            # F1-score on training set
                        
                        'per_class_metrics': {k: float(v) if not isinstance(v, int) else v for k, v in classical_class_metrics.items()},
                        
                        'timing': {
                            'training_time': float(classical_results.get('training_time', 0)),
                            'prediction_time': float(classical_results.get('prediction_time', 0)),
                            'total_time': float(classical_total_time),
                            'feature_importance_time': stage_timings.get('classical_feature_importance', 0)
                        },
                        'training_time': float(classical_results.get('training_time', 0)),
                        'prediction_time': float(classical_results.get('prediction_time', 0)),
                        'total_time': float(classical_total_time),
                        
                        # Model Information - Classical SVM specifics
                        'support_vectors': classical_results.get('support_vectors', 'N/A'),             # Number of support vectors
                        'samples_used_training': len(X_train),
                        'samples_used_validation': len(X_val),
                        'resource_usage': 'See resource_monitoring.per_stage_stats.classical_svm_training for stage-specific data',
                        
                        # Feature Importance Analysis - Detailed breakdown of most important features
                        'feature_importance_analysis': classical_feature_importance,                      # Comprehensive feature importance analysis
                        'cached_predictions_file': classical_cache_fn if is_deterministic_backend else None
                    },
                    'quantum': {
                        'description': f"Quantum SVM Results: {('QSVC with FidelityStatevectorKernel' if is_deterministic_backend else 'QSVC with FidelityQuantumKernel')} using {FEATURE_MAP_TYPE} FeatureMap, FEATURE_MODE={FEATURE_MODE}.",
                        # Overall Performance (Validation-based) - Main scientific metrics
                        'accuracy': float(quantum_acc),                      # Overall accuracy on validation set
                        'precision': float(quantum_results.get('precision', 0)),          # Overall precision on validation set
                        'recall': float(quantum_results.get('recall', 0)),                # Overall recall on validation set
                        'f1_score': float(quantum_results.get('f1_score', 0)),            # Overall F1-score on validation set
                        
                        # Training Performance - For overfitting analysis
                        'training_accuracy': float(quantum_results.get('training_accuracy', 0)),        # Accuracy on training set
                        'training_precision': float(quantum_results.get('training_precision', 0)),      # Precision on training set
                        'training_recall': float(quantum_results.get('training_recall', 0)),            # Recall on training set
                        'training_f1_score': float(quantum_results.get('training_f1', 0)),              # F1-score on training set
                        
                        'per_class_metrics': {k: float(v) if not isinstance(v, int) else v for k, v in quantum_class_metrics.items()},
                        
                        # Timing Metrics - Detailed breakdown for reproducibility and analysis
                        'timing': {
                            'core_computation': {
                                'circuit_training_time': float(quantum_results.get('circuit_training_time', 0)),  # Quantum kernel matrix computation (seconds)
                                'prediction_time': float(quantum_results.get('prediction_time', 0)),              # Validation predictions on quantum circuits (seconds)
                                'final_model_time': float(quantum_results.get('final_model_time', 0)),            # Final model training time (seconds, usually 0 for no-CV)
                                'core_total_time': float(quantum_core_time),                                      # Total core computation time (training + prediction)
                                'description': 'Core quantum computation time (training + prediction, no post-processing)'
                            },
                            'post_processing': {
                                'post_processing_time': float(post_processing_time),                              # Post-processing time (t-SNE, DBI, feature importance, file I/O)
                                'components': 'Includes: t-SNE visualization, Davies-Bouldin Index, feature importance analysis, silhouette analysis, JSON/CSV generation, model serialization',
                                'description': 'Time spent on analysis, visualization, and data export after model training'
                            },
                            'total': {
                                'total_time': float(quantum_total_time),                                          # Complete wall-clock time (core + post-processing)
                                'description': 'Total experimental time from start to finish (includes everything)'
                            },
                            'breakdown_percent': {
                                'circuit_training_pct': float(quantum_results.get('circuit_training_time', 0) / quantum_total_time * 100) if quantum_total_time > 0 else 0,
                                'prediction_pct': float(quantum_results.get('prediction_time', 0) / quantum_total_time * 100) if quantum_total_time > 0 else 0,
                                'post_processing_pct': float(post_processing_time / quantum_total_time * 100) if quantum_total_time > 0 else 0
                            }
                        },
                        
                        # Legacy timing fields (for backward compatibility)
                        'circuit_training_time': float(quantum_results.get('circuit_training_time', 0)), # DEPRECATED: Use timing.core_computation.circuit_training_time
                        'prediction_time': float(quantum_results.get('prediction_time', 0)),            # DEPRECATED: Use timing.core_computation.prediction_time
                        'total_time': float(quantum_total_time),                                         # DEPRECATED: Use timing.total.total_time
                        'final_model_time': float(quantum_results.get('final_model_time', 0)),          # DEPRECATED: Use timing.core_computation.final_model_time
                        
                        # Quantum Information - Quantum-specific parameters and results
                        'feature_dimension': quantum_results.get('feature_dimension', QUANTUM_FEATURE_DIM),           # Number of quantum features (PCA components)
                        'quantum_shots': ('exact' if STATEVECTOR_SHOTS is None else STATEVECTOR_SHOTS) if is_deterministic_backend else quantum_results.get('quantum_shots', QUANTUM_SHOTS),
                        'pca_explained_variance': float(quantum_results.get('pca_variance_explained') or 0),            # Variance explained by PCA (0.0 when skip_pca=True)
                        'kernel_matrix_entries': len(X_train) * len(X_train),
                        'kernel_computation_method': 'exact_statevector' if is_deterministic_backend else f'sampled_{QUANTUM_SHOTS}_shots',
                        'samples_used_training': quantum_results.get('samples_used', len(X_train)),
                        'samples_used_validation': quantum_results.get('validation_samples_used', len(X_val)),
                        'resource_usage': 'See resource_monitoring.per_stage_stats.quantum_svm_training for stage-specific data',
                        
                        # Feature Importance Analysis - Detailed breakdown of most important features
                        'feature_importance_analysis': quantum_feature_importance,                                    # Comprehensive feature importance analysis
                        
                        # Cross-Validation Results (if CV monitoring was enabled)
                        'cross_validation_enabled': QUANTUM_CV_FOLDS > 1,
                        'cv_folds_used': QUANTUM_CV_FOLDS if QUANTUM_CV_FOLDS > 1 else None,
                        'cv_mean_train_accuracy': float(quantum_results['cv_mean_train_accuracy']) if quantum_results.get('cv_mean_train_accuracy') is not None else None,
                        'cv_std_train_accuracy': float(quantum_results['cv_std_train_accuracy']) if quantum_results.get('cv_std_train_accuracy') is not None else None,
                        'cv_mean_val_accuracy': float(quantum_results['cv_mean_val_accuracy']) if quantum_results.get('cv_mean_val_accuracy') is not None else None,
                        'cv_std_val_accuracy': float(quantum_results['cv_std_val_accuracy']) if quantum_results.get('cv_std_val_accuracy') is not None else None,
                        'cv_per_fold_results': quantum_results.get('cv_results', {}).get('fold_results', None) if QUANTUM_CV_FOLDS > 1 else None,
                        'quantum_training_log_filename': get_dynamic_filename(QUANTUM_LOG_FILENAME) if ENABLE_QUANTUM_TRAINING_LOG else None,  # CSV log filename
                        
                        # t-SNE Analysis Results
                        'tsne_report_enabled': TSNE_REPORT,                                                               # Whether t-SNE data analysis was performed
                        'tsne_visualization_enabled': ENABLE_TSNE_VISUALIZATION,                                          # Whether PNG visualizations were auto-generated
                        'tsne_parameters': {
                            'perplexity': TSNE_PERPLEXITY,
                            'learning_rate': TSNE_LEARNING_RATE,
                            'iterations': TSNE_ITERATIONS,
                            'separation_analysis': TSNE_ANALYZE_SEPARATION
                        } if TSNE_REPORT else None,
                        'tsne_analysis_report': get_dynamic_filename('tsne_analysis_report.json') if TSNE_REPORT else None,  # t-SNE analysis report filename
                        # Auto-Slicing Results
                        "auto_slicing_enabled": AUTO_SLICE,
                        "slicing_used": slicing_info["slicing_needed"] if slicing_info else False,
                        "slicing_slices": slicing_info["optimal_slices"] if slicing_info and slicing_info["slicing_needed"] else 0,
                        "slicing_memory_gb": slicing_info["total_memory_gb"] if slicing_info else 0,
                        "slicing_memory_percent": slicing_info["memory_usage_percent"] if slicing_info else 0,
                        "slicing_server": slicing_info["server_name"] if slicing_info else "Unknown",
                        "slicing_threshold_gb": slicing_info["slicing_threshold_gb"] if slicing_info else SLICING_THRESHOLD_GB,
                        'cached_predictions_file': quantum_cache_fn if is_deterministic_backend else None,
                        'cached_quantum_features_file': quantum_feats_fn if is_deterministic_backend else None

                    },
                    'pipeline_timing': {
                        'description': 'Wall-clock seconds for each pipeline stage. Post-processing values are patched after completion.',
                        'data_loading': stage_timings.get('data_loading', 0),
                        'preprocessing': stage_timings.get('preprocessing', 0),
                        'classical_svm_training': stage_timings.get('classical_svm_training', 0),
                        'classical_feature_importance': stage_timings.get('classical_feature_importance', 0),
                        'quantum_svm_training': stage_timings.get('quantum_svm_training', 0),
                        'quantum_feature_importance': stage_timings.get('quantum_feature_importance', 0),
                        'model_serialization': 0,
                        'json_results_writing': 0,
                        'tsne_dbi_silhouette': 0,
                        'file_io_and_artifacts': 0,
                        'post_processing_total': 0,
                        'pipeline_total': 0,
                        '_note': 'Zeros are patched with actual values after the pipeline completes.'
                    },
                    'resource_monitoring': {
                        '_note': 'Full per-stage resource data (memory peaks, CPU, tree memory) is patched into this section after all stages complete.',
                        'monitoring_enabled': MONITOR_RESOURCES,
                        'sampling_interval_seconds': MONITORING_INTERVAL,
                        'stage_tracking_enabled': True
                    }
                }
                
                results_filename = get_dynamic_filename('training_results.json')
                results_path = os.path.join(temp_output_path, results_filename)
                with open(results_path, 'w') as f:
                    json.dump(comparison_data, f, indent=2)
                saved_files.append(results_filename)
                print(f"✅ Results saved as {results_filename}")
                
                # Create overview file (without feature importance details)
                overview_filename = get_dynamic_filename('training_results_overview.json')
                overview_path = os.path.join(temp_output_path, overview_filename)
                
                import copy
                overview_data = copy.deepcopy(comparison_data)
                
                if 'classical' in overview_data and 'feature_importance_analysis' in overview_data['classical']:
                    del overview_data['classical']['feature_importance_analysis']
                if 'quantum' in overview_data and 'feature_importance_analysis' in overview_data['quantum']:
                    del overview_data['quantum']['feature_importance_analysis']
                if 'quantum' in overview_data and 'cv_per_fold_results' in overview_data['quantum']:
                    del overview_data['quantum']['cv_per_fold_results']
                
                overview_data['_overview_note'] = "Concise overview. Full feature importance and raw resource time-series in training_results.json and resource_timeseries.csv."
                
                # Add a quick-read summary at the top of the overview
                overview_data['quick_summary'] = {
                    'classical_accuracy': float(classical_acc),
                    'quantum_accuracy': float(quantum_acc),
                    'accuracy_difference': float(quantum_acc - classical_acc),
                    'quantum_better': bool(quantum_acc > classical_acc),
                    'classical_training_time_s': round(classical_total_time, 2),
                    'quantum_core_time_s': round(quantum_core_time, 2),
                    'speedup_factor': round(classical_total_time / quantum_core_time, 2) if quantum_core_time > 0 else 0,
                    'feature_mode': FEATURE_MODE,
                    'feature_pipeline': _feat_method_short,
                    'mi_selected_features': _mi_features,
                    'skip_pca': _skip_pca,
                    'quantum_feature_dim': QUANTUM_FEATURE_DIM,
                    'training_samples': len(X_train),
                    'validation_samples': len(X_val),
                    'backend': actual_backend_used,
                    'svm_c': SVM_C,
                    'qsvm_c': QSVM_C,
                    'cv_folds': QUANTUM_CV_FOLDS if QUANTUM_CV_FOLDS > 1 else 'disabled',
                    'batch_training': QSVM_BATCH_TRAINING
                }
                
                # Inject cache file references for quick overview if present
                if is_deterministic_backend:
                    overview_data.setdefault('classical', {})['cached_predictions_file'] = classical_cache_fn
                    overview_data.setdefault('quantum', {})['cached_predictions_file'] = quantum_cache_fn
                    overview_data.setdefault('quantum', {})['cached_quantum_features_file'] = quantum_feats_fn

                with open(overview_path, 'w') as f:
                    json.dump(overview_data, f, indent=2)
                saved_files.append(overview_filename)
                print(f"✅ Overview saved as {overview_filename}")
                stage_timings['json_results_writing'] = time.time() - _json_start
            
            # Save quantum training log if available
            if quantum_monitor and ENABLE_QUANTUM_TRAINING_LOG:
                try:
                    csv_path, csv_filename = quantum_monitor.save_log(temp_output_path)
                    if csv_path:
                        saved_files.append(csv_filename)
                        print(f"✅ Quantum training log saved as {csv_filename}")
                        
                        # Print summary statistics
                        summary_stats = quantum_monitor.get_summary_stats()
                        if summary_stats:
                            print(f"📊 Training log summary:")
                            print(f"   • Total checkpoints: {summary_stats.get('total_checkpoints', 0)}")
                            print(f"   • CV folds: {summary_stats.get('total_folds', 0)}")
                            if summary_stats.get('best_val_accuracy'):
                                print(f"   • Best validation accuracy: {summary_stats['best_val_accuracy']:.4f}")
                            if summary_stats.get('avg_cpu_usage'):
                                print(f"   • Average CPU usage: {summary_stats['avg_cpu_usage']:.1f}%")
                            if summary_stats.get('peak_memory_usage'):
                                print(f"   • Peak memory usage: {summary_stats['peak_memory_usage']:.1f} MB")
                except Exception as e:
                    print(f"⚠️  Failed to save quantum training log: {e}")
            
            # Compute shots_label for use in t-SNE, artifacts, and metadata
            shots_label = (
                ('exact' if STATEVECTOR_SHOTS is None else STATEVECTOR_SHOTS)
                if is_deterministic_backend
                else QUANTUM_SHOTS
            )
            
            # Generate t-SNE data and/or visualizations if enabled
            if TSNE_REPORT:
                try:
                    monitor.mark_stage('tsne_dbi_silhouette')
                    _tsne_start = time.time()
                    print("🎨 Generating t-SNE analysis...")
                    from tsne_visualizer import TSNEVisualizer
                    
                    quantum_params = {
                        'feature_dim': QUANTUM_FEATURE_DIM,
                        'shots': shots_label,
                        'train_samples': len(X_train),
                        'val_samples': len(X_val),
                        'backend': actual_backend_used
                    }
                    
                    # Initialize visualizer (PNG generation controlled by ENABLE_TSNE_VISUALIZATION)
                    tsne_viz = TSNEVisualizer(
                        output_dir=temp_output_path,
                        dynamic_naming=DYNAMIC_NAMING,
                        quantum_params=quantum_params
                    )
                    
                    # Update t-SNE parameters
                    tsne_viz.tsne_params.update({
                        'perplexity': TSNE_PERPLEXITY,
                        'learning_rate': TSNE_LEARNING_RATE,
                        'max_iter': TSNE_ITERATIONS
                    })
                    
                    # Generate comprehensive analysis (always generates data)
                    tsne_results = tsne_viz.generate_comprehensive_analysis(
                        X_train=X_train,
                        X_val=X_val,
                        y_train=y_train,
                        y_val=y_val,
                        classical_model=classical_results.get('model'),
                        quantum_model=qsvm,  # Pass the full quantum SVM object, not just the sklearn model
                        quantum_pca=qsvm.pca if hasattr(qsvm, 'pca') and qsvm.pca is not None else None,
                        class_names=['Gemma 3', 'Qwen 2.5'],
                        generate_plots=ENABLE_TSNE_VISUALIZATION,  # Control PNG generation
                        precomputed_classical_predictions=precomp_classical_preds if is_deterministic_backend else None,
                        precomputed_quantum_predictions=precomp_quantum_preds if is_deterministic_backend else None,
                        precomputed_quantum_features=precomp_quantum_features if is_deterministic_backend else None
                    )
                    
                    # Save analysis report (always save the JSON data)
                    tsne_report_path = tsne_viz.save_analysis_report(tsne_results)
                    
                    # Add t-SNE generated artifacts (PNGs and any raw files like DBI JSON/CSV)
                    for plot_info in tsne_viz.plots_generated:
                        saved_files.append(plot_info['filename'])
                    if ENABLE_TSNE_VISUALIZATION:
                        print(f"✅ t-SNE analysis complete: {len(tsne_viz.plots_generated)} artifacts generated")
                    else:
                        print("✅ t-SNE data analysis complete (PNG generation disabled)")
                    
                    # Add report to saved files (always add the JSON report)
                    report_filename = os.path.basename(tsne_report_path)
                    saved_files.append(report_filename)
                    
                    # Add raw t-SNE data files to saved files list (if they were generated)
                    if hasattr(tsne_viz, '_raw_tsne_data') and tsne_viz._raw_tsne_data:
                        # Add the main raw data JSON file
                        raw_data_json = tsne_viz.get_dynamic_filename('tsne_raw_data.json')
                        saved_files.append(raw_data_json)
                        
                        # Add all the CSV files for each data type
                        for data_type in tsne_viz._raw_tsne_data.keys():
                            csv_filename = tsne_viz.get_dynamic_filename(f'tsne_raw_data_{data_type}.csv')
                            saved_files.append(csv_filename)
                        
                        print(f"✅ Raw t-SNE data files added to file list for dynamic naming")

                    # Ensure DBI raw artifacts (JSON/CSV) are captured even if plotting disabled
                    try:
                        for fn in os.listdir(temp_output_path):
                            if fn.startswith('Davies-Bouldin_Index_raw_data') and (fn.endswith('.json') or fn.endswith('.csv')):
                                if fn not in saved_files:
                                    saved_files.append(fn)
                    except Exception:
                        pass
                    
                    # Print separation quality insights
                    if TSNE_ANALYZE_SEPARATION and 'summary' in tsne_results:
                        summary = tsne_results['summary']
                        comparison = summary.get('separation_quality_comparison', {})
                        
                        if 'insights' in comparison and comparison['insights']:
                            print("📊 t-SNE Separation Analysis:")
                            for insight in comparison['insights']:
                                print(f"   • {insight}")
                        
                        if 'best_separation' in comparison and comparison['best_separation']:
                            print(f"   • Best separation: {comparison['best_separation']} space")
                    
                    stage_timings['tsne_dbi_silhouette'] = time.time() - _tsne_start
                except Exception as e:
                    stage_timings['tsne_dbi_silhouette'] = time.time() - _tsne_start
                    print(f"⚠️  t-SNE visualization failed: {e}")
                    import traceback
                    print("   Full error details:")
                    traceback.print_exc()
            
            # Handle dynamic naming - move files to new directory
            _file_io_start = time.time()
            if DYNAMIC_NAMING:
                dynamic_dir = get_output_directory()
                final_output_path = os.path.join(script_dir, dynamic_dir)
                
                # Ensure uniqueness if a folder with the same name already exists
                if os.path.exists(final_output_path):
                    base_dir = dynamic_dir
                    suffix_idx = 2
                    while os.path.exists(os.path.join(script_dir, f"{base_dir}_run{suffix_idx}")):
                        suffix_idx += 1
                    dynamic_dir = f"{base_dir}_run{suffix_idx}"
                    final_output_path = os.path.join(script_dir, dynamic_dir)
                    print(f"⚠️  Output directory exists; using unique directory: {dynamic_dir}")
                else:
                    print(f"📁 Creating dynamic directory: {dynamic_dir}")
                os.makedirs(final_output_path, exist_ok=True)
                
                monitor.mark_stage('file_io_and_artifacts')
                import shutil
                for filename in saved_files:
                    src_path = os.path.join(temp_output_path, filename)
                    dst_path = os.path.join(final_output_path, filename)
                    if os.path.exists(src_path):
                        shutil.move(src_path, dst_path)
                        print(f"📦 Moved {filename} to {dynamic_dir}/")
                
                # Move silhouette analysis files (they're in a subdirectory)
                silhouette_src_dir = os.path.join(temp_output_path, "silhouette_analysis")
                silhouette_dst_dir = os.path.join(final_output_path, "silhouette_analysis")
                if os.path.exists(silhouette_src_dir):
                    if os.path.exists(silhouette_dst_dir):
                        shutil.rmtree(silhouette_dst_dir)
                    shutil.move(silhouette_src_dir, silhouette_dst_dir)
                    print(f"📦 Moved silhouette_analysis/ to {dynamic_dir}/")
                
                monitor.mark_stage('inference_artifacts')
                try:
                    inference_dirname = f"inference_artifacts_{QUANTUM_FEATURE_DIM}-{_shots_label()}-{TRAINING_SAMPLES}-{VALIDATION_SAMPLES}-Cs{SVM_C:g}_Cq{QSVM_C:g}"
                    inference_path = os.path.join(final_output_path, inference_dirname)
                    os.makedirs(inference_path, exist_ok=True)

                    # Dump common preprocessing artifacts
                    try:
                        import joblib
                        processor = data_dict.get('processor', None)
                        if processor and getattr(processor, 'vectorizer', None) is not None:
                            joblib.dump(processor.vectorizer, os.path.join(inference_path, 'tfidf_vectorizer.joblib'))
                        if processor and getattr(processor, 'feature_scaler', None) is not None:
                            joblib.dump(processor.feature_scaler, os.path.join(inference_path, 'linguistic_scaler.joblib'))
                        if processor and getattr(processor, 'label_encoder', None) is not None:
                            joblib.dump(processor.label_encoder, os.path.join(inference_path, 'label_encoder.joblib'))
                        # Mode-specific artifacts for inference pipeline reconstruction
                        if processor and getattr(processor, 'char_vectorizer', None) is not None:
                            joblib.dump(processor.char_vectorizer, os.path.join(inference_path, 'char_tfidf_vectorizer.joblib'))
                        if processor and getattr(processor, 'char_pca', None) is not None:
                            joblib.dump(processor.char_pca, os.path.join(inference_path, 'char_pca.joblib'))
                        if processor and getattr(processor, 'mi_selector', None) is not None:
                            joblib.dump(processor.mi_selector, os.path.join(inference_path, 'mi_selector.joblib'))
                    except Exception as dump_err:
                        print(f"⚠️  Failed to dump common preprocessing artifacts: {dump_err}")

                    # QSVM-only artifacts: PCA and normalization params
                    try:
                        import joblib
                        if hasattr(qsvm, 'pca') and qsvm.pca is not None:
                            joblib.dump(qsvm.pca, os.path.join(inference_path, 'QSVM_pca_quantum.joblib'))
                        norm_min = getattr(qsvm, 'quant_norm_min', None)
                        norm_max = getattr(qsvm, 'quant_norm_max', None)
                        
                        if norm_min is None:
                            norm_min = getattr(qsvm, '_norm_min', None)
                        if norm_max is None:
                            _norm_range = getattr(qsvm, '_norm_range', None)
                            if norm_min is not None and _norm_range is not None:
                                norm_max = norm_min + _norm_range
                        
                        if norm_min is not None and norm_max is not None:
                            try:
                                np.savez(
                                    os.path.join(inference_path, 'QSVM_quantum_normalization_params.npz'),
                                    x_min=norm_min,
                                    x_max=norm_max
                                )
                            except Exception as norm_err:
                                print(f"⚠️  Failed to save QSVM normalization params: {norm_err}")
                        else:
                            try:
                                if hasattr(qsvm, 'pca') and qsvm.pca is not None:
                                    fb_scaler = getattr(qsvm, 'feature_scaler', getattr(qsvm, 'scaler', None))
                                    X_for_pca = fb_scaler.transform(X_train) if fb_scaler is not None else X_train
                                    X_train_pca = qsvm.pca.transform(X_for_pca)
                                    x_min = X_train_pca.min(axis=0)
                                    x_max = X_train_pca.max(axis=0)
                                    np.savez(
                                        os.path.join(inference_path, 'QSVM_quantum_normalization_params.npz'),
                                        x_min=x_min,
                                        x_max=x_max
                                    )
                            except Exception as norm_fb_err:
                                print(f"⚠️  Fallback save of QSVM normalization params failed: {norm_fb_err}")

                        # If cuTensorNet-style precomputed-kernel embeddings are available, save them for strict inference
                        try:
                            X_train_quantum = getattr(qsvm, 'X_train_quantum', None)
                            if X_train_quantum is not None:
                                np.savez_compressed(
                                    os.path.join(inference_path, 'QSVM_train_quantum_embeddings.npz'),
                                    X_train_quantum=X_train_quantum
                                )
                        except Exception as emb_err:
                            print(f"⚠️  Failed to save QSVM train quantum embeddings: {emb_err}")

                        # Save kernel mode metadata to guide inference
                        try:
                            kernel_mode = 'precomputed' if getattr(qsvm, 'X_train_quantum', None) is not None else 'internal'
                            backend_used = actual_backend_used
                            kernel_meta = {
                                'kernel_mode': kernel_mode,
                                'backend': backend_used,
                                'feature_dim': QUANTUM_FEATURE_DIM,
                                'shots': shots_label,
                                'svm_C': SVM_C,
                                'qsvm_C': QSVM_C
                            }
                            with open(os.path.join(inference_path, 'QSVM_kernel_mode.json'), 'w') as kmf:
                                json.dump(kernel_meta, kmf, indent=2)
                        except Exception as km_err:
                            print(f"⚠️  Failed to write QSVM kernel metadata: {km_err}")
                    except Exception as qsvm_art_err:
                        print(f"⚠️  Failed to save QSVM inference artifacts: {qsvm_art_err}")

                    # Metadata JSON
                    try:
                        import json
                        processor = data_dict.get('processor', None)
                        linguistic_names = []
                        if processor and hasattr(processor, 'get_linguistic_feature_names'):
                            try:
                                linguistic_names = processor.get_linguistic_feature_names()
                            except Exception:
                                linguistic_names = []
                        tfidf_params = {}
                        if processor and hasattr(processor, 'tfidf_params'):
                            tfidf_params = processor.tfidf_params
                        label_classes = []
                        if processor and getattr(processor, 'label_encoder', None) is not None:
                            try:
                                label_classes = processor.label_encoder.classes_.tolist()
                            except Exception:
                                label_classes = []
                        metadata = {
                            'feature_mode': FEATURE_MODE,
                            'skip_pca': data_dict.get('skip_pca', False),
                            'mi_selected_feature_names': data_dict.get('mi_selected_feature_names'),
                            'tfidf_params': tfidf_params,
                            'linguistic_feature_names': linguistic_names,
                            'label_classes': label_classes,
                            'feature_dim': QUANTUM_FEATURE_DIM,
                            'shots': shots_label,
                            'svm_C': SVM_C,
                            'qsvm_C': QSVM_C,
                            'training_samples': TRAINING_SAMPLES,
                            'validation_samples': VALIDATION_SAMPLES
                        }
                        with open(os.path.join(inference_path, 'metadata.json'), 'w') as mf:
                            json.dump(metadata, mf, indent=2)
                    except Exception as meta_err:
                        print(f"⚠️  Failed to write inference metadata: {meta_err}")

                    # Standardized model filenames for inference
                    try:
                        # QSVM model
                        qsvm_dynamic_name = None
                        for fn in os.listdir(final_output_path):
                            if fn.startswith('quantum_model_') and fn.endswith('.joblib'):
                                qsvm_dynamic_name = fn
                                break
                        if qsvm_dynamic_name:
                            shutil.copyfile(
                                os.path.join(final_output_path, qsvm_dynamic_name),
                                os.path.join(inference_path, 'QSVM_model.joblib')
                            )
                        # Classical SVM model
                        svm_dynamic_name = None
                        for fn in os.listdir(final_output_path):
                            if fn.startswith('classical_model_') and fn.endswith('.joblib'):
                                svm_dynamic_name = fn
                                break
                        if svm_dynamic_name:
                            shutil.copyfile(
                                os.path.join(final_output_path, svm_dynamic_name),
                                os.path.join(inference_path, 'SVM_model.joblib')
                            )
                        print(f"✅ Inference artifacts saved in: {inference_path}")
                    except Exception as model_copy_err:
                        print(f"⚠️  Failed to standardize model filenames for inference: {model_copy_err}")

                except Exception as inf_err:
                    print(f"⚠️  Inference artifacts bundle creation failed: {inf_err}")

                # Move the streaming resource CSV to the final directory
                monitor.relocate_csv(final_output_path)
                
                print(f"✅ All files organized in: {final_output_path}")
                # Best-effort cleanup of the temp directory
                try:
                    shutil.rmtree(temp_output_path, ignore_errors=True)
                except Exception:
                    pass
            else:
                print(f"✅ Files saved in: {temp_output_path}")
        else:
            _file_io_start = time.time()
            print("💾 Saving disabled (SAVE_MODELS=False, SAVE_RESULTS=False)")
        
        stage_timings['file_io_and_artifacts'] = time.time() - _file_io_start
        
        # Mark end of post-processing and calculate times
        monitor.mark_stage('finalize')
        post_processing_end_time = time.time()
        post_processing_time = post_processing_end_time - post_processing_start_time
        stage_timings['post_processing_total'] = post_processing_time
        stage_timings['pipeline_total'] = post_processing_end_time - pipeline_start_time
        quantum_total_time = post_processing_end_time - quantum_start_time
        
        # Stop the pipeline-wide monitor now that everything is done
        monitor.stop_monitoring()
        final_monitoring_stats = monitor.get_statistics()
        per_stage_stats = monitor.get_per_stage_stats()
        
        # Patch timing and per-stage resource data into saved JSONs
        import json
        def _save_vmhwm_profile(out_dir, mon, pss, fms, st):
            """Write a per-run VmHWM memory profile CSV for cross-configuration comparison.
            
            Format: one header + per-stage rows + a TOTAL summary row.  Each row
            carries the full configuration key so files from different runs can be
            concatenated with ``head -1 run1/vmhwm*.csv > all.csv && tail -q -n+2
            */vmhwm*.csv >> all.csv`` for tabular analysis.
            """
            import csv as _csv
            vmhwm_final = get_vmhwm_mb()
            if vmhwm_final is None and not pss:
                return
            fname = os.path.join(out_dir, 'vmhwm_memory_profile.csv')
            config_key = (
                f"{QUANTUM_FEATURE_DIM}D_{QUANTUM_BACKEND}"
                f"_sv{STATEVECTOR_SHOTS}_sh{QUANTUM_SHOTS}"
                f"_Cs{SVM_C}_Cq{QSVM_C}"
                f"_tr{TRAINING_SAMPLES}_va{VALIDATION_SAMPLES}"
            )
            fieldnames = [
                'config_key', 'backend', 'feature_dim', 'shots',
                'statevector_shots', 'svm_c', 'qsvm_c',
                'training_samples', 'validation_samples', 'cv_folds',
                'stage', 'stage_duration_s',
                'vmhwm_at_stage_mb', 'process_peak_rss_mb', 'tree_peak_rss_mb',
                'cpu_max_percent', 'system_peak_memory_gb',
                'timestamp_iso'
            ]
            try:
                with open(fname, 'w', newline='') as fh:
                    writer = _csv.DictWriter(fh, fieldnames=fieldnames)
                    writer.writeheader()
                    base_row = {
                        'config_key': config_key,
                        'backend': QUANTUM_BACKEND,
                        'feature_dim': QUANTUM_FEATURE_DIM,
                        'shots': QUANTUM_SHOTS,
                        'statevector_shots': STATEVECTOR_SHOTS,
                        'svm_c': SVM_C,
                        'qsvm_c': QSVM_C,
                        'training_samples': TRAINING_SAMPLES,
                        'validation_samples': VALIDATION_SAMPLES,
                        'cv_folds': QUANTUM_CV_FOLDS,
                        'timestamp_iso': datetime.now().isoformat()
                    }
                    vmhwm_lookup = {}
                    for snap in mon.stage_snapshots:
                        vmhwm_lookup[snap.get('prev_stage', '')] = snap.get('vmhwm_mb')
                    if mon.stage_snapshots:
                        last_snap = mon.stage_snapshots[-1]
                        vmhwm_lookup[last_snap['stage']] = last_snap.get('vmhwm_mb')

                    if pss:
                        for stage, stats in pss.items():
                            row = dict(base_row)
                            row['stage'] = stage
                            row['stage_duration_s'] = stats.get('duration_seconds', '')
                            row['vmhwm_at_stage_mb'] = vmhwm_lookup.get(stage, '')
                            row['process_peak_rss_mb'] = stats.get('process_memory_peak_mb', '')
                            row['tree_peak_rss_mb'] = stats.get('tree_memory_peak_mb', '')
                            row['cpu_max_percent'] = stats.get('cpu_max_percent', '')
                            row['system_peak_memory_gb'] = stats.get('system_memory_peak_gb', '')
                            writer.writerow(row)

                    total_row = dict(base_row)
                    total_row['stage'] = 'TOTAL'
                    total_row['stage_duration_s'] = st.get('pipeline_total', '')
                    total_row['vmhwm_at_stage_mb'] = vmhwm_final
                    if fms:
                        total_row['process_peak_rss_mb'] = fms.get('process_usage', {}).get('peak_memory_mb', '')
                        total_row['tree_peak_rss_mb'] = fms.get('process_usage', {}).get('peak_tree_memory_mb', '')
                        total_row['cpu_max_percent'] = fms.get('cpu_usage', {}).get('max_percent', '')
                        total_row['system_peak_memory_gb'] = fms.get('memory_usage', {}).get('peak_used_gb', '')
                    writer.writerow(total_row)
                print(f"📊 VmHWM memory profile saved: vmhwm_memory_profile.csv ({len(pss) + 1 if pss else 1} rows)")
            except Exception as e:
                print(f"⚠️  VmHWM profile save failed: {e}")

        def _patch_timing(json_path):
            """Patch post-processing timing, pipeline timing, and resource data into a saved JSON file."""
            if not os.path.exists(json_path):
                return
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Patch quantum timing section
            q = data.get('quantum', {})
            if 'timing' in q:
                q['timing']['post_processing']['post_processing_time'] = float(post_processing_time)
                q['timing']['total']['total_time'] = float(quantum_total_time)
                if quantum_total_time > 0:
                    q['timing']['breakdown_percent'] = {
                        'circuit_training_pct': float(quantum_results.get('circuit_training_time', 0) / quantum_total_time * 100),
                        'prediction_pct': float(quantum_results.get('prediction_time', 0) / quantum_total_time * 100),
                        'post_processing_pct': float(post_processing_time / quantum_total_time * 100)
                    }
            q['total_time'] = float(quantum_total_time)
            
            # Patch pipeline_timing with final values
            pt = data.get('pipeline_timing', {})
            for k, v in stage_timings.items():
                pt[k] = round(float(v), 3)
            if '_note' in pt:
                del pt['_note']
            data['pipeline_timing'] = pt
            
            # Patch resource_monitoring with full data
            data['resource_monitoring'] = {
                'monitoring_enabled': MONITOR_RESOURCES,
                'sampling_interval_seconds': monitor.interval,
                'total_samples': monitor._sample_count,
                'resource_timeseries_csv': os.path.basename(monitor._csv_path) if monitor._csv_path else None,
                'pipeline_wide_stats': final_monitoring_stats,
                'per_stage_stats': per_stage_stats,
                'stage_snapshots': monitor.stage_snapshots,
                'kernel_peak_memory_mb_VmHWM': get_vmhwm_mb()
            }
            
            # Enrich quick_summary with final timing data (available after pipeline completes)
            qs = data.get('quick_summary', {})
            if qs:
                qs['post_processing_time_s'] = round(post_processing_time, 2)
                qs['pipeline_total_time_s'] = round(stage_timings.get('pipeline_total', 0), 2)
                qs['pipeline_total_time_h'] = round(stage_timings.get('pipeline_total', 0) / 3600, 3)
                if final_monitoring_stats:
                    qs['peak_process_memory_mb'] = final_monitoring_stats['process_usage']['peak_memory_mb']
                    qs['peak_tree_memory_mb'] = final_monitoring_stats['process_usage']['peak_tree_memory_mb']
                    qs['peak_system_memory_gb'] = final_monitoring_stats['memory_usage']['peak_used_gb']
                    qs['kernel_peak_memory_mb_VmHWM'] = get_vmhwm_mb()
            
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        try:
            # Determine where files ended up
            # Determine where files ended up for JSON patching
            if DYNAMIC_NAMING and 'final_output_path' in dir() and os.path.isdir(final_output_path):
                output_dir = final_output_path
            elif 'temp_output_path' in dir() and os.path.isdir(temp_output_path):
                output_dir = temp_output_path
            else:
                output_dir = None
            
            if output_dir:
                if 'results_filename' in dir():
                    _patch_timing(os.path.join(output_dir, results_filename))
                if 'overview_filename' in dir():
                    _patch_timing(os.path.join(output_dir, overview_filename))
            
            # Resource CSV was streamed incrementally -- just report its location
            if monitor._csv_path and os.path.exists(monitor._csv_path):
                print(f"📊 Resource time-series saved: {os.path.basename(monitor._csv_path)} ({monitor._sample_count} samples, streamed)")
            
            # VmHWM memory profile -- one self-contained CSV per run for cross-configuration comparison
            if SAVE_VMHWM and output_dir:
                _save_vmhwm_profile(output_dir, monitor, per_stage_stats, final_monitoring_stats, stage_timings)
        except Exception as _patch_err:
            print(f"⚠️  Post-processing timing patch failed: {_patch_err}")

        # Print timing breakdown
        print()
        print("=" * 60)
        print("⏱️  TIMING BREAKDOWN")
        print("=" * 60)
        print()
        _total = stage_timings.get('pipeline_total', quantum_total_time)
        _pct = lambda v: f"{v/_total*100:.1f}%" if _total > 0 else "N/A"
        print("Pipeline Stage Timings:")
        print(f"  Data Loading:              {stage_timings.get('data_loading', 0):8.1f}s  ({_pct(stage_timings.get('data_loading', 0))})")
        print(f"  Preprocessing:             {stage_timings.get('preprocessing', 0):8.1f}s  ({_pct(stage_timings.get('preprocessing', 0))})")
        print(f"  Classical SVM Training:    {stage_timings.get('classical_svm_training', 0):8.1f}s  ({_pct(stage_timings.get('classical_svm_training', 0))})")
        print(f"  Classical Feature Import:  {stage_timings.get('classical_feature_importance', 0):8.1f}s  ({_pct(stage_timings.get('classical_feature_importance', 0))})")
        print(f"  Quantum SVM Training:      {stage_timings.get('quantum_svm_training', 0):8.1f}s  ({_pct(stage_timings.get('quantum_svm_training', 0))})")
        print(f"    Circuit Training:        {quantum_results.get('circuit_training_time', 0):8.1f}s")
        print(f"    Prediction:              {quantum_results.get('prediction_time', 0):8.1f}s")
        print(f"  Quantum Feature Import:    {stage_timings.get('quantum_feature_importance', 0):8.1f}s  ({_pct(stage_timings.get('quantum_feature_importance', 0))})")
        print(f"  Model Serialization:       {stage_timings.get('model_serialization', 0):8.1f}s  ({_pct(stage_timings.get('model_serialization', 0))})")
        print(f"  JSON Results Writing:      {stage_timings.get('json_results_writing', 0):8.1f}s  ({_pct(stage_timings.get('json_results_writing', 0))})")
        print(f"  t-SNE / DBI / Silhouette:  {stage_timings.get('tsne_dbi_silhouette', 0):8.1f}s  ({_pct(stage_timings.get('tsne_dbi_silhouette', 0))})")
        print(f"  File I/O & Artifacts:      {stage_timings.get('file_io_and_artifacts', 0):8.1f}s  ({_pct(stage_timings.get('file_io_and_artifacts', 0))})")
        print(f"  ────────────────────────────────────────")
        print(f"  Post-Processing Total:     {post_processing_time:8.1f}s  ({_pct(post_processing_time)})")
        print(f"  Pipeline Total:            {_total:8.1f}s  ({_total/3600:.2f}h)")
        
        if per_stage_stats:
            print()
            print("Per-Stage Resource Peaks:")
            for stage, stats in per_stage_stats.items():
                if stats['samples'] > 0:
                    print(f"  {stage:30s}  {stats['duration_seconds']:7.1f}s  proc:{stats['process_memory_peak_mb']:8.1f}MB  tree/proc:{stats['tree_memory_peak_mb']:8.1f}MB  sys:{stats['system_memory_peak_gb']:6.2f}GB  cpu:{stats['cpu_max_percent']:5.1f}%")
        
        if final_monitoring_stats:
            print()
            vmhwm = get_vmhwm_mb()
            vmhwm_str = f", VmHWM={vmhwm:.1f}MB" if vmhwm else ""
            print(f"Pipeline-wide peaks: process={final_monitoring_stats['process_usage']['peak_memory_mb']:.1f}MB, tree/proc={final_monitoring_stats['process_usage']['peak_tree_memory_mb']:.1f}MB{vmhwm_str}, system={final_monitoring_stats['memory_usage']['peak_used_gb']:.2f}GB (max {final_monitoring_stats['process_usage'].get('max_parallel_processes', 1)} parallel procs)")
            print(f"Monitoring: {final_monitoring_stats['samples_collected']} samples over {final_monitoring_stats['monitoring_duration_seconds']:.1f}s")
        print("=" * 60)
        
        print()
        print("🎉 Training Complete!")
        print(f"⏱️  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 