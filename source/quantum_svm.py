from quantum_svm_parallel_worker import train_single_fold_worker, train_model_with_batching, QuantumEnsemble
import numpy as np
import time
import logging
from typing import Dict, Any, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Quantum computing imports
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel, FidelityStatevectorKernel

# Classical ML imports for comparison
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

# cuTensorNet integration
try:
    from quantum_svm_cutensornet import CuTensorNetQuantumSVM, create_cutensornet_qsvm
    CUTENSORNET_AVAILABLE = True
except ImportError:
    CUTENSORNET_AVAILABLE = False

logger = logging.getLogger(__name__)

def create_optimal_quantum_svm(feature_dim: int = 4, shots: int = 1024,
                              quantum_backend: str = 'aer_simulator_gpu',
                              feature_map_reps: int = 2, feature_map_type: str = 'ZZ',
                              slicing_info: dict = None,
                              sample_size: str = 'full',
                              batch_training: bool = False, batch_size: int = 5,
                              C: float = 1.0,
                              statevector_shots=None,
                              skip_pca: bool = False,
                              **kwargs):
    """
    Create the optimal Quantum SVM based on available backends
    
    Args:
        feature_dim: Quantum feature dimension
        shots: Number of quantum shots
        quantum_backend: Requested backend
        feature_map_reps: Feature map repetitions
        slicing_info: Optional slicing configuration for memory management
        sample_size: Training sample size - 'full' for all samples, or integer string (e.g. '1000')
        batch_training: Whether to use batch (ensemble) training
        batch_size: Number of batches/splits for training
        C: SVM regularization parameter (typical values: 0.1, 1, 10, 100)
        statevector_shots: Shot-noise control for statevector backend.
            None → exact noiseless kernel. int → emulate shot noise.
        **kwargs: Additional parameters
        
    Returns:
        Optimal quantum SVM instance
    """
    
    # Check if cuTensorNet is requested and available
    if 'cutensornet' in quantum_backend.lower() and CUTENSORNET_AVAILABLE:
        print("🚀 Creating cuTensorNet Quantum SVM (GPU accelerated)")
        backend_type = 'cutensornet_gpu' if 'gpu' in quantum_backend.lower() else 'cutensornet_cpu'
        return create_cutensornet_qsvm(
            feature_dim=feature_dim,
            shots=shots,
            backend=backend_type, slicing_info=slicing_info,
            feature_map_reps=feature_map_reps,
            batch_training=batch_training,
            batch_size=batch_size,
            C=C,
            skip_pca=skip_pca,
            **kwargs
        )

    # Auto-select cuTensorNet if available and GPU backend requested
    elif 'gpu' in quantum_backend.lower() and CUTENSORNET_AVAILABLE:
        print("🚀 Auto-selecting cuTensorNet GPU backend (superior to Qiskit AER)")
        return create_cutensornet_qsvm(
            feature_dim=feature_dim,
            shots=shots,
            backend='cutensornet_gpu',
            feature_map_reps=feature_map_reps,
            batch_training=batch_training,
            batch_size=batch_size,
            C=C,
            skip_pca=skip_pca,
            **kwargs
        )
    
    # Fallback to traditional Qiskit implementation
    else:
        if CUTENSORNET_AVAILABLE:
            print("🔧 Using traditional Qiskit implementation (cuTensorNet available but not selected)")
        else:
            print("⚠️ cuTensorNet not available, using Qiskit fallback")
        return QuantumSVM(
            feature_dim=feature_dim,
            shots=shots,
            quantum_backend=quantum_backend,
            feature_map_reps=feature_map_reps,
            feature_map_type=feature_map_type,
            slicing_info=slicing_info,
            sample_size=sample_size,
            batch_training=batch_training,
            batch_size=batch_size,
            C=C,
            statevector_shots=statevector_shots,
            skip_pca=skip_pca,
            **kwargs
        )

class QuantumSVM:
    """
    Quantum Support Vector Machine implementation using Qiskit
    """
    
    def __init__(self, feature_dim: int = 4, shots: int = 1024,
                 sample_size: str = 'full', quantum_backend: str = 'aer_simulator',
                 feature_map_reps: int = 2, feature_map_type: str = 'ZZ',
                 slicing_info: dict = None,
                 batch_training: bool = False, batch_size: int = 5,
                 C: float = 1.0, statevector_shots=None,
                 skip_pca: bool = False):
        """
        Initialize Quantum SVM

        Args:
            feature_dim: Number of features for quantum encoding
            shots: Number of quantum circuit shots (AER / cuTensorNet backends)
            sample_size: Training sample size - 'full' for all samples, or integer string
            quantum_backend: Quantum backend to use
            feature_map_reps: Number of feature map repetitions (1 or 2 recommended)
            slicing_info: Optional slicing configuration for memory management
            batch_training: Whether to use batch (ensemble) training
            batch_size: Number of batches/splits for training
            C: SVM regularization parameter
            statevector_shots: Shot-noise control for statevector backend.
                None → exact noiseless kernel (FidelityStatevectorKernel).
                int  → emulate shot noise via binomial sampling.
            skip_pca: If True, bypass PCA reduction inside fit(). Use when the
                feature matrix is already QUANTUM_FEATURE_DIM-dimensional (e.g.
                FEATURE_MODE='hybrid' or 'stylometric_direct' from data_processor).
                StandardScaler + [0,2π] normalization are still applied.
        """
        self.feature_dim = feature_dim
        self.shots = shots
        self.sample_size = sample_size
        self.quantum_backend = quantum_backend
        self.feature_map_reps = feature_map_reps
        self.feature_map_type = feature_map_type
        self.C = C
        self.statevector_shots = statevector_shots
        self.skip_pca = skip_pca
        self.validation_size = 'full'
        self.slicing_info = slicing_info
        self.is_deterministic = quantum_backend in ['qiskit_statevector_deterministic', 'statevector_deterministic']
        
        # Batch training configuration
        self.batch_training = batch_training
        self.batch_size = batch_size
        
        # Initialize quantum components
        self.feature_map = None
        self.quantum_kernel = None
        self.qsvm_model = None # This will hold either QSVC or QuantumEnsemble
        self.pca = None
        
        # Timing information
        self.training_time = None
        self.prediction_time = None
        
        # Setup quantum backend
        self.backend = self._setup_backend()
        
        # Log slicing information if provided
        if slicing_info:
            logger.info(f"Slicing configuration: {slicing_info['slicing_needed']} (slices: {slicing_info.get('optimal_slices', 0)})")
        
        # Log batch training info
        if batch_training:
            logger.info(f"🔄 Batch training ENABLED: {batch_size} splits")
        
        logger.info(f"Quantum SVM initialized with {feature_dim} features, {shots} shots, sample_size={sample_size}")
    
    def _setup_backend(self):
        """Setup quantum backend with CPU/GPU support"""
        try:
            if self.quantum_backend == 'aer_simulator_gpu':
                # GPU-accelerated AER simulator
                try:
                    # Try to create GPU-accelerated simulator
                    backend = AerSimulator(method="statevector", device="GPU")
                    logger.info(f"Using GPU-accelerated quantum backend: {backend.name} (method: statevector, device: GPU)")
                    
                    # Test if GPU is actually available
                    from qiskit import QuantumCircuit
                    test_circuit = QuantumCircuit(1)
                    test_circuit.h(0)
                    test_circuit.measure_all()
                    
                    # Try a small test run
                    test_job = backend.run(test_circuit, shots=10)
                    test_result = test_job.result()
                    logger.info("✅ GPU backend test successful")
                    
                    return backend
                    
                except Exception as gpu_error:
                    logger.warning(f"GPU backend failed: {gpu_error}")
                    logger.info("🔄 Falling back to CPU backend")
                    # Fall back to CPU
                    backend = AerSimulator()
                    logger.info(f"Using fallback CPU quantum backend: {backend.name}")
                    return backend
            
            elif self.quantum_backend == 'aer_simulator_cpu':
                # CPU-only AER simulator (explicit)
                backend = AerSimulator()
                logger.info(f"Using CPU quantum backend: {backend.name}")
                return backend
            

            elif self.quantum_backend in ['qiskit_statevector_deterministic', 'statevector_deterministic']:
                # Deterministic statevector simulation (ignores shots)
                backend = AerSimulator(method="statevector")
                sv_mode = "exact noiseless" if self.statevector_shots is None else f"shot-noise emulation ({self.statevector_shots} shots)"
                logger.info(f"Using statevector backend: {backend.name} ({sv_mode})")
                return backend
            else:
                # Default/legacy behavior
                backend = AerSimulator()
                logger.info(f"Using default quantum backend: {backend.name}")
                return backend
                
        except Exception as e:
            logger.error(f"Error setting up quantum backend: {e}")
            raise
    
    def _create_feature_map(self) -> QuantumCircuit:
        """
        Create quantum feature map for data encoding.

        Supported types (FEATURE_MAP_TYPE env var):
          'ZZ'          → ZZFeatureMap (default); paulis=['Z','ZZ'], entanglement='linear'
          'Pauli_XX'    → PauliFeatureMap; paulis=['Z','XX'], entanglement='linear'
                          X-type 2-body interactions; different inductive bias from ZZ
          'Pauli_ZZ_XX' → PauliFeatureMap; paulis=['Z','ZZ','XX'], entanglement='linear'
                          Combines both Z and X type entanglement; richer kernel
          'Pauli_full'  → PauliFeatureMap; paulis=['Z','ZX','XX'], entanglement='full'
                          Full 2-local Pauli operators with all-to-all entanglement
        """
        PAULI_CONFIGS = {
            'ZZ':          {'cls': 'ZZ'},
            'Pauli_XX':    {'cls': 'Pauli', 'paulis': ['Z', 'XX'],       'entanglement': 'linear'},
            'Pauli_ZZ_XX': {'cls': 'Pauli', 'paulis': ['Z', 'ZZ', 'XX'], 'entanglement': 'linear'},
            'Pauli_full':  {'cls': 'Pauli', 'paulis': ['Z', 'ZX', 'XX'], 'entanglement': 'full'},
        }

        cfg = PAULI_CONFIGS.get(self.feature_map_type)
        if cfg is None:
            logger.warning(f"Unknown FEATURE_MAP_TYPE '{self.feature_map_type}', falling back to ZZ")
            cfg = PAULI_CONFIGS['ZZ']

        if cfg['cls'] == 'ZZ':
            feature_map = ZZFeatureMap(
                feature_dimension=self.feature_dim,
                reps=self.feature_map_reps,
                entanglement='linear'
            )
            logger.info(f"Created ZZFeatureMap: dim={self.feature_dim}, reps={self.feature_map_reps}")
        else:
            feature_map = PauliFeatureMap(
                feature_dimension=self.feature_dim,
                reps=self.feature_map_reps,
                paulis=cfg['paulis'],
                entanglement=cfg['entanglement']
            )
            logger.info(f"Created PauliFeatureMap ({self.feature_map_type}): "
                        f"dim={self.feature_dim}, reps={self.feature_map_reps}, "
                        f"paulis={cfg['paulis']}, entanglement={cfg['entanglement']}")

        return feature_map
    
    def _prepare_quantum_data(self, X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for quantum processing by reducing dimensionality
        
        Args:
            X_train: Training features
            X_val: Validation features
            
        Returns:
            Tuple of quantum-ready training and validation data
        """
        from sklearn.preprocessing import StandardScaler

        # StandardScaler is always applied regardless of skip_pca
        if not hasattr(self, 'feature_scaler'):
            self.feature_scaler = StandardScaler()
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.feature_scaler.transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)

        if getattr(self, 'skip_pca', False):
            # FEATURE_MODE=hybrid or stylometric_direct: data is already QUANTUM_FEATURE_DIM-wide.
            # Skip PCA entirely; use scaled features directly for quantum encoding.
            logger.info(f"skip_pca=True: bypassing PCA, using {X_train_scaled.shape[1]} features directly "
                        f"(FEATURE_MODE produces pre-reduced {self.feature_dim}-dim input)")
            self.pca = None
            self.explained_variance = None  # no PCA variance to report
            X_train_quantum = X_train_scaled
            X_val_quantum   = X_val_scaled
        else:
            logger.info(f"Reducing dimensionality from {X_train.shape[1]} to {self.feature_dim} via PCA")
            logger.info("✓ Applied StandardScaler before PCA (all features normalized to mean=0, std=1)")

            self.pca = PCA(n_components=self.feature_dim, random_state=42)
            X_train_quantum = self.pca.fit_transform(X_train_scaled)
            X_val_quantum   = self.pca.transform(X_val_scaled)

            # Verify PCA centering
            if hasattr(self.pca, 'mean_'):
                mean_max = np.max(np.abs(self.pca.mean_))
                if mean_max > 0.01:
                    logger.warning(f"⚠️  PCA mean not near zero (max={mean_max:.6f})!")
                else:
                    logger.info(f"✓ PCA centering verified (mean max={mean_max:.10f})")

            if hasattr(self.pca, 'explained_variance_ratio_'):
                self.explained_variance = np.sum(self.pca.explained_variance_ratio_)
            else:
                self.explained_variance = 0.0
            logger.info(f"PCA explained variance: {self.explained_variance:.3f}")

        # [0, 2π] normalization: fitted on training, applied to both
        self.quant_norm_min = X_train_quantum.min(axis=0)
        self.quant_norm_max = X_train_quantum.max(axis=0)
        X_train_quantum = self._normalize_to_2pi(X_train_quantum)
        X_val_quantum   = self._normalize_to_2pi(X_val_quantum)
        logger.info("✓ [0, 2π] normalization fitted on training data and applied to both splits")

        return X_train_quantum, X_val_quantum
    
    def _normalize_to_2pi(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize features to [0, 2π] using training-set min/max.
        Must call after quant_norm_min/max are set from training data.
        Validation values may fall slightly outside [0, 2π] -- clipped for safety.
        """
        X_range = self.quant_norm_max - self.quant_norm_min
        X_range[X_range == 0] = 1
        
        X_normalized = 2 * np.pi * (X - self.quant_norm_min) / X_range
        return np.clip(X_normalized, 0.0, 2 * np.pi)
    
    def create_quantum_kernel(self):
        """
        Create quantum kernel with the appropriate implementation.
        
        For statevector deterministic backends:
          - FidelityStatevectorKernel computes exact |<φ(x)|φ(y)>|² via Statevector
            inner products (no circuit sampling, no shot noise).
          - Optional shot-noise emulation via binomial sampling when statevector_shots
            is set, to predict real-hardware degradation.
        
        For AER / other backends:
          - FidelityQuantumKernel with ComputeUncompute sampler (circuit-based).
        """
        self.feature_map = self._create_feature_map()
        
        if self.is_deterministic:
            self.quantum_kernel = FidelityStatevectorKernel(
                feature_map=self.feature_map,
                shots=self.statevector_shots,
                enforce_psd=(self.statevector_shots is not None),
            )
            if self.statevector_shots is None:
                logger.info(f"✓ Exact statevector kernel created (FidelityStatevectorKernel, noiseless)")
            else:
                logger.info(f"✓ Statevector kernel with shot-noise emulation ({self.statevector_shots} shots)")
        else:
            from qiskit_machine_learning.state_fidelities import ComputeUncompute
            from qiskit.primitives import StatevectorSampler
            
            sampler = StatevectorSampler(default_shots=self.shots)
            fidelity = ComputeUncompute(sampler=sampler)
            self.quantum_kernel = FidelityQuantumKernel(
                feature_map=self.feature_map,
                fidelity=fidelity,
                enforce_psd=True
            )
            logger.info(f"Quantum kernel created with ComputeUncompute sampler, backend: {self.backend.name}, shots: {self.shots}")
        
        return self.quantum_kernel
    
    def train(self, X_train: np.ndarray, X_val: np.ndarray, 
             y_train: np.ndarray, y_val: np.ndarray, 
             train_categories: Optional[List] = None, val_categories: Optional[List] = None) -> Dict[str, Any]:
        """
        Train the quantum SVM model
        
        Args:
            X_train: Training features
            X_val: Validation features  
            y_train: Training labels
            y_val: Validation labels
            
        Returns:
            Dictionary containing training results
        """
        logger.info("Starting quantum SVM training...")
        
        # Prepare quantum data
        X_train_quantum, X_val_quantum = self._prepare_quantum_data(X_train, X_val)
        
        # Create quantum kernel
        self.create_quantum_kernel()
        
        # Measure training time
        start_time = time.time()
        
        try:
            # Determine sample size based on user selection
            if self.sample_size == 'full':
                max_samples = len(X_train_quantum)
                logger.info(f"Using FULL training set: {max_samples} samples")
            else:
                max_samples = int(self.sample_size)
                logger.info(f"Using {max_samples} samples from {len(X_train_quantum)} total training samples")
                
            if len(X_train_quantum) <= max_samples:
                X_train_quantum_subset = X_train_quantum
                y_train_subset = y_train
                logger.info(f"Using all {len(X_train_quantum_subset)} available training samples")
            else:
                logger.info(f"Applying stratified sampling across categories: {max_samples} samples")
                from sklearn.model_selection import train_test_split
                
                # If we have category information, create combined stratification key
                if train_categories is not None and len(train_categories) == len(y_train):
                    # Create combined stratification key (model + category)
                    stratify_keys = [f"{label}_{cat}" for label, cat in zip(y_train, train_categories)]
                    
                    # Check if we have enough samples for each category combination
                    from collections import Counter
                    stratify_counts = Counter(stratify_keys)
                    min_samples_per_key = min(stratify_counts.values())
                    
                    if min_samples_per_key >= 2 and max_samples <= len(y_train):
                        # Use category-aware stratification
                        X_train_quantum_subset, _, y_train_subset, _, train_cats_subset, _ = train_test_split(
                            X_train_quantum, y_train, train_categories,
                            train_size=max_samples,
                            stratify=stratify_keys,
                            random_state=42
                        )
                        logger.info(f"Applied category-aware stratified sampling across {len(set(train_cats_subset))} categories")
                    else:
                        X_train_quantum_subset, _, y_train_subset, _ = train_test_split(
                            X_train_quantum, y_train, 
                            train_size=max_samples,
                            stratify=y_train,
                            random_state=42
                        )
                        logger.info("Applied model-only stratified sampling (insufficient category samples for joint stratification)")
                else:
                    # Original stratification by model only
                    X_train_quantum_subset, _, y_train_subset, _ = train_test_split(
                        X_train_quantum, y_train, 
                        train_size=max_samples,
                        stratify=y_train,
                        random_state=42
                    )
                    logger.info("Applied model-only stratified sampling")
            
            # Train the quantum SVM using helper (handles single or batch/ensemble)
            logger.info(f"Starting quantum training for {len(X_train_quantum_subset)} samples...")
            if self.batch_training:
                logger.info(f"Using BATCH TRAINING with {self.batch_size} splits")
            
            self.qsvm_model = train_model_with_batching(
                X_train_quantum_subset, y_train_subset, 
                kernel=self.quantum_kernel,
                batch_training=self.batch_training,
                n_batches=self.batch_size,
                C=self.C, class_weight='balanced'
            )
            
            self.training_time = time.time() - start_time
            
            logger.info(f"Quantum SVM training completed in {self.training_time:.2f} seconds")
            
            # Calculate training metrics on the subset used for training
            start_pred_time = time.time()
            train_predictions = self.qsvm_model.predict(X_train_quantum_subset)
            train_pred_time = time.time() - start_pred_time
            
            train_accuracy = accuracy_score(y_train_subset, train_predictions)
            train_precision = precision_score(y_train_subset, train_predictions, average='weighted')
            train_recall = recall_score(y_train_subset, train_predictions, average='weighted')
            train_f1 = f1_score(y_train_subset, train_predictions, average='weighted')
            
            # Evaluate on validation set (validation_size is passed from train_and_evaluate)
            logger.info("Starting validation evaluation...")
            val_results = self.evaluate(X_val_quantum, y_val)
            
            return {
                'accuracy': val_results['accuracy'],
                'precision': val_results['precision'],
                'recall': val_results['recall'],
                'f1_score': val_results['f1_score'],
                'classification_report': val_results.get('classification_report', {}),
                'training_accuracy': train_accuracy,
                'training_precision': train_precision,
                'training_recall': train_recall,
                'training_f1': train_f1,
                'training_time': self.training_time,
                'prediction_time': val_results['prediction_time'],
                'feature_dimension': self.feature_dim,
                'quantum_shots': self.shots,
                'pca_variance_explained': self.explained_variance,
                'samples_used': len(X_train_quantum_subset),
                'validation_samples_used': val_results.get('validation_samples_used', 'N/A'),
                'X_train_quantum': X_train_quantum,
                'X_val_quantum': X_val_quantum
            }
            
        except Exception as e:
            error_msg = f"Error during quantum SVM training: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Training was attempted with {self.feature_dim} features")
            
            import traceback
            logger.error(traceback.format_exc())
            
            # Return error information instead of raising
            return {
                'error': True,
                'error_message': error_msg,
                'error_type': type(e).__name__,
                'samples_attempted': 'unknown',
                'feature_dimension': self.feature_dim,
                'quantum_shots': self.shots
            }
    
    def train_with_cv_monitoring(self, X_train: np.ndarray, X_val: np.ndarray, 
                                y_train: np.ndarray, y_val: np.ndarray,
                                cv_folds: int = 5, monitor=None,
                                train_categories: Optional[List] = None, val_categories: Optional[List] = None) -> Dict[str, Any]:
        """
        Train quantum SVM with cross-validation and detailed monitoring.
        
        IMPORTANT -- CV scope:
        Preprocessing (StandardScaler, PCA, [0,2π] normalization) is fitted once on the
        full X_train before the CV split.  The k-fold CV therefore only cross-validates
        the SVC decision boundary on fixed quantum features, NOT the full pipeline.
        This is intentional: re-fitting TF-IDF + PCA inside each fold would be
        prohibitively expensive and the upstream features are stable across folds.
        The held-out X_val (never touched by preprocessing fit) provides the unbiased
        final evaluation.
        
        Args:
            X_train: Training features (raw TF-IDF + linguistic)
            X_val: Validation features (raw TF-IDF + linguistic, held out)
            y_train: Training labels
            y_val: Validation labels
            cv_folds: Number of cross-validation folds
            monitor: QuantumTrainingMonitor instance
            
        Returns:
            Dictionary containing training results with CV metrics
        """
        logger.info(f"Starting quantum SVM training with {cv_folds}-fold cross-validation monitoring...")
        logger.info("ℹ️  CV scope: inner CV on quantum SVC stage only (preprocessing fitted on full X_train)")
        if self.batch_training:
            logger.info(f"🔄 Batch training enabled for CV: {self.batch_size} splits")
        
        # Preprocessing fitted once on full X_train (scaler, PCA, [0,2π] norm).
        # CV folds below cross-validate the SVC on these fixed features.
        X_train_quantum, X_val_quantum = self._prepare_quantum_data(X_train, X_val)
        
        # Create quantum kernel
        self.create_quantum_kernel()
        
        # Use subset of training data if specified
        if self.sample_size != 'full':
            max_samples = int(self.sample_size)
            if len(X_train_quantum) > max_samples:
                from sklearn.model_selection import train_test_split
                X_train_quantum, _, y_train, _ = train_test_split(
                    X_train_quantum, y_train, 
                    train_size=max_samples,
                    stratify=y_train,
                    random_state=42
                )
                logger.info(f"Using {max_samples} samples for CV training")
        
        # Perform cross-validation with continuous monitoring
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        import threading
        import time
        import multiprocessing as mp

        cv_results = {
            'fold_results': [],
            'cv_train_scores': [],
            'cv_val_scores': [],
            'cv_times': []
        }

        overall_start_time = time.time()
        
        # Parallel CV is only safe with FidelityStatevectorKernel (picklable)
        # and only on platforms that support fork-based multiprocessing.
        # macOS (Python 3.8+) defaults to 'spawn' which requires full pickling
        # of quantum kernel objects -- often fails on C-extension internals.
        import sys
        use_parallelism = False
        if hasattr(self, 'slicing_info') and self.slicing_info:
            parallel_requested = (self.slicing_info.get('parallelism_mode') == 'statevector_cv_parallel' and 
                                  self.slicing_info.get('total_processes', 0) > 1)
            if parallel_requested:
                if sys.platform == 'darwin':
                    logger.warning("⚠️  Parallel CV requested but macOS uses 'spawn' multiprocessing -- falling back to sequential (pickle-safety)")
                else:
                    from qiskit_machine_learning.kernels import FidelityStatevectorKernel as _FSK
                    if isinstance(self.quantum_kernel, _FSK):
                        use_parallelism = True
                    else:
                        logger.warning("⚠️  Parallel CV requested but kernel is not FidelityStatevectorKernel -- falling back to sequential (pickling issue)")
        
        if use_parallelism:
            logger.info(f"🚀 Using parallel CV execution with {self.slicing_info.get('total_processes', cv_folds)} processes")
            cv_results = self._train_cv_parallel(X_train_quantum, y_train, cv_folds, monitor)
        else:
            logger.info(f"🔄 Using sequential CV execution with {cv_folds} folds")
            # Sequential execution (original implementation)
            cv_results = self._train_cv_sequential(X_train_quantum, y_train, cv_folds, monitor)
        
        # Train final model on full training set
        logger.info("Training final model on full training set...")
        
        # Track final model training time (equivalent to non-monitoring training)
        final_model_start_time = time.time()
        
        final_circuit_start = time.time()
        
        # Use helper for batch training support on final model
        self.qsvm_model = train_model_with_batching(
            X_train_quantum, y_train,
            kernel=self.quantum_kernel,
            batch_training=self.batch_training,
            n_batches=self.batch_size,
            C=self.C, class_weight='balanced'
        )
        
        final_circuit_time = time.time() - final_circuit_start
        
        # Evaluate final model
        final_pred_start = time.time()
        final_train_pred = self.qsvm_model.predict(X_train_quantum)
        final_val_pred = self.qsvm_model.predict(X_val_quantum)
        final_prediction_time = time.time() - final_pred_start
        
        # Calculate final model total time (training + evaluation)
        final_model_total_time = time.time() - final_model_start_time
        
        # Final metrics
        final_train_acc = accuracy_score(y_train, final_train_pred)
        final_val_acc = accuracy_score(y_val, final_val_pred)
        
        # Calculate CV statistics — guard against all folds failing (empty lists → NaN)
        if not cv_results['cv_train_scores']:
            raise RuntimeError(
                f"All {cv_folds} CV folds failed. Check fold error messages above for details."
            )
        cv_mean_train = np.mean(cv_results['cv_train_scores'])
        cv_std_train = np.std(cv_results['cv_train_scores'])
        cv_mean_val = np.mean(cv_results['cv_val_scores'])
        cv_std_val = np.std(cv_results['cv_val_scores'])
        
        total_time = time.time() - overall_start_time
        self.training_time = total_time
        
        # Log final checkpoint
        if monitor:
            final_metrics = {
                'final_model': True,
                'cv_mean_train_acc': round(cv_mean_train, 6),
                'cv_std_train_acc': round(cv_std_train, 6),
                'cv_mean_val_acc': round(cv_mean_val, 6),
                'cv_std_val_acc': round(cv_std_val, 6)
            }
            
            monitor.log_checkpoint(
                step=0,  # Final model step
                fold=0,  # Final model
                train_acc=final_train_acc,
                val_acc=final_val_acc,
                circuit_time=final_circuit_time,
                prediction_time=final_prediction_time,
                additional_metrics=final_metrics
            )
        
        # Prepare final results
        final_results = {
            'accuracy': final_val_acc,
            'precision': precision_score(y_val, final_val_pred, average='weighted'),
            'recall': recall_score(y_val, final_val_pred, average='weighted'),
            'f1_score': f1_score(y_val, final_val_pred, average='weighted'),
            'training_accuracy': final_train_acc,
            'training_precision': precision_score(y_train, final_train_pred, average='weighted'),
            'training_recall': recall_score(y_train, final_train_pred, average='weighted'),
            'training_f1': f1_score(y_train, final_train_pred, average='weighted'),
            'training_time': total_time,
            'circuit_training_time': final_circuit_time,
            'prediction_time': final_prediction_time,
            'final_model_time': final_model_total_time,  # Time for just the main model (monitoring OFF equivalent)
            'feature_dimension': self.feature_dim,
            'quantum_shots': self.shots,
            'pca_variance_explained': self.explained_variance,
            'samples_used': len(X_train_quantum),
            'validation_samples_used': len(X_val_quantum),
            'cv_results': cv_results,
            'cv_scope': 'inner_cv_on_quantum_svc_only',
            'cv_scope_note': 'CV cross-validates the SVC decision boundary on fixed quantum features; preprocessing (scaler, PCA, normalization) was fitted once on full X_train before folding. Held-out X_val provides unbiased final evaluation.',
            'cv_mean_train_accuracy': cv_mean_train,
            'cv_std_train_accuracy': cv_std_train,
            'cv_mean_val_accuracy': cv_mean_val,
            'cv_std_val_accuracy': cv_std_val,
            'cv_folds': cv_folds,
            'classification_report': classification_report(y_val, final_val_pred, output_dict=True)
        }
        
        logger.info(f"Cross-validation completed: {cv_mean_val:.4f}±{cv_std_val:.4f} validation accuracy")
        return final_results

    def predict(self, X_quantum: np.ndarray) -> np.ndarray:
        """
        Make predictions with quantum SVM
        
        Args:
            X_quantum: Quantum-processed features
            
        Returns:
            Predictions array
        """
        if self.qsvm_model is None:
            raise ValueError("Quantum SVM must be trained before making predictions")
        
        start_time = time.time()
        predictions = self.qsvm_model.predict(X_quantum)
        self.prediction_time = time.time() - start_time
        
        return predictions
    
    def evaluate(self, X_val_quantum: np.ndarray, y_val: np.ndarray, validation_size: str = 'full') -> Dict[str, Any]:
        """
        Evaluate quantum SVM on validation data
        
        Args:
            X_val_quantum: Quantum-processed validation features
            y_val: Validation labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating quantum SVM...")
        
        # Handle validation size selection - check if validation_size is set from parent call
        actual_validation_size = validation_size
        if hasattr(self, 'validation_size'):
            actual_validation_size = self.validation_size
            
        if actual_validation_size == 'full':
            X_val_subset = X_val_quantum
            y_val_subset = y_val
            logger.info(f"Using full validation set: {len(X_val_subset)} samples")
        else:
            # Use specified number of validation samples
            val_samples = int(actual_validation_size)
            max_val_samples = min(val_samples, len(X_val_quantum))
            
            if len(X_val_quantum) > max_val_samples:
                logger.info(f"Using {max_val_samples} validation samples from {len(X_val_quantum)} total")
                from sklearn.model_selection import train_test_split
                X_val_subset, _, y_val_subset, _ = train_test_split(
                    X_val_quantum, y_val, 
                    train_size=max_val_samples,
                    stratify=y_val,
                    random_state=42
                )
            else:
                X_val_subset = X_val_quantum
                y_val_subset = y_val
                logger.info(f"Using all {len(X_val_subset)} available validation samples")
        
        # Make predictions
        predictions = self.predict(np.array(X_val_subset))
        
        # Calculate metrics
        accuracy = accuracy_score(y_val_subset, predictions)
        precision = precision_score(y_val_subset, predictions, average='weighted')
        recall = recall_score(y_val_subset, predictions, average='weighted')
        f1 = f1_score(y_val_subset, predictions, average='weighted')
        
        # Generate detailed reports
        class_report = classification_report(y_val_subset, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y_val_subset, predictions)
        
        logger.info(f"Quantum SVM Validation Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'prediction_time': self.prediction_time,
            'predictions': predictions,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'model_type': 'Quantum SVM',
            'circuit_count': len(predictions),  # Each prediction requires circuit execution
            'validation_samples_used': len(X_val_subset)
        }
    
    def train_and_evaluate(self, X_train: np.ndarray, X_val: np.ndarray,
                          y_train: np.ndarray, y_val: np.ndarray, validation_size: str = 'full',
                          train_categories: Optional[List] = None, val_categories: Optional[List] = None) -> Dict[str, Any]:
        """
        Complete quantum SVM training and evaluation pipeline
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            
        Returns:
            Dictionary containing all results
        """
        # Store validation_size so evaluate() picks it up
        self.validation_size = validation_size

        # Track total time from start to finish (including setup, training, validation, and overhead)
        total_start_time = time.time()
        logger.info("Starting complete quantum SVM workflow...")

        # Train the model
        train_results = self.train(X_train, X_val, y_train, y_val, train_categories, val_categories)
        
        # If training failed, return early
        if train_results.get('error', False):
            return train_results
        
        # Calculate total time including all operations
        total_time = time.time() - total_start_time
        
        # train() already called evaluate(), so train_results has all metrics
        combined_results = dict(train_results)
        
        # Update timing information
        combined_results['circuit_training_time'] = combined_results.get('training_time', 0)  # Preserve original circuit training time
        combined_results['total_training_time'] = total_time  # Total time for all operations
        combined_results['training_time'] = total_time  # Replace with total time for main display
        
        logger.info(f"Complete quantum workflow finished in {total_time:.2f} seconds (circuit training: {combined_results.get('circuit_training_time', 0):.2f}s)")
        
        # Remove intermediate data to save memory
        del combined_results['X_train_quantum']
        del combined_results['X_val_quantum']
        
        # Add quantum-specific information
        combined_results.update({
            'n_qubits': self.feature_dim,
            'n_training_samples': X_train.shape[0],
            'n_validation_samples': X_val.shape[0],
            'quantum_backend': self.backend.name if hasattr(self.backend, 'name') else 'simulator'
        })
        
        return combined_results
    
    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get information about the quantum backend
        
        Returns:
            Dictionary containing backend information
        """
        try:
            backend_info = {
                'backend_name': self.backend.name if hasattr(self.backend, 'name') else 'Unknown',
                'backend_version': getattr(self.backend, 'version', 'Unknown'),
                'n_qubits': getattr(self.backend, 'num_qubits', self.feature_dim),
                'simulator': True,
                'shots': self.shots
            }
            
            return backend_info
            
        except Exception as e:
            logger.error(f"Error getting backend info: {e}")
            return {'error': str(e)}
    
    def visualize_quantum_circuit(self) -> str:
        """
        Get string representation of the quantum circuit
        
        Returns:
            String representation of quantum circuit
        """
        if self.feature_map is None:
            self.feature_map = self._create_feature_map()
        
        return str(self.feature_map)
    
    def get_quantum_advantage_metrics(self) -> Dict[str, Any]:
        """
        Calculate potential quantum advantage metrics
        
        Returns:
            Dictionary containing quantum-specific metrics
        """
        if self.feature_map is None:
            return {}
        
        circuit_depth = self.feature_map.depth()
        num_parameters = self.feature_map.num_parameters
        
        return {
            'circuit_depth': circuit_depth,
            'num_parameters': num_parameters,
            'entanglement_structure': 'linear',
            'feature_map_type': 'ZZFeatureMap',
            'quantum_dimension': 2 ** self.feature_dim
        }

    def _train_cv_parallel(self, X_train_quantum: np.ndarray, y_train: np.ndarray, 
                           cv_folds: int, monitor=None) -> Dict[str, Any]:
        """
        Train CV folds in parallel using multiprocessing
        
        Args:
            X_train_quantum: Quantum training features
            y_train: Training labels
            cv_folds: Number of CV folds
            monitor: Monitoring instance
            
        Returns:
            Dictionary containing parallel CV results
        """
        from sklearn.model_selection import StratifiedKFold
        import multiprocessing as mp
        
        logger.info(f"🚀 Starting parallel CV training with {cv_folds} folds")
        
        # Initialize CV results dictionary
        cv_results = {
            'fold_results': [],
            'cv_train_scores': [],
            'cv_val_scores': [],
            'cv_times': []
        }
        
        # Prepare fold data
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_data = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_quantum, y_train)):
            X_fold_train = X_train_quantum[train_idx]
            X_fold_val = X_train_quantum[val_idx]
            y_fold_train = y_train[train_idx]
            y_fold_val = y_train[val_idx]
            
            fold_data.append({
                'fold_idx': fold_idx,
                'X_train': X_fold_train,
                'X_val': X_fold_val,
                'y_train': y_fold_train,
                'y_val': y_fold_val,
                'quantum_kernel': self.quantum_kernel,
                'batch_training': self.batch_training,
                'batch_size': self.batch_size,
                'C': self.C,
                'class_weight': 'balanced'
            })
        
        # Use multiprocessing pool
        # Determine number of processes
        n_processes = min(cv_folds, mp.cpu_count())
        logger.info(f"🔄 Using {n_processes} parallel processes for CV training")
        
        with mp.Pool(processes=n_processes) as pool:
            fold_results = pool.map(train_single_fold_worker, fold_data)
        
        # Process results
        for fold_result in fold_results:
            if fold_result['success']:
                cv_results['fold_results'].append(fold_result)
                cv_results['cv_train_scores'].append(fold_result['train_accuracy'])
                cv_results['cv_val_scores'].append(fold_result['val_accuracy'])
                cv_results['cv_times'].append(fold_result['total_time'])
                
                if monitor:
                    monitor.log_checkpoint(
                        step=fold_result['fold'] - 1,
                        fold=fold_result['fold'],
                        train_acc=fold_result['train_accuracy'],
                        val_acc=fold_result['val_accuracy'],
                        circuit_time=fold_result['training_time'],
                        prediction_time=fold_result['prediction_time'],
                        additional_metrics={
                            'training_status': 'completed',
                            'fold_training_elapsed': fold_result['total_time'],
                            'fold_train_f1': round(fold_result.get('train_f1', 0), 6),
                            'fold_val_f1': round(fold_result.get('val_f1', 0), 6),
                            'execution_mode': 'parallel'
                        }
                    )
            else:
                logger.error(f"Fold {fold_result['fold']} failed: {fold_result.get('error', 'Unknown error')}")
        
        logger.info(f"✅ Parallel CV completed: {len(cv_results['fold_results'])}/{cv_folds} folds successful")
        return cv_results
    
    def _train_cv_sequential(self, X_train_quantum: np.ndarray, y_train: np.ndarray, 
                             cv_folds: int, monitor=None) -> Dict[str, Any]:
        """
        Train CV folds sequentially (original implementation)
        
        Args:
            X_train_quantum: Quantum training features
            y_train: Training labels
            cv_folds: Number of CV folds
            monitor: Monitoring instance
            
        Returns:
            Dictionary containing sequential CV results
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        import threading
        import time
        
        logger.info(f"🔄 Starting sequential CV training with {cv_folds} folds")
        
        cv_results = {
            'fold_results': [],
            'cv_train_scores': [],
            'cv_val_scores': [],
            'cv_times': []
        }
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        step = 0
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_quantum, y_train)):
            fold_start_time = time.time()
            
            # Split data for this fold
            X_fold_train, X_fold_val = X_train_quantum[train_idx], X_train_quantum[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Initialize QSVM for this fold (using helper for batch support)
            # Note: We can't easily use the train_model_with_batching here if we want continuous monitoring of progress
            # But for batching, progress monitoring is tricky anyway. 
            # Let's trust the helper.
            
            # Set up continuous monitoring during training
            training_complete = threading.Event()
            training_start_time = time.time()
            
            def continuous_monitor():
                """Monitor training progress every log_interval seconds"""
                monitor_step = 0
                _log_interval = getattr(monitor, 'log_interval', 30) if monitor else 30
                while not training_complete.is_set():
                    # Always block with a timeout so the thread never busy-loops
                    if training_complete.wait(timeout=_log_interval):
                        break  # Training completed

                    if monitor:
                        # Log intermediate progress (training in progress)
                        elapsed = time.time() - training_start_time
                        monitor.log_checkpoint(
                            step=f"{step}.{monitor_step}",
                            fold=fold_idx + 1,
                            train_acc=None,
                            val_acc=None,
                            circuit_time=elapsed,
                            prediction_time=None,
                            additional_metrics={
                                'training_status': 'in_progress',
                                'fold_training_elapsed': elapsed,
                                'fold_train_samples': len(X_fold_train),
                                'fold_val_samples': len(X_fold_val),
                                'monitor_step': monitor_step
                            }
                        )
                    monitor_step += 1
            
            # Start continuous monitoring in background
            if monitor:
                monitor_thread = threading.Thread(target=continuous_monitor, daemon=True)
                monitor_thread.start()
            
            # Train on fold
            circuit_start_time = time.time()
            try:
                fold_qsvm = train_model_with_batching(
                    X_fold_train, y_fold_train,
                    kernel=self.quantum_kernel,
                    batch_training=self.batch_training,
                    n_batches=self.batch_size,
                    C=self.C, class_weight='balanced'
                )
                circuit_time = time.time() - circuit_start_time
                training_success = True
            except Exception as e:
                circuit_time = time.time() - circuit_start_time
                training_success = False
                logger.error(f"Training failed for fold {fold_idx + 1}: {e}")
            finally:
                # Signal that training is complete
                training_complete.set()
                if monitor:
                    monitor_thread.join(timeout=1.0)  # Wait for monitor thread to finish
            
            if not training_success:
                continue  # Skip this fold if training failed
            
            # Evaluate on fold validation set
            pred_start_time = time.time()
            fold_train_pred = fold_qsvm.predict(X_fold_train)
            fold_val_pred = fold_qsvm.predict(X_fold_val)
            prediction_time = time.time() - pred_start_time
            
            # Calculate metrics
            fold_train_acc = accuracy_score(y_fold_train, fold_train_pred)
            fold_val_acc = accuracy_score(y_fold_val, fold_val_pred)
            
            fold_train_precision = precision_score(y_fold_train, fold_train_pred, average='weighted')
            fold_val_precision = precision_score(y_fold_val, fold_val_pred, average='weighted')
            
            fold_train_recall = recall_score(y_fold_train, fold_train_pred, average='weighted')
            fold_val_recall = recall_score(y_fold_val, fold_val_pred, average='weighted')
            
            fold_train_f1 = f1_score(y_fold_train, fold_train_pred, average='weighted')
            fold_val_f1 = f1_score(y_fold_val, fold_val_pred, average='weighted')
            
            fold_time = time.time() - fold_start_time
            
            # Store fold results
            fold_result = {
                'fold': fold_idx + 1,
                'train_accuracy': fold_train_acc,
                'val_accuracy': fold_val_acc,
                'train_precision': fold_train_precision,
                'val_precision': fold_val_precision,
                'train_recall': fold_train_recall,
                'val_recall': fold_val_recall,
                'train_f1': fold_train_f1,
                'val_f1': fold_val_f1,
                'training_time': circuit_time,
                'prediction_time': prediction_time,
                'total_time': fold_time
            }
            
            cv_results['fold_results'].append(fold_result)
            cv_results['cv_train_scores'].append(fold_train_acc)
            cv_results['cv_val_scores'].append(fold_val_acc)
            cv_results['cv_times'].append(fold_time)
            
            # Log fold completion to monitor CSV
            if monitor:
                monitor.log_checkpoint(
                    step=step,
                    fold=fold_idx + 1,
                    train_acc=fold_train_acc,
                    val_acc=fold_val_acc,
                    circuit_time=circuit_time,
                    prediction_time=prediction_time,
                    additional_metrics={
                        'training_status': 'completed',
                        'fold_training_elapsed': fold_time,
                        'fold_train_samples': len(X_fold_train),
                        'fold_val_samples': len(X_fold_val),
                        'fold_train_f1': round(fold_train_f1, 6),
                        'fold_val_f1': round(fold_val_f1, 6)
                    }
                )
            
            logger.info(f"Fold {fold_idx + 1}/{cv_folds} completed: "
                       f"Train Acc: {fold_train_acc:.4f}, Val Acc: {fold_val_acc:.4f}, "
                       f"Time: {fold_time:.2f}s")
            
            step += 1
        
        return cv_results
