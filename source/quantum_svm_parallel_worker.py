"""
Parallel worker functions for quantum SVM training
This module contains picklable functions for multiprocessing
"""

import time
import numpy as np
from scipy.stats import mode
from qiskit_machine_learning.algorithms import QSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

class QuantumEnsemble:
    """
    Ensemble of Quantum SVM models trained on data batches
    """
    def __init__(self, models, classes=None):
        self.models = models
        self.classes_ = classes
    
    def predict(self, X):
        """
        Predict using majority voting from all models
        """
        if not self.models:
            return np.array([])
        
        # Get predictions from all models: shape (n_models, n_samples)
        # Note: This sequentially calls predict on each model. 
        # Since predictions are computed on the quantum kernel, this might be slow if not optimized.
        preds = np.array([model.predict(X) for model in self.models])
        
        final_preds, _ = mode(preds, axis=0, keepdims=False)
        return final_preds  # preserve original dtype; do not cast to int
    
    def predict_proba(self, X):
        """
        Predict class probabilities by averaging across all ensemble models.
        Falls back to hard-vote fractions if sub-models lack predict_proba.
        """
        if not self.models:
            return np.array([])
        
        if hasattr(self.models[0], 'predict_proba'):
            probas = np.array([model.predict_proba(X) for model in self.models])
            return probas.mean(axis=0)
        
        preds = np.array([model.predict(X) for model in self.models])
        # Use known training classes (not just those in predictions) for consistent output shape
        if self.classes_ is not None:
            all_classes = sorted(self.classes_)
        else:
            all_classes = sorted(np.unique(preds))
        n_classes = len(all_classes)
        n_samples = preds.shape[1]
        proba = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            for cls_idx, cls in enumerate(all_classes):
                proba[i, cls_idx] = np.mean(preds[:, i] == cls)
        return proba

def train_model_with_batching(X, y, kernel, batch_training=False, n_batches=5, C=1.0, class_weight='balanced'):
    """
    Train a model (single or ensemble) based on batching configuration
    """
    if not batch_training:
        model = QSVC(quantum_kernel=kernel, C=C, class_weight=class_weight)
        model.fit(X, y)
        return model
    
    # Batch training (Ensemble)
    n_samples = len(X)
    
    # Ensure n_batches is valid
    n_splits = min(n_batches, n_samples)
    if n_splits < 2:
         # Fallback to single model if we can't split
         model = QSVC(quantum_kernel=kernel, C=C, class_weight=class_weight)
         model.fit(X, y)
         return model

    # Use StratifiedKFold to create balanced batches
    # We use the TEST indices as the training batch to get 1/k splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []
    
    # Iterate through folds and train a model on each 'test' split (which is 1/k of data)
    for _, batch_idx in skf.split(X, y):
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]
        
        model = QSVC(quantum_kernel=kernel, C=C, class_weight=class_weight)
        model.fit(X_batch, y_batch)
        models.append(model)
        
    return QuantumEnsemble(models, classes=np.unique(y))

def train_single_fold_worker(fold_data_dict):
    """
    Worker function to train a single CV fold
    This function must be at module level to be picklable
    
    Args:
        fold_data_dict: Dictionary containing fold data
        
    Returns:
        Dictionary with fold training results
    """
    fold_idx = -1  # sentinel so the except block can always reference it
    try:
        fold_idx = fold_data_dict['fold_idx']
        X_fold_train = fold_data_dict['X_train']
        X_fold_val = fold_data_dict['X_val']
        y_fold_train = fold_data_dict['y_train']
        y_fold_val = fold_data_dict['y_val']
        quantum_kernel = fold_data_dict['quantum_kernel']
        
        batch_training = fold_data_dict.get('batch_training', False)
        batch_size = fold_data_dict.get('batch_size', 5)
        C = fold_data_dict.get('C', 1.0)
        class_weight = fold_data_dict.get('class_weight', 'balanced')
        
        fold_start_time = time.time()
        
        fold_qsvm = train_model_with_batching(
            X_fold_train, y_fold_train, 
            kernel=quantum_kernel,
            batch_training=batch_training,
            n_batches=batch_size,
            C=C,
            class_weight=class_weight
        )
        
        training_time = time.time() - fold_start_time
        
        # Make predictions
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
        
        return {
            'fold': fold_idx + 1,
            'train_accuracy': fold_train_acc,
            'val_accuracy': fold_val_acc,
            'train_precision': fold_train_precision,
            'val_precision': fold_val_precision,
            'train_recall': fold_train_recall,
            'val_recall': fold_val_recall,
            'train_f1': fold_train_f1,
            'val_f1': fold_val_f1,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'total_time': fold_time,
            'success': True
        }
        
    except Exception as e:
        import traceback
        return {
            'fold': fold_idx + 1,
            'success': False,
            'error': f"{str(e)}\n{traceback.format_exc()}"
        }
