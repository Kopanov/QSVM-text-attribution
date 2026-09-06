import numpy as np
import time
import logging
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ClassicalSVM:
    """
    Classical SVM implementation with the exact parameters specified in the requirements
    """
    
    def __init__(self, C: float = 1.0):
        """Initialize classical SVM with specified hyperparameters"""
        self.svm_params = {
            'C': C,
            'class_weight': 'balanced',
            'dual': False,
            'max_iter': 2000,
            'penalty': 'l2',
            'tol': 1e-4,
            'loss': 'squared_hinge',
            'random_state': 42
        }
        
        # Calibration parameters — n_jobs only available in sklearn >= 1.2
        import sklearn
        _cal_params = {'method': 'isotonic', 'cv': 5}
        if tuple(int(x) for x in sklearn.__version__.split('.')[:2]) >= (1, 2):
            _cal_params['n_jobs'] = 1
        self.calibration_params = _cal_params
        
        self.model = None
        self.training_time = None
        self.prediction_time = None
    
    def create_model(self):
        """Create the calibrated SVM model"""
        # Create base LinearSVC
        base_svm = LinearSVC(**self.svm_params)
        
        # Wrap with calibration
        self.model = CalibratedClassifierCV(
            base_svm, 
            **self.calibration_params
        )
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train the classical SVM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary containing training metrics and model info
        """
        logger.info("Starting classical SVM training...")
        
        # Create model
        self.create_model()
        
        # Ensure model is created
        if self.model is None:
            raise ValueError("Model not properly initialized")
            
        # Measure training time
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        
        logger.info(f"Classical SVM training completed in {self.training_time:.2f} seconds")
        
        # Get training predictions for metrics
        start_pred_time = time.time()
        train_predictions = self.model.predict(X_train)
        train_pred_time = time.time() - start_pred_time
        
        # Calculate training metrics
        train_accuracy = accuracy_score(y_train, train_predictions)
        train_precision = precision_score(y_train, train_predictions, average='weighted')
        train_recall = recall_score(y_train, train_predictions, average='weighted')
        train_f1 = f1_score(y_train, train_predictions, average='weighted')
        
        return {
            'training_accuracy': train_accuracy,
            'training_precision': train_precision,
            'training_recall': train_recall,
            'training_f1': train_f1,
            'training_time': self.training_time,
            'model_params': self.svm_params,
            'calibration_params': self.calibration_params
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with the trained model
        
        Args:
            X: Features to predict
            
        Returns:
            Tuple of (predictions, prediction_probabilities)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        start_time = time.time()
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        self.prediction_time = time.time() - start_time
        
        return predictions, probabilities
    
    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on validation data
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating classical SVM...")
        
        # Make predictions
        predictions, probabilities = self.predict(X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, predictions)
        precision = precision_score(y_val, predictions, average='weighted')
        recall = recall_score(y_val, predictions, average='weighted')
        f1 = f1_score(y_val, predictions, average='weighted')
        
        # Generate detailed reports
        class_report = classification_report(y_val, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y_val, predictions)
        
        logger.info(f"Classical SVM Validation Results:")
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
            'probabilities': probabilities,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'model_type': 'Classical SVM'
        }
    
    def train_and_evaluate(self, X_train: np.ndarray, X_val: np.ndarray, 
                          y_train: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Complete training and evaluation pipeline
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            
        Returns:
            Dictionary containing all training and evaluation results
        """
        # Train the model
        train_results = self.train(X_train, y_train)
        
        # Evaluate the model
        eval_results = self.evaluate(X_val, y_val)
        
        # Combine results
        combined_results = {**train_results, **eval_results}
        
        # Add model-specific information
        combined_results.update({
            'support_vectors': 'N/A (LinearSVC)',
            'n_features': X_train.shape[1],
            'n_training_samples': X_train.shape[0],
            'n_validation_samples': X_val.shape[0]
        })
        
        return combined_results
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance from the trained SVM
        
        Returns:
            Array of feature importance scores
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        # CalibratedClassifierCV wraps the actual estimator; access it via .estimator
        calibrated = self.model.calibrated_classifiers_[0]
        inner = getattr(calibrated, 'estimator', getattr(calibrated, 'base_estimator', calibrated))
        if hasattr(inner, 'coef_'):
            return np.abs(inner.coef_[0])
        else:
            raise ValueError("Trained model does not have feature importance (coef_ not found)")
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        import joblib
        if self.model is None:
            raise ValueError("No trained model to save")
        joblib.dump(self.model, filepath)
        logger.info(f"Classical SVM model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        import joblib
        self.model = joblib.load(filepath)
        logger.info(f"Classical SVM model loaded from {filepath}")
