"""
Classification models for machine learning.

This module provides implementations of various classification models
with consistent interfaces for training, prediction, and evaluation.
"""

__author__ = "Usman Ahmad"

from typing import Dict, List, Union, Optional, Tuple, Any, cast
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import logging

from ml.metrics import classification_metrics

# Configure logging
logger = logging.getLogger(__name__)


class LogisticRegressionModel:
    """
    Logistic Regression model with standardized interface for training,
    prediction, and evaluation.
    
    This class wraps scikit-learn's LogisticRegression with additional
    functionality for hyperparameter tuning, evaluation, and consistent
    interface with other models.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        penalty: str = 'l2',
        solver: str = 'lbfgs',
        max_iter: int = 1000,
        random_state: Optional[int] = None,
        class_weight: Optional[Union[Dict[int, float], str]] = None,
        multi_class: str = 'auto'
    ) -> None:
        """
        Initialize a LogisticRegressionModel with specified parameters.
        
        Args:
            C: Inverse of regularization strength; smaller values specify stronger regularization.
            penalty: Penalty norm to use ('l1', 'l2', 'elasticnet', or 'none').
            solver: Algorithm for optimization problem.
            max_iter: Maximum number of iterations for solver.
            random_state: Seed for random number generation.
            class_weight: Weights associated with classes. If 'balanced', uses class frequencies.
            multi_class: Strategy for multi-class classification ('auto', 'ovr', or 'multinomial').
        """
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.class_weight = class_weight
        self.multi_class = multi_class
        
        # Initialize model
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
            class_weight=class_weight,
            multi_class=multi_class
        )
        
        # Initialize pipeline with scaling
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', self.model)
        ])
        
        # Track if model has been trained
        self.is_trained = False
        
        logger.info(f"Initialized LogisticRegressionModel with C={C}, penalty={penalty}, "
                   f"solver={solver}, max_iter={max_iter}")
    
    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, List],
        tune_hyperparameters: bool = False,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train the logistic regression model on the provided data.
        
        Args:
            X: Feature matrix for training.
            y: Target variable for training.
            tune_hyperparameters: Whether to perform hyperparameter tuning.
            cv_folds: Number of cross-validation folds for hyperparameter tuning.
            
        Returns:
            Dictionary containing training results:
                - 'model': Trained model
                - 'best_params': Best parameters if tuning was performed
                - 'cv_results': Cross-validation results if tuning was performed
        """
        # Validate inputs
        self._validate_inputs(X, y)
        
        # Convert inputs to numpy arrays for consistency
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else np.array(y)
        
        # Log training start
        logger.info(f"Training LogisticRegressionModel on data with shape X: {X_array.shape}, "
                   f"y: {y_array.shape}")
        
        results: Dict[str, Any] = {}
        
        try:
            if tune_hyperparameters:
                # Define parameter grid for tuning
                param_grid = {
                    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'classifier__penalty': ['l2'],  # 'l1' requires 'liblinear' solver
                    'classifier__solver': ['lbfgs', 'newton-cg', 'sag']
                }
                
                logger.info(f"Performing hyperparameter tuning with {cv_folds}-fold CV")
                
                # Create grid search
                grid_search = GridSearchCV(
                    self.pipeline,
                    param_grid,
                    cv=cv_folds,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1
                )
                
                # Fit grid search
                grid_search.fit(X_array, y_array)
                
                # Get best model
                self.pipeline = grid_search.best_estimator_
                self.model = cast(LogisticRegression, self.pipeline.named_steps['classifier'])
                
                # Store results
                results['best_params'] = grid_search.best_params_
                results['cv_results'] = grid_search.cv_results_
                
                logger.info(f"Best parameters found: {grid_search.best_params_}")
                logger.info(f"Best CV accuracy: {grid_search.best_score_:.4f}")
            else:
                # Train model without tuning
                self.pipeline.fit(X_array, y_array)
            
            # Mark model as trained
            self.is_trained = True
            
            # Store model in results
            results['model'] = self.pipeline
            
            logger.info("LogisticRegressionModel training completed successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training LogisticRegressionModel: {str(e)}")
            raise
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix for prediction.
            
        Returns:
            Array of predicted class labels.
            
        Raises:
            ValueError: If model has not been trained.
        """
        if not self.is_trained:
            logger.error("Model has not been trained. Call train() before predict()")
            raise ValueError("Model has not been trained. Call train() before predict()")
        
        # Convert input to numpy array if DataFrame
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        logger.info(f"Making predictions with LogisticRegressionModel on data with shape: {X_array.shape}")
        
        try:
            predictions = self.pipeline.predict(X_array)
            logger.info(f"Predictions completed. Output shape: {predictions.shape}")
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict class probabilities using the trained model.
        
        Args:
            X: Feature matrix for prediction.
            
        Returns:
            Array of predicted class probabilities.
            
        Raises:
            ValueError: If model has not been trained.
        """
        if not self.is_trained:
            logger.error("Model has not been trained. Call train() before predict_proba()")
            raise ValueError("Model has not been trained. Call train() before predict_proba()")
        
        # Convert input to numpy array if DataFrame
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        logger.info(f"Predicting probabilities with LogisticRegressionModel on data with shape: {X_array.shape}")
        
        try:
            probabilities = self.pipeline.predict_proba(X_array)
            logger.info(f"Probability predictions completed. Output shape: {probabilities.shape}")
            return probabilities
        except Exception as e:
            logger.error(f"Error predicting probabilities: {str(e)}")
            raise
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, List],
        average: str = 'binary'
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Feature matrix for evaluation.
            y: True target values.
            average: Parameter for precision, recall, and f1 score.
                Options: 'binary', 'micro', 'macro', 'weighted', 'samples'.
                
        Returns:
            Dictionary containing evaluation metrics.
            
        Raises:
            ValueError: If model has not been trained.
        """
        if not self.is_trained:
            logger.error("Model has not been trained. Call train() before evaluate()")
            raise ValueError("Model has not been trained. Call train() before evaluate()")
        
        # Validate inputs
        self._validate_inputs(X, y)
        
        # Convert inputs to numpy arrays for consistency
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else np.array(y)
        
        logger.info(f"Evaluating LogisticRegressionModel on data with shape X: {X_array.shape}, "
                   f"y: {y_array.shape}")
        
        try:
            # Make predictions
            y_pred = self.predict(X_array)
            
            # Get probabilities for ROC AUC (only for binary classification)
            y_prob = None
            if len(np.unique(y_array)) == 2:
                y_prob = self.predict_proba(X_array)[:, 1]  # Probability of positive class
            
            # Calculate metrics
            metrics = classification_metrics(y_array, y_pred, y_prob, average=average)
            
            logger.info(f"Evaluation completed. Accuracy: {metrics['accuracy']:.4f}, "
                       f"Precision: {metrics['precision']:.4f}, "
                       f"Recall: {metrics['recall']:.4f}, "
                       f"F1: {metrics['f1']:.4f}")
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get feature importance from the trained logistic regression model.
        
        For logistic regression, the coefficients are used as feature importance.
        For multi-class problems, the average absolute coefficient across classes is used.
        
        Args:
            feature_names: List of feature names. If None, generic names will be used.
            
        Returns:
            DataFrame with feature names and their importance scores,
            sorted by absolute importance in descending order.
            
        Raises:
            ValueError: If model has not been trained.
        """
        if not self.is_trained:
            logger.error("Model has not been trained. Call train() before get_feature_importance()")
            raise ValueError("Model has not been trained. Call train() before get_feature_importance()")
        
        try:
            # Get coefficients from the model
            coefficients = self.model.coef_
            
            # For multi-class, take the average absolute coefficient across classes
            if len(coefficients.shape) > 1 and coefficients.shape[0] > 1:
                importance = np.mean(np.abs(coefficients), axis=0)
            else:
                importance = np.abs(coefficients[0])
            
            # Create feature names if not provided
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importance))]
            
            # Ensure we have the right number of feature names
            if len(feature_names) != len(importance):
                logger.warning(f"Length mismatch between feature_names ({len(feature_names)}) "
                              f"and coefficients ({len(importance)})")
                feature_names = [f"feature_{i}" for i in range(len(importance))]
            
            # Create DataFrame with feature importances
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })
            
            # Sort by importance in descending order
            importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
            
            logger.info(f"Feature importance extracted for {len(feature_names)} features")
            
            return importance_df
        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
            raise
    
    def _validate_inputs(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, List]
    ) -> None:
        """
        Validate input data for training and evaluation.
        
        Args:
            X: Feature matrix.
            y: Target variable.
            
        Raises:
            TypeError: If input types are invalid.
            ValueError: If inputs have incompatible shapes.
        """
        # Check X type
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            logger.error("TypeError: X must be a pandas DataFrame or numpy array")
            raise TypeError("X must be a pandas DataFrame or numpy array")
        
        # Check y type
        if not isinstance(y, (pd.Series, np.ndarray, list)):
            logger.error("TypeError: y must be a pandas Series, numpy array, or list")
            raise TypeError("y must be a pandas Series, numpy array, or list")
        
        # Check shapes
        X_shape = X.shape[0]
        y_shape = len(y)
        
        if X_shape != y_shape:
            logger.error(f"ValueError: X and y have incompatible shapes: {X_shape} vs {y_shape}")
            raise ValueError(f"X and y have incompatible shapes: {X_shape} vs {y_shape}")
