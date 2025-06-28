"""
Random Forest Classifier implementation.

This module provides a RandomForestClassifier class with a standardized
interface for training, prediction, and evaluation.
"""

__author__ = "Usman Ahmad"

from typing import Dict, List, Union, Optional, Tuple, Any, cast
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import logging

from ml.metrics import classification_metrics

# Configure logging
logger = logging.getLogger(__name__)


class RandomForestClassifier:
    """
    Random Forest Classifier with standardized interface for training,
    prediction, and evaluation.
    
    This class wraps scikit-learn's RandomForestClassifier with additional
    functionality for hyperparameter tuning, evaluation, and consistent
    interface with other models.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[str, int, float]] = 'sqrt',
        bootstrap: bool = True,
        criterion: str = 'gini',
        random_state: Optional[int] = None,
        class_weight: Optional[Union[Dict[int, float], str]] = None
    ) -> None:
        """
        Initialize a RandomForestClassifier with specified parameters.
        
        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of the trees. None means unlimited.
            min_samples_split: Minimum samples required to split a node.
            min_samples_leaf: Minimum samples required at a leaf node.
            max_features: Number of features to consider for best split.
            bootstrap: Whether to use bootstrap samples.
            criterion: Function to measure split quality ('gini' or 'entropy').
            random_state: Seed for random number generation.
            class_weight: Weights for classes. If 'balanced', uses class frequencies.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.random_state = random_state
        self.class_weight = class_weight
        
        # Initialize model
        self.model = SklearnRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            criterion=criterion,
            random_state=random_state,
            class_weight=class_weight
        )
        
        # Initialize pipeline
        self.pipeline = Pipeline([
            ('classifier', self.model)
        ])
        
        # Track if model has been trained
        self.is_trained = False
        
        logger.info(f"Initialized RandomForestClassifier with n_estimators={n_estimators}, "
                   f"max_depth={max_depth}, criterion={criterion}")
    
    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, List],
        tune_hyperparameters: bool = False,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train the random forest classifier on the provided data.
        
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
                - 'feature_importance': Feature importance scores
        """
        # Validate inputs
        self._validate_inputs(X, y)
        
        # Convert inputs to numpy arrays for consistency
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else np.array(y)
        
        # Get feature names if available
        feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        
        # Log training start
        logger.info(f"Training RandomForestClassifier on data with shape X: {X_array.shape}, "
                   f"y: {y_array.shape}")
        
        results: Dict[str, Any] = {}
        
        try:
            if tune_hyperparameters:
                # Define parameter grid for tuning
                param_grid = {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [None, 10, 20, 30],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4],
                    'classifier__max_features': ['sqrt', 'log2', None]
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
                self.model = cast(SklearnRandomForestClassifier, self.pipeline.named_steps['classifier'])
                
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
            
            # Get feature importance
            if feature_names:
                results['feature_importance'] = self.get_feature_importance(feature_names)
            
            logger.info("RandomForestClassifier training completed successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training RandomForestClassifier: {str(e)}")
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
        
        logger.info(f"Making predictions with RandomForestClassifier on data with shape: {X_array.shape}")
        
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
        
        logger.info(f"Predicting probabilities with RandomForestClassifier on data with shape: {X_array.shape}")
        
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
        
        logger.info(f"Evaluating RandomForestClassifier on data with shape X: {X_array.shape}, "
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
        Get feature importance from the trained random forest model.
        
        Args:
            feature_names: List of feature names. If None, generic names will be used.
            
        Returns:
            DataFrame with feature names and their importance scores,
            sorted by importance in descending order.
            
        Raises:
            ValueError: If model has not been trained.
        """
        if not self.is_trained:
            logger.error("Model has not been trained. Call train() before get_feature_importance()")
            raise ValueError("Model has not been trained. Call train() before get_feature_importance()")
        
        try:
            # Get feature importances from the model
            importances = self.model.feature_importances_
            
            # Create feature names if not provided
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            
            # Ensure we have the right number of feature names
            if len(feature_names) != len(importances):
                error_msg = f"Length mismatch between feature_names ({len(feature_names)}) and feature_importances ({len(importances)})"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Create DataFrame with feature importances
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
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
