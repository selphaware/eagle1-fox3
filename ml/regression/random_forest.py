"""
Random Forest Regression model implementation.

This module provides a wrapper for scikit-learn's RandomForestRegressor
with additional functionality for training, prediction, evaluation,
and feature importance extraction.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ml.metrics.regression import regression_metrics

logger = logging.getLogger(__name__)


class RandomForestRegressionModel:
    """
    Random Forest Regression model wrapper for scikit-learn's RandomForestRegressor.
    
    This class provides a consistent interface for training, prediction,
    evaluation, and feature importance extraction with proper input validation
    and error handling.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        max_features: Optional[Union[str, int, float]] = 'sqrt',
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the Random Forest Regression model.
        
        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of the trees. If None, nodes are expanded
                until all leaves are pure or contain min_samples_split samples.
            min_samples_split: Minimum number of samples required to split a node.
            min_samples_leaf: Minimum number of samples required at a leaf node.
            max_features: Number of features to consider for best split.
                If "sqrt", then max_features=sqrt(n_features).
                If "log2", then max_features=log2(n_features).
                If None, then max_features=n_features.
            random_state: Seed for random number generation.
            n_jobs: Number of jobs to run in parallel. None means using one core.
            **kwargs: Additional parameters to pass to RandomForestRegressor.
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )
        self.is_fitted = False
        self.feature_names: List[str] = []
        logger.info(
            f"Initialized RandomForestRegressionModel with n_estimators={n_estimators}, "
            f"max_depth={max_depth}, random_state={random_state}"
        )
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None,
        target_name: str = "target"
    ) -> "RandomForestRegressionModel":
        """
        Fit the Random Forest Regression model.
        
        Args:
            X: Features as DataFrame or numpy array.
            y: Target variable as Series or numpy array.
            feature_names: List of feature names when X is a numpy array.
                If X is a DataFrame, feature_names are extracted from it.
            target_name: Name of the target variable.
            
        Returns:
            Self for method chaining.
            
        Raises:
            ValueError: If input validation fails.
            TypeError: If input types are not supported.
        """
        # Validate inputs
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            self.feature_names = X.columns.tolist()
        elif isinstance(X, np.ndarray):
            X_array = X
            if feature_names is not None:
                if len(feature_names) != X.shape[1]:
                    raise ValueError(
                        f"Length of feature_names ({len(feature_names)}) does not match "
                        f"number of features in X ({X.shape[1]})"
                    )
                self.feature_names = feature_names
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            raise TypeError(
                f"X must be a pandas DataFrame or numpy array, got {type(X)}"
            )
        
        if isinstance(y, pd.Series):
            y_array = y.values
            self.target_name = y.name if y.name is not None else target_name
        elif isinstance(y, np.ndarray):
            y_array = y
            self.target_name = target_name
        else:
            raise TypeError(
                f"y must be a pandas Series or numpy array, got {type(y)}"
            )
        
        # Fit the model
        try:
            self.model.fit(X_array, y_array)
            self.is_fitted = True
            logger.info(
                f"Fitted RandomForestRegressionModel on {X_array.shape[0]} samples "
                f"with {X_array.shape[1]} features"
            )
            return self
        except Exception as e:
            logger.error(f"Error fitting RandomForestRegressionModel: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X: Features as DataFrame or numpy array.
            
        Returns:
            Array of predictions.
            
        Raises:
            ValueError: If model is not fitted or input validation fails.
            TypeError: If input types are not supported.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Validate input
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        elif isinstance(X, np.ndarray):
            X_array = X
        else:
            raise TypeError(
                f"X must be a pandas DataFrame or numpy array, got {type(X)}"
            )
        
        # Check feature count
        if X_array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Number of features in X ({X_array.shape[1]}) does not match "
                f"number of features the model was trained on ({len(self.feature_names)})"
            )
        
        # Make predictions
        try:
            predictions = self.model.predict(X_array)
            logger.info(f"Made predictions for {X_array.shape[0]} samples")
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_n_steps_ahead(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        n_steps: int,
        feature_update_func: Optional[callable] = None
    ) -> np.ndarray:
        """
        Make predictions for n steps ahead using recursive forecasting.
        
        This method predicts multiple steps into the future by recursively using
        previous predictions as inputs for subsequent predictions. It's particularly
        useful for time series forecasting.
        
        Args:
            X: Initial features to predict on (typically the last known data point).
            n_steps: Number of steps to predict ahead.
            feature_update_func: Optional function that takes (features, prediction, step_index)
                                and returns updated features for the next prediction step.
                                If None, a simple shift strategy is used for time series data.
            
        Returns:
            Array of n_steps predictions.
            
        Raises:
            ValueError: If model is not fitted, if X has wrong shape, or if n_steps < 1.
            TypeError: If X is not a pandas DataFrame or numpy ndarray.
        """
        # Check if model is fitted
        if not self.is_fitted:
            logger.error("Model must be fitted before making predictions")
            raise ValueError("Model must be fitted before making predictions")
        
        # Validate input
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            logger.error("X must be a pandas DataFrame or numpy ndarray")
            raise TypeError("X must be a pandas DataFrame or numpy ndarray")
            
        if n_steps < 1:
            logger.error("n_steps must be at least 1")
            raise ValueError("n_steps must be at least 1")
        
        # Convert to numpy array if needed and ensure we have a copy to modify
        if isinstance(X, pd.DataFrame):
            features = X.copy()
            is_dataframe = True
        else:
            features = np.copy(X)
            is_dataframe = False
            
        # Prepare storage for predictions
        predictions = np.zeros(n_steps)
        
        logger.info(f"Making {n_steps}-step ahead predictions")
        
        try:
            for i in range(n_steps):
                # Make single-step prediction
                current_pred = self.predict(features)
                predictions[i] = current_pred[0]  # Store the prediction
                
                # Update features for next prediction
                if feature_update_func is not None:
                    # Use custom function to update features
                    features = feature_update_func(features, current_pred, i)
                else:
                    # Default behavior: shift features and add new prediction as most recent feature
                    # This assumes time series data where each row is a time step
                    if is_dataframe:
                        # For DataFrame, create a new row with properly typed values
                        # Get column names for clearer indexing
                        col_names = features.columns.tolist()
                        
                        # Create a dictionary with the updated values
                        new_values = {}
                        
                        # Shift values (all except last column)
                        for i in range(len(col_names) - 1):
                            new_values[col_names[i]] = features.iloc[0, i+1]
                            
                        # Set the last column to the prediction, ensuring proper type conversion
                        last_col = col_names[-1]
                        # Convert to the same dtype as the target column
                        pred_value = np.array([current_pred[0]]).astype(features[last_col].dtype)[0]
                        new_values[last_col] = pred_value
                        
                        # Update the DataFrame row with the new values
                        for col, val in new_values.items():
                            features.loc[0, col] = val
                    else:
                        # For numpy array, shift values and update the last column
                        features[0, :-1] = features[0, 1:]
                        features[0, -1] = current_pred[0]
            
            logger.info(f"{n_steps}-step ahead predictions generated successfully")
            return predictions
        except Exception as e:
            logger.error(f"Error making n-step ahead predictions: {str(e)}")
            raise
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate the model using regression metrics.
        
        Args:
            X: Features as DataFrame or numpy array.
            y: True target values as Series or numpy array.
            
        Returns:
            Dictionary of regression metrics.
            
        Raises:
            ValueError: If model is not fitted or input validation fails.
            TypeError: If input types are not supported.
        """
        y_pred = self.predict(X)
        
        # Convert y to numpy array if it's a pandas Series
        if isinstance(y, pd.Series):
            y_true = y.values
        elif isinstance(y, np.ndarray):
            y_true = y
        else:
            raise TypeError(
                f"y must be a pandas Series or numpy array, got {type(y)}"
            )
        
        # Calculate metrics
        metrics = regression_metrics(y_true, y_pred)
        logger.info(
            f"Evaluated model: MSE={metrics['mse']:.4f}, "
            f"RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}"
        )
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            DataFrame with feature names and their importance scores,
            sorted by importance in descending order.
            
        Raises:
            ValueError: If model is not fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Extract feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame with feature names and importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        # Sort by importance in descending order
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        logger.info(f"Extracted feature importance for {len(self.feature_names)} features")
        return importance_df
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model.
        
        Returns:
            Dictionary with model information.
            
        Raises:
            ValueError: If model is not fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting summary")
        
        # Get model parameters
        params = self.model.get_params()
        
        # Create summary dictionary
        summary = {
            'model_type': 'RandomForestRegression',
            'n_estimators': params['n_estimators'],
            'max_depth': params['max_depth'],
            'min_samples_split': params['min_samples_split'],
            'min_samples_leaf': params['min_samples_leaf'],
            'max_features': params['max_features'],
            'random_state': params['random_state'],
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        logger.info(f"Generated model summary for RandomForestRegressionModel")
        return summary
