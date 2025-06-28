"""
Linear Regression model implementation.

This module provides a wrapper around scikit-learn's LinearRegression
with additional functionality for financial data analysis.
"""

__author__ = "Usman Ahmad"

from typing import Dict, List, Union, Optional, Tuple, Any, cast
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import logging
from ml.metrics.regression import regression_metrics

# Configure logging
logger = logging.getLogger(__name__)


class LinearRegressionModel:
    """
    Linear Regression model for financial data analysis.
    
    This class provides a wrapper around scikit-learn's LinearRegression
    with additional functionality for model evaluation, coefficient analysis,
    and prediction.
    
    Attributes:
        model: Trained scikit-learn LinearRegression model.
        feature_names: List of feature names used during training.
        target_name: Name of the target variable.
        is_fitted: Boolean indicating if the model has been trained.
    """
    
    def __init__(
        self,
        fit_intercept: bool = True,
        copy_X: bool = True,
        n_jobs: Optional[int] = None
    ) -> None:
        """
        Initialize a LinearRegressionModel.
        
        Args:
            fit_intercept: Whether to calculate the intercept for this model.
                If set to False, no intercept will be used in calculations.
            copy_X: If True, X will be copied; else, it may be overwritten.
            n_jobs: The number of jobs to use for the computation.
                None means 1 unless in a joblib.parallel_backend context.
                -1 means using all processors.
        """
        logger.info("Initializing LinearRegressionModel")
        self.model = LinearRegression(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            n_jobs=n_jobs
        )
        self.feature_names: List[str] = []
        self.target_name: str = ""
        self.is_fitted: bool = False
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None,
        target_name: str = "target"
    ) -> "LinearRegressionModel":
        """
        Fit the linear regression model.
        
        Args:
            X: Training data features.
            y: Target values.
            feature_names: Names of features. If None and X is a DataFrame,
                column names will be used.
            target_name: Name of the target variable.
            
        Returns:
            Self for method chaining.
            
        Raises:
            ValueError: If X and y have incompatible shapes or if feature_names
                length doesn't match X columns.
            TypeError: If inputs are not of expected types.
        """
        # Validate inputs
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            logger.error("X must be a pandas DataFrame or numpy ndarray")
            raise TypeError("X must be a pandas DataFrame or numpy ndarray")
        
        if not isinstance(y, (pd.Series, np.ndarray)):
            logger.error("y must be a pandas Series or numpy ndarray")
            raise TypeError("y must be a pandas Series or numpy ndarray")
        
        # Get feature names if X is a DataFrame
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Validate feature names
        if len(feature_names) != (X.shape[1] if hasattr(X, 'shape') else len(X[0])):
            logger.error(
                f"Feature names length ({len(feature_names)}) doesn't match "
                f"X columns ({X.shape[1] if hasattr(X, 'shape') else len(X[0])})"
            )
            raise ValueError(
                f"Feature names length ({len(feature_names)}) doesn't match "
                f"X columns ({X.shape[1] if hasattr(X, 'shape') else len(X[0])})"
            )
        
        # Convert to numpy arrays if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Fit the model
        logger.info(f"Fitting LinearRegressionModel with {len(feature_names)} features")
        try:
            self.model.fit(X_array, y_array)
            self.feature_names = feature_names
            self.target_name = target_name
            self.is_fitted = True
            logger.info("LinearRegressionModel fitted successfully")
            return self
        except Exception as e:
            logger.error(f"Error fitting LinearRegressionModel: {str(e)}")
            raise
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on.
            
        Returns:
            Array of predictions.
            
        Raises:
            ValueError: If model is not fitted or if X has wrong shape.
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
        
        # Convert to numpy array if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Validate shape
        expected_features = len(self.feature_names)
        if X_array.shape[1] != expected_features:
            logger.error(
                f"X has {X_array.shape[1]} features, but model was trained with "
                f"{expected_features} features"
            )
            raise ValueError(
                f"X has {X_array.shape[1]} features, but model was trained with "
                f"{expected_features} features"
            )
        
        # Make predictions
        logger.info(f"Making predictions for {X_array.shape[0]} samples")
        try:
            predictions = self.model.predict(X_array)
            logger.info("Predictions generated successfully")
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Union[float, str]]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features.
            y: True target values.
            
        Returns:
            Dictionary of evaluation metrics including MSE, RMSE, MAE, R², etc.
            
        Raises:
            ValueError: If model is not fitted or if inputs have wrong shapes.
            TypeError: If inputs are not of expected types.
        """
        # Check if model is fitted
        if not self.is_fitted:
            logger.error("Model must be fitted before evaluation")
            raise ValueError("Model must be fitted before evaluation")
        
        # Validate inputs
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            logger.error("X must be a pandas DataFrame or numpy ndarray")
            raise TypeError("X must be a pandas DataFrame or numpy ndarray")
        
        if not isinstance(y, (pd.Series, np.ndarray)):
            logger.error("y must be a pandas Series or numpy ndarray")
            raise TypeError("y must be a pandas Series or numpy ndarray")
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Convert y to numpy array if needed
        y_true = y.values if isinstance(y, pd.Series) else y
        
        # Calculate metrics using our regression_metrics function
        logger.info("Evaluating LinearRegressionModel")
        try:
            metrics = regression_metrics(y_true, y_pred)
            logger.info("Model evaluation completed successfully")
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def get_coefficients(self) -> pd.DataFrame:
        """
        Get model coefficients with feature names.
        
        Returns:
            DataFrame with feature names and their corresponding coefficients,
            sorted by absolute coefficient value in descending order.
            
        Raises:
            ValueError: If model is not fitted.
        """
        # Check if model is fitted
        if not self.is_fitted:
            logger.error("Model must be fitted before getting coefficients")
            raise ValueError("Model must be fitted before getting coefficients")
        
        # Get coefficients
        try:
            coefficients = self.model.coef_
            
            # Create DataFrame
            coef_df = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': coefficients
            })
            
            # Add intercept if available
            if hasattr(self.model, 'intercept_') and self.model.fit_intercept:
                intercept_df = pd.DataFrame({
                    'feature': ['intercept'],
                    'coefficient': [self.model.intercept_]
                })
                coef_df = pd.concat([coef_df, intercept_df], ignore_index=True)
            
            # Sort by absolute coefficient value
            coef_df['abs_coef'] = coef_df['coefficient'].abs()
            coef_df = coef_df.sort_values('abs_coef', ascending=False)
            coef_df = coef_df.drop('abs_coef', axis=1).reset_index(drop=True)
            
            logger.info(f"Retrieved {len(coef_df)} coefficients")
            return coef_df
        except Exception as e:
            logger.error(f"Error getting coefficients: {str(e)}")
            raise
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model.
        
        Returns:
            Dictionary containing model information including:
            - Number of features
            - Feature names
            - Target name
            - Coefficients
            - Intercept (if applicable)
            - R² on training data (if available)
            
        Raises:
            ValueError: If model is not fitted.
        """
        # Check if model is fitted
        if not self.is_fitted:
            logger.error("Model must be fitted before getting summary")
            raise ValueError("Model must be fitted before getting summary")
        
        # Create summary dictionary
        try:
            summary_dict = {
                'model_type': 'Linear Regression',
                'num_features': len(self.feature_names),
                'feature_names': self.feature_names,
                'target_name': self.target_name,
                'coefficients': self.model.coef_.tolist(),
            }
            
            # Add intercept if available
            if hasattr(self.model, 'intercept_') and self.model.fit_intercept:
                summary_dict['intercept'] = float(self.model.intercept_)
            
            # Add R² score if available
            if hasattr(self.model, 'score'):
                summary_dict['r2_training'] = 'Not available for already fitted model'
            
            logger.info("Model summary generated successfully")
            return summary_dict
        except Exception as e:
            logger.error(f"Error generating model summary: {str(e)}")
            raise
