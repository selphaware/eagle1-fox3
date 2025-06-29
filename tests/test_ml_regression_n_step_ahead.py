"""
Unit tests for N-step ahead prediction in regression models.

This module tests the N-step ahead prediction functionality in all regression models.
"""

import logging
import numpy as np
import pandas as pd
import pytest
from typing import Tuple, List, Dict, Any, Optional, Union, Callable, cast
from _pytest.logging import LogCaptureFixture
from pytest_mock import MockerFixture

from ml.base import (
    LinearRegressionModel,
    RandomForestRegressionModel,
    TensorFlowDNNRegressor
)


@pytest.fixture
def sample_time_series_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create sample time series data for testing N-step ahead prediction.
    
    Returns:
        Tuple containing features DataFrame and target Series.
    """
    # Create a simple AR(2) process: y_t = 0.8*y_{t-1} - 0.2*y_{t-2} + noise
    np.random.seed(42)
    n_samples = 100
    noise = np.random.normal(0, 0.1, n_samples)
    
    y = np.zeros(n_samples)
    # Initialize first two values
    y[0] = 0.5
    y[1] = 0.7
    
    # Generate the rest of the series
    for t in range(2, n_samples):
        y[t] = 0.8 * y[t-1] - 0.2 * y[t-2] + noise[t]
    
    # Create features (lagged values)
    X = np.zeros((n_samples - 2, 2))
    for t in range(2, n_samples):
        X[t-2, 0] = y[t-2]  # y_{t-2}
        X[t-2, 1] = y[t-1]  # y_{t-1}
    
    # Target is the current value
    y_target = y[2:]
    
    # Convert to pandas
    X_df = pd.DataFrame(X, columns=['y_lag2', 'y_lag1'])
    y_series = pd.Series(y_target, name='y')
    
    return X_df, y_series


class TestRegressionNStepAhead:
    """Test cases for N-step ahead prediction in regression models."""
    
    def test_linear_regression_n_step_ahead(self, sample_time_series_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test N-step ahead prediction with LinearRegressionModel."""
        X, y = sample_time_series_data
        
        # Split data for training and testing
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        model = LinearRegressionModel()
        model.fit(X_train, y_train)
        
        # Test single-step prediction
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)
        
        # Test N-step ahead prediction
        n_steps = 5
        initial_features = X_test.iloc[[0]].copy()  # First test sample
        
        # Make predictions
        predictions = model.predict_n_steps_ahead(initial_features, n_steps)
        
        # Check predictions
        assert len(predictions) == n_steps
        assert isinstance(predictions, np.ndarray)
        
        # Check that predictions are reasonable (not NaN or inf)
        assert np.all(np.isfinite(predictions))
    
    def test_random_forest_n_step_ahead(self, sample_time_series_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test N-step ahead prediction with RandomForestRegressionModel."""
        X, y = sample_time_series_data
        
        # Split data for training and testing
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model with fewer trees for faster testing
        model = RandomForestRegressionModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Test single-step prediction
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)
        
        # Test N-step ahead prediction
        n_steps = 5
        initial_features = X_test.iloc[[0]].copy()  # First test sample
        
        # Make predictions
        predictions = model.predict_n_steps_ahead(initial_features, n_steps)
        
        # Check predictions
        assert len(predictions) == n_steps
        assert isinstance(predictions, np.ndarray)
        
        # Check that predictions are reasonable (not NaN or inf)
        assert np.all(np.isfinite(predictions))
    
    def test_tensorflow_dnn_n_step_ahead(self, sample_time_series_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test N-step ahead prediction with TensorFlowDNNRegressor."""
        X, y = sample_time_series_data
        
        # Split data for training and testing
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model with minimal configuration for faster testing
        model = TensorFlowDNNRegressor(
            hidden_layers=[8],
            epochs=5,
            batch_size=16,
            early_stopping_patience=2,
            random_state=42
        )
        model.fit(X_train, y_train, verbose=0)
        
        # Test single-step prediction
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)
        
        # Test N-step ahead prediction
        n_steps = 5
        initial_features = X_test.iloc[[0]].copy()  # First test sample
        
        # Make predictions
        predictions = model.predict_n_steps_ahead(initial_features, n_steps)
        
        # Check predictions
        assert len(predictions) == n_steps
        assert isinstance(predictions, np.ndarray)
        
        # Check that predictions are reasonable (not NaN or inf)
        assert np.all(np.isfinite(predictions))
    
    def test_custom_feature_update_function(self, sample_time_series_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test N-step ahead prediction with custom feature update function."""
        X, y = sample_time_series_data
        
        # Split data for training and testing
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        model = LinearRegressionModel()
        model.fit(X_train, y_train)
        
        # Define custom feature update function
        def custom_update(
            features: Union[pd.DataFrame, np.ndarray],
            prediction: np.ndarray,
            step_idx: int
        ) -> Union[pd.DataFrame, np.ndarray]:
            """Custom function to update features for next prediction step."""
            if isinstance(features, pd.DataFrame):
                new_features = features.copy()
                # Shift features (y_lag2 = y_lag1, y_lag1 = new_prediction)
                new_features.iloc[0, 0] = new_features.iloc[0, 1]  # y_lag2 = y_lag1
                new_features.iloc[0, 1] = prediction[0]  # y_lag1 = new prediction
                return new_features
            else:
                new_features = np.copy(features)
                new_features[0, 0] = new_features[0, 1]  # y_lag2 = y_lag1
                new_features[0, 1] = prediction[0]  # y_lag1 = new prediction
                return new_features
        
        # Test N-step ahead prediction with custom update function
        n_steps = 5
        initial_features = X_test.iloc[[0]].copy()  # First test sample
        
        # Make predictions
        predictions = model.predict_n_steps_ahead(
            initial_features,
            n_steps,
            feature_update_func=custom_update
        )
        
        # Check predictions
        assert len(predictions) == n_steps
        assert isinstance(predictions, np.ndarray)
        
        # Check that predictions are reasonable (not NaN or inf)
        assert np.all(np.isfinite(predictions))
    
    def test_n_step_ahead_input_validation(self) -> None:
        """Test input validation in predict_n_steps_ahead method."""
        # Create model
        model = LinearRegressionModel()
        
        # Test with unfitted model
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict_n_steps_ahead(pd.DataFrame([[1, 2]]), 3)
        
        # Create and fit a model with minimal data
        X = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=['a', 'b'])
        y = pd.Series([10, 20, 30])
        model.fit(X, y)
        
        # Test with invalid n_steps
        with pytest.raises(ValueError, match="n_steps must be at least 1"):
            model.predict_n_steps_ahead(X.iloc[[0]], 0)
        
        # Test with invalid input type
        with pytest.raises(TypeError, match="must be a pandas DataFrame or numpy ndarray"):
            model.predict_n_steps_ahead([1, 2], 3)  # type: ignore
    
    def test_n_step_ahead_logging(self, caplog: LogCaptureFixture) -> None:
        """Test logging in predict_n_steps_ahead method."""
        # Create and fit a model with minimal data
        X = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=['a', 'b'])
        y = pd.Series([10, 20, 30])
        
        model = LinearRegressionModel()
        model.fit(X, y)
        
        # Set log level to INFO to capture logs
        caplog.set_level(logging.INFO)
        
        # Make predictions
        model.predict_n_steps_ahead(X.iloc[[0]], 3)
        
        # Check logs
        assert "Making 3-step ahead predictions" in caplog.text
        assert "3-step ahead predictions generated successfully" in caplog.text
