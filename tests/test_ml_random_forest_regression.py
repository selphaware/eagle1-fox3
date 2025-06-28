"""
Unit tests for Random Forest Regression model.

This module tests the RandomForestRegressionModel class from ml.regression.random_forest.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError

from ml.regression import RandomForestRegressionModel

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestRandomForestRegressionModel:
    """Test cases for RandomForestRegressionModel."""

    def test_initialization(self) -> None:
        """Test model initialization with default and custom parameters."""
        # Test with default parameters
        model = RandomForestRegressionModel()
        assert isinstance(model.model, RandomForestRegressor)
        assert model.is_fitted is False
        assert model.feature_names == []

        # Test with custom parameters
        model = RandomForestRegressionModel(
            n_estimators=50,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        assert model.model.n_estimators == 50
        assert model.model.max_depth == 10
        assert model.model.min_samples_split == 5
        assert model.model.random_state == 42
        assert model.is_fitted is False

    def test_fit_with_dataframe(self) -> None:
        """Test model fitting with pandas DataFrame and Series."""
        # Create sample data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })
        y = pd.Series([10, 12, 14, 16, 18], name='target')

        # Fit model
        model = RandomForestRegressionModel(random_state=42)
        model.fit(X, y)

        # Check model state
        assert model.is_fitted is True
        assert model.feature_names == ['feature1', 'feature2']
        assert model.target_name == 'target'

    def test_fit_with_numpy(self) -> None:
        """Test model fitting with numpy arrays."""
        # Create sample data
        X = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]])
        y = np.array([10, 12, 14, 16, 18])

        # Fit model with feature names
        model = RandomForestRegressionModel(random_state=42)
        feature_names = ['feature1', 'feature2']
        model.fit(X, y, feature_names=feature_names)

        # Check model state
        assert model.is_fitted is True
        assert model.feature_names == feature_names
        assert model.target_name == 'target'  # Default target name

        # Fit model without feature names
        model = RandomForestRegressionModel(random_state=42)
        model.fit(X, y)

        # Check auto-generated feature names
        assert model.is_fitted is True
        assert model.feature_names == ['feature_0', 'feature_1']

    def test_fit_with_invalid_inputs(self) -> None:
        """Test model fitting with invalid inputs."""
        model = RandomForestRegressionModel()

        # Test with invalid X type
        with pytest.raises(TypeError):
            model.fit("not_a_dataframe", np.array([1, 2, 3]))

        # Test with invalid y type
        with pytest.raises(TypeError):
            model.fit(np.array([[1, 2], [3, 4]]), "not_an_array")

        # Test with mismatched feature_names length
        with pytest.raises(ValueError):
            model.fit(
                np.array([[1, 2], [3, 4]]),
                np.array([1, 2]),
                feature_names=['feature1']  # Should be 2 features
            )

    def test_predict(self) -> None:
        """Test model prediction."""
        # Create and fit model
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })
        y = pd.Series([10, 12, 14, 16, 18])

        model = RandomForestRegressionModel(random_state=42)
        model.fit(X, y)

        # Test prediction with DataFrame
        predictions = model.predict(X)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 5

        # Test prediction with numpy array
        predictions = model.predict(X.values)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 5

    def test_predict_with_unfitted_model(self) -> None:
        """Test prediction with unfitted model."""
        model = RandomForestRegressionModel()
        X = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError):
            model.predict(X)

    def test_predict_with_invalid_inputs(self) -> None:
        """Test prediction with invalid inputs."""
        # Create and fit model
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])

        model = RandomForestRegressionModel(random_state=42)
        model.fit(X, y)

        # Test with invalid X type
        with pytest.raises(TypeError):
            model.predict("not_a_dataframe")

        # Test with wrong number of features
        with pytest.raises(ValueError):
            model.predict(np.array([[1, 2, 3], [4, 5, 6]]))  # 3 features instead of 2

    def test_evaluate(self) -> None:
        """Test model evaluation."""
        # Create and fit model
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })
        y = pd.Series([10, 12, 14, 16, 18])

        model = RandomForestRegressionModel(random_state=42)
        model.fit(X, y)

        # Evaluate model
        metrics = model.evaluate(X, y)

        # Check metrics
        assert isinstance(metrics, dict)
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'explained_variance' in metrics
        assert 'rmse_to_stdev' in metrics

        # R² should be good for training data
        assert metrics['r2'] > 0.5

    def test_get_feature_importance(self) -> None:
        """Test feature importance extraction."""
        # Create and fit model
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [5, 4, 3, 2, 1, 5, 4, 3, 2, 1],
            'feature3': [2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        })
        # Create a target with strong dependence on feature1
        y = pd.Series([10, 12, 14, 16, 18, 20, 22, 24, 26, 28])

        model = RandomForestRegressionModel(random_state=42)
        model.fit(X, y)

        # Get feature importance
        importance_df = model.get_feature_importance()

        # Check structure
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == 3  # 3 features

        # Check if sorted by importance
        assert importance_df['importance'].is_monotonic_decreasing

        # Feature1 should be most important
        assert importance_df.iloc[0]['feature'] == 'feature1'

    def test_get_feature_importance_unfitted(self) -> None:
        """Test feature importance with unfitted model."""
        model = RandomForestRegressionModel()

        with pytest.raises(ValueError):
            model.get_feature_importance()

    def test_summary(self) -> None:
        """Test model summary."""
        # Create and fit model
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })
        y = pd.Series([10, 12, 14, 16, 18])

        model = RandomForestRegressionModel(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        model.fit(X, y)

        # Get summary
        summary = model.summary()

        # Check summary content
        assert isinstance(summary, dict)
        assert summary['model_type'] == 'RandomForestRegression'
        assert summary['n_estimators'] == 50
        assert summary['max_depth'] == 10
        assert summary['random_state'] == 42
        assert summary['n_features'] == 2
        assert summary['feature_names'] == ['feature1', 'feature2']
        assert summary['is_fitted'] is True

    def test_summary_unfitted(self) -> None:
        """Test summary with unfitted model."""
        model = RandomForestRegressionModel()

        with pytest.raises(ValueError):
            model.summary()

    def test_with_nonlinear_data(self) -> None:
        """Test model with nonlinear data where Random Forest should excel."""
        # Create nonlinear data
        np.random.seed(42)
        X = np.random.rand(100, 2)  # 100 samples, 2 features
        # Nonlinear function: y = x1^2 + sin(x2) + noise
        y = X[:, 0]**2 + np.sin(4 * X[:, 1]) + np.random.normal(0, 0.1, 100)

        X_df = pd.DataFrame(X, columns=['feature1', 'feature2'])
        y_series = pd.Series(y, name='target')

        # Fit Random Forest model
        rf_model = RandomForestRegressionModel(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        rf_model.fit(X_df, y_series)

        # Evaluate model
        metrics = rf_model.evaluate(X_df, y_series)

        # Random Forest should handle nonlinear data well
        assert metrics['r2'] > 0.8
        assert metrics['mse'] < 0.05

    def test_with_missing_values(self) -> None:
        """Test model behavior with missing values."""
        # Create data with missing values
        X = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [5, np.nan, 3, 2, 1]
        })
        y = pd.Series([10, 12, 14, 16, 18])

        # Random Forest cannot handle missing values directly
        model = RandomForestRegressionModel(random_state=42)
        
        # Fitting should raise an exception
        with pytest.raises(Exception):
            model.fit(X, y)

    def test_logging(self, caplog: "LogCaptureFixture") -> None:
        """Test that the model logs important events."""
        caplog.set_level(logging.INFO)

        # Create and fit model
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })
        y = pd.Series([10, 12, 14, 16, 18])

        model = RandomForestRegressionModel(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        model.fit(X, y)
        model.predict(X)
        model.evaluate(X, y)
        model.get_feature_importance()
        model.summary()

        # Check that logs were created
        assert "Initialized RandomForestRegressionModel" in caplog.text
        assert "Fitted RandomForestRegressionModel" in caplog.text
        assert "Made predictions" in caplog.text
        assert "Evaluated model" in caplog.text
        assert "Extracted feature importance" in caplog.text
        assert "Generated model summary" in caplog.text
