"""
Unit tests for advanced regression metrics.

This module tests the advanced regression metrics functions from ml.metrics.advanced_regression.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from ml.base import advanced_regression_metrics, regression_metrics_by_group, confidence_interval_metrics

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestAdvancedRegressionMetrics:
    """Test cases for advanced regression metrics."""

    @pytest.fixture
    def sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sample regression data."""
        # Create synthetic regression data
        X, y = make_regression(
            n_samples=100,
            n_features=5,
            noise=0.1,
            random_state=42
        )
        
        # Train a model to get predictions
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        return y, y_pred

    @pytest.fixture
    def positive_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create positive-only regression data for testing metrics that require positive values."""
        # Create synthetic regression data with positive values
        X, y = make_regression(
            n_samples=100,
            n_features=5,
            noise=0.1,
            random_state=42
        )
        
        # Make y positive
        y = np.abs(y) + 1.0
        
        # Train a model to get predictions
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        return y, y_pred

    def test_advanced_regression_metrics_basic(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test basic functionality of advanced_regression_metrics."""
        y_true, y_pred = sample_data
        
        # Calculate metrics
        metrics = advanced_regression_metrics(y_true, y_pred)
        
        # Check that basic metrics are included
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'explained_variance' in metrics
        
        # Check that advanced metrics are included
        assert 'residual_std' in metrics
        assert 'prediction_std' in metrics
        assert 'd2_absolute' in metrics
        assert 'd2_pinball' in metrics
        
        # Check that values are reasonable
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['r2'] <= 1.0
        assert metrics['residual_std'] >= 0
        
        # Check that metrics without basic metrics don't include them
        metrics_without_basic = advanced_regression_metrics(y_true, y_pred, include_basic_metrics=False)
        assert 'mse' not in metrics_without_basic
        assert 'rmse' not in metrics_without_basic
        assert 'residual_std' in metrics_without_basic

    def test_advanced_regression_metrics_positive_data(self, positive_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test advanced_regression_metrics with positive data."""
        y_true, y_pred = positive_data
        
        # Calculate metrics
        metrics = advanced_regression_metrics(y_true, y_pred)
        
        # Check that metrics for positive data are included and valid
        assert 'msle' in metrics
        assert 'rmsle' in metrics
        assert 'gamma_deviance' in metrics
        assert 'tweedie_deviance' in metrics
        
        assert not np.isnan(metrics['msle'])
        assert not np.isnan(metrics['rmsle'])
        assert not np.isnan(metrics['gamma_deviance'])
        assert not np.isnan(metrics['tweedie_deviance'])
        
        assert metrics['msle'] >= 0
        assert metrics['rmsle'] >= 0

    def test_advanced_regression_metrics_with_negative_data(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test advanced_regression_metrics with data containing negative values."""
        y_true, y_pred = sample_data
        
        # Ensure some negative values
        y_true = y_true - np.max(y_true) - 1
        y_pred = y_pred - np.max(y_pred) - 1
        
        # Calculate metrics
        metrics = advanced_regression_metrics(y_true, y_pred)
        
        # Check that metrics requiring positive data are NaN
        assert 'msle' in metrics
        assert 'rmsle' in metrics
        assert 'gamma_deviance' in metrics
        
        assert np.isnan(metrics['msle'])
        assert np.isnan(metrics['rmsle'])
        assert np.isnan(metrics['gamma_deviance'])
        
        # But other metrics should still be valid
        assert not np.isnan(metrics['mse'])
        assert not np.isnan(metrics['rmse'])
        assert not np.isnan(metrics['r2'])

    def test_advanced_regression_metrics_input_validation(self) -> None:
        """Test input validation in advanced_regression_metrics."""
        # Test with invalid y_true type
        with pytest.raises(TypeError):
            advanced_regression_metrics("not_an_array", np.array([1, 2, 3]))
        
        # Test with invalid y_pred type
        with pytest.raises(TypeError):
            advanced_regression_metrics(np.array([1, 2, 3]), "not_an_array")
        
        # Test with incompatible shapes
        with pytest.raises(ValueError):
            advanced_regression_metrics(np.array([1, 2, 3]), np.array([1, 2]))
        
        # Test with empty arrays
        with pytest.raises(ValueError):
            advanced_regression_metrics(np.array([]), np.array([]))

    def test_regression_metrics_by_group(self) -> None:
        """Test regression_metrics_by_group function."""
        # Create synthetic data with groups
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        groups = np.array(['A'] * 50 + ['B'] * 50)
        
        # Train a model and get predictions
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate metrics by group
        metrics_df = regression_metrics_by_group(y, y_pred, groups)
        
        # Check that we have metrics for both groups
        assert len(metrics_df) == 2
        assert set(metrics_df['group']) == {'A', 'B'}
        
        # Check that all metrics are present for each group
        for _, row in metrics_df.iterrows():
            assert 'mse' in row
            assert 'rmse' in row
            assert 'r2' in row
            assert 'count' in row
        
        # Test with pandas Series inputs
        y_series = pd.Series(y)
        y_pred_series = pd.Series(y_pred)
        groups_series = pd.Series(groups)
        
        metrics_df_series = regression_metrics_by_group(y_series, y_pred_series, groups_series)
        assert len(metrics_df_series) == 2

    def test_regression_metrics_by_group_validation(self) -> None:
        """Test input validation in regression_metrics_by_group."""
        # Test with incompatible shapes
        with pytest.raises(ValueError):
            regression_metrics_by_group(
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                np.array(['A', 'B'])  # One group label missing
            )

    def test_confidence_interval_metrics(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test confidence_interval_metrics function."""
        y_true, y_pred = sample_data
        
        # Calculate metrics with confidence intervals
        ci_metrics = confidence_interval_metrics(
            y_true, y_pred,
            confidence_level=0.95,
            bootstrap_samples=100,  # Use fewer samples for faster testing
            random_state=42
        )
        
        # Check structure of results
        assert isinstance(ci_metrics, dict)
        assert 'mse' in ci_metrics
        assert 'rmse' in ci_metrics
        assert 'r2' in ci_metrics
        
        # Check that each metric has value, lower_bound, and upper_bound
        for metric_name, metric_data in ci_metrics.items():
            assert 'value' in metric_data
            assert 'lower_bound' in metric_data
            assert 'upper_bound' in metric_data
            
            # Check that bounds make sense
            assert metric_data['lower_bound'] <= metric_data['value'] <= metric_data['upper_bound']

    def test_confidence_interval_metrics_validation(self) -> None:
        """Test input validation in confidence_interval_metrics."""
        # Test with invalid confidence level
        with pytest.raises(ValueError):
            confidence_interval_metrics(
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                confidence_level=1.5  # Invalid confidence level
            )
        
        # Test with invalid input types
        with pytest.raises(TypeError):
            confidence_interval_metrics(
                "not_an_array",
                np.array([1, 2, 3])
            )
        
        # Test with incompatible shapes
        with pytest.raises(ValueError):
            confidence_interval_metrics(
                np.array([1, 2, 3]),
                np.array([1, 2])
            )

    def test_with_different_data_distributions(self) -> None:
        """Test metrics with different data distributions."""
        # Create datasets with different distributions
        np.random.seed(42)
        
        # Normal distribution
        y_true_normal = np.random.normal(0, 1, 100)
        y_pred_normal = y_true_normal + np.random.normal(0, 0.2, 100)
        
        # Uniform distribution
        y_true_uniform = np.random.uniform(-1, 1, 100)
        y_pred_uniform = y_true_uniform + np.random.normal(0, 0.2, 100)
        
        # Exponential distribution (positive values)
        y_true_exp = np.random.exponential(1, 100)
        y_pred_exp = y_true_exp * np.random.lognormal(0, 0.1, 100)
        
        # Calculate metrics for each distribution
        metrics_normal = advanced_regression_metrics(y_true_normal, y_pred_normal)
        metrics_uniform = advanced_regression_metrics(y_true_uniform, y_pred_uniform)
        metrics_exp = advanced_regression_metrics(y_true_exp, y_pred_exp)
        
        # Check that all metrics are calculated
        for metrics in [metrics_normal, metrics_uniform, metrics_exp]:
            assert 'mse' in metrics
            assert 'rmse' in metrics
            assert 'r2' in metrics
            assert 'residual_std' in metrics
        
        # For exponential data, MSLE should be valid
        assert not np.isnan(metrics_exp['msle'])
        assert not np.isnan(metrics_exp['rmsle'])
        
        # For normal and uniform data, MSLE might be NaN if there are negative values
        if np.any(y_true_normal <= 0) or np.any(y_pred_normal <= 0):
            assert np.isnan(metrics_normal['msle'])
        
        if np.any(y_true_uniform <= 0) or np.any(y_pred_uniform <= 0):
            assert np.isnan(metrics_uniform['msle'])

    def test_with_extreme_values(self) -> None:
        """Test metrics with extreme values."""
        # Very large values
        y_true_large = np.array([1e10, 2e10, 3e10])
        y_pred_large = np.array([1.1e10, 2.1e10, 3.1e10])
        
        # Very small values
        y_true_small = np.array([1e-10, 2e-10, 3e-10])
        y_pred_small = np.array([1.1e-10, 2.1e-10, 3.1e-10])
        
        # Calculate metrics
        metrics_large = advanced_regression_metrics(y_true_large, y_pred_large)
        metrics_small = advanced_regression_metrics(y_true_small, y_pred_small)
        
        # Check that metrics are calculated and not NaN
        for metrics in [metrics_large, metrics_small]:
            assert not np.isnan(metrics['mse'])
            assert not np.isnan(metrics['rmse'])
            assert not np.isnan(metrics['r2'])
            assert not np.isnan(metrics['residual_std'])

    def test_logging(self, caplog: "LogCaptureFixture") -> None:
        """Test that the function logs important events."""
        caplog.set_level(logging.INFO)
        
        # Create simple data
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        # Calculate metrics
        advanced_regression_metrics(y_true, y_pred)
        
        # Check that logs were created
        assert "Advanced regression metrics calculated" in caplog.text
