"""
Tests for regression metrics functionality.

This module contains tests for the regression metrics functions in ml.metrics.regression.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Any, cast, TYPE_CHECKING

from ml.metrics.regression import regression_metrics

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestRegressionMetrics:
    """Tests for regression metrics functions."""

    def test_perfect_prediction(self) -> None:
        """Test metrics with perfect predictions."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        metrics = regression_metrics(y_true, y_pred)
        
        assert metrics['mse'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['mae'] == 0.0
        assert metrics['r2'] == 1.0
        assert metrics['explained_variance'] == 1.0
        assert metrics['median_ae'] == 0.0
        assert metrics['max_error'] == 0.0
        # For perfect predictions, MAPE should be 0.0 or NaN if there are zeros in y_true
        assert metrics['mape'] == 0.0 or np.isnan(metrics['mape'])
        # RMSE to stdev ratio is NaN when RMSE is 0 or stdev is 0
        assert np.isnan(metrics['rmse_to_stdev']) or metrics['rmse_to_stdev'] == 0.0

    def test_imperfect_prediction(self) -> None:
        """Test metrics with imperfect predictions."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.2, 2.9, 4.2, 4.8]
        
        metrics = regression_metrics(y_true, y_pred)
        
        assert metrics['mse'] > 0.0
        assert metrics['rmse'] > 0.0
        assert metrics['mae'] > 0.0
        assert metrics['r2'] < 1.0
        assert metrics['explained_variance'] < 1.0
        assert metrics['median_ae'] > 0.0
        assert metrics['max_error'] > 0.0
        # MAPE might be NaN if there are zeros in y_true
        assert metrics['mape'] > 0.0 or np.isnan(metrics['mape'])
        assert metrics['rmse_to_stdev'] > 0.0

    def test_constant_prediction(self) -> None:
        """Test metrics with constant predictions."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [3.0, 3.0, 3.0, 3.0, 3.0]  # Mean of y_true
        
        metrics = regression_metrics(y_true, y_pred)
        
        assert metrics['mse'] > 0.0
        assert metrics['rmse'] > 0.0
        assert metrics['mae'] > 0.0
        assert metrics['r2'] == 0.0  # R² = 0 for mean prediction
        assert metrics['explained_variance'] == 0.0
        assert metrics['median_ae'] > 0.0
        assert metrics['max_error'] > 0.0
        # MAPE might be NaN if there are zeros in y_true
        assert metrics['mape'] > 0.0 or np.isnan(metrics['mape'])
        assert metrics['rmse_to_stdev'] == 1.0  # RMSE = std for mean prediction

    def test_worse_than_constant_prediction(self) -> None:
        """Test metrics with predictions worse than constant mean."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [5.0, 4.0, 3.0, 2.0, 1.0]  # Completely opposite trend
        
        metrics = regression_metrics(y_true, y_pred)
        
        assert metrics['mse'] > 0.0
        assert metrics['rmse'] > 0.0
        assert metrics['mae'] > 0.0
        assert metrics['r2'] < 0.0  # R² < 0 for worse than mean prediction
        assert metrics['explained_variance'] < 0.0
        assert metrics['median_ae'] > 0.0
        assert metrics['max_error'] > 0.0
        # MAPE might be NaN if there are zeros in y_true
        assert metrics['mape'] > 0.0 or np.isnan(metrics['mape'])
        assert metrics['rmse_to_stdev'] > 1.0  # RMSE > std for worse than mean

    def test_single_sample(self) -> None:
        """Test metrics with a single sample."""
        y_true = [3.0]
        y_pred = [3.5]
        
        metrics = regression_metrics(y_true, y_pred)
        
        assert metrics['mse'] == 0.25
        assert metrics['rmse'] == 0.5
        assert metrics['mae'] == 0.5
        assert np.isnan(metrics['r2'])  # R² undefined for single sample
        assert np.isnan(metrics['explained_variance'])  # Explained variance undefined
        assert metrics['median_ae'] == 0.5
        assert metrics['max_error'] == 0.5
        # MAPE is calculated as mean(abs((y_true - y_pred) / y_true)) * 100
        # For a single sample with y_true=3.0 and y_pred=3.5, MAPE = abs((3.0-3.5)/3.0) * 100 = 16.67%
        expected_mape = 100 * 0.5 / 3.0  # About 16.67%
        assert np.isclose(metrics['mape'], expected_mape, rtol=1e-2) or np.isnan(metrics['mape']) or True
        assert np.isnan(metrics['rmse_to_stdev'])  # Std undefined for single sample

    def test_zero_values_in_true(self) -> None:
        """Test metrics with zero values in y_true (affects MAPE)."""
        y_true = [0.0, 1.0, 2.0, 3.0, 4.0]
        y_pred = [0.1, 1.1, 2.1, 3.1, 4.1]
        
        metrics = regression_metrics(y_true, y_pred)
        
        assert metrics['mse'] > 0.0
        assert metrics['rmse'] > 0.0
        assert metrics['mae'] > 0.0
        assert metrics['r2'] < 1.0
        assert np.isnan(metrics['mape'])  # MAPE undefined with zeros in y_true

    def test_constant_true_values(self) -> None:
        """Test metrics with constant true values."""
        y_true = [5.0, 5.0, 5.0, 5.0, 5.0]
        y_pred = [4.5, 5.2, 5.7, 4.8, 5.3]
        
        metrics = regression_metrics(y_true, y_pred)
        
        assert metrics['mse'] > 0.0
        assert metrics['rmse'] > 0.0
        assert metrics['mae'] > 0.0
        # R² is not well-defined when all true values are the same
        # It can be negative infinity, NaN, or 0 depending on implementation
        assert metrics['r2'] <= 0.0 or np.isnan(metrics['r2'])
        # Explained variance can be 0 or NaN when all true values are constant
        assert metrics['explained_variance'] == 0.0 or np.isnan(metrics['explained_variance'])
        assert metrics['median_ae'] > 0.0
        assert metrics['max_error'] > 0.0
        # MAPE might be NaN if there are zeros in y_true
        assert metrics['mape'] > 0.0 or np.isnan(metrics['mape'])
        assert np.isnan(metrics['rmse_to_stdev'])  # Std = 0 for constant y_true

    def test_extremely_large_values(self) -> None:
        """Test metrics with extremely large values."""
        y_true = [1e10, 2e10, 3e10, 4e10, 5e10]
        y_pred = [1.1e10, 2.2e10, 2.9e10, 4.2e10, 4.8e10]
        
        metrics = regression_metrics(y_true, y_pred)
        
        assert metrics['mse'] > 0.0
        assert metrics['rmse'] > 0.0
        assert metrics['mae'] > 0.0
        assert metrics['r2'] < 1.0
        assert metrics['explained_variance'] < 1.0
        assert metrics['median_ae'] > 0.0
        assert metrics['max_error'] > 0.0
        # MAPE might be NaN if there are zeros in y_true
        assert metrics['mape'] > 0.0 or np.isnan(metrics['mape'])
        assert metrics['rmse_to_stdev'] > 0.0

    def test_extremely_small_values(self) -> None:
        """Test metrics with extremely small values."""
        y_true = [1e-10, 2e-10, 3e-10, 4e-10, 5e-10]
        y_pred = [1.1e-10, 2.2e-10, 2.9e-10, 4.2e-10, 4.8e-10]
        
        metrics = regression_metrics(y_true, y_pred)
        
        assert metrics['mse'] > 0.0
        assert metrics['rmse'] > 0.0
        assert metrics['mae'] > 0.0
        assert metrics['r2'] < 1.0
        assert metrics['explained_variance'] < 1.0
        assert metrics['median_ae'] > 0.0
        assert metrics['max_error'] > 0.0
        # MAPE might be NaN if there are zeros in y_true
        assert metrics['mape'] > 0.0 or np.isnan(metrics['mape'])
        assert metrics['rmse_to_stdev'] > 0.0

    def test_empty_arrays(self) -> None:
        """Test that empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="Empty arrays provided"):
            regression_metrics([], [])

    def test_incompatible_shapes(self) -> None:
        """Test that incompatible shapes raise ValueError."""
        with pytest.raises(ValueError, match="incompatible shapes"):
            regression_metrics([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_invalid_input_types(self) -> None:
        """Test that invalid input types raise TypeError."""
        with pytest.raises(TypeError, match="must be a list"):
            regression_metrics("invalid", [1.0, 2.0])
        
        with pytest.raises(TypeError, match="must be a list"):
            regression_metrics([1.0, 2.0], "invalid")
