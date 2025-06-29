"""
Unit tests for regression metrics visualization.

This module tests the visualization functions from ml.metrics.visualization.
"""

import os
import tempfile
from typing import Dict, List, Optional, Tuple, Union, Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from ml.base import (
    plot_residuals,
    plot_actual_vs_predicted,
    plot_prediction_error_distribution,
    plot_metrics_comparison,
    plot_prediction_intervals
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestRegressionMetricsVisualization:
    """Test cases for regression metrics visualization functions."""

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
    def temp_dir(self) -> str:
        """Create a temporary directory for saving plots."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield tmpdirname

    def test_plot_residuals_basic(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test basic functionality of plot_residuals."""
        y_true, y_pred = sample_data
        
        # Create plot
        fig = plot_residuals(y_true, y_pred)
        
        # Check that a figure was returned
        assert isinstance(fig, Figure)
        
        # Check figure properties
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Predicted Values"
        assert ax.get_ylabel() == "Residuals"
        assert ax.get_title() == "Residual Plot"

    def test_plot_residuals_custom_params(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test plot_residuals with custom parameters."""
        y_true, y_pred = sample_data
        
        # Create plot with custom parameters
        title = "Custom Residual Plot"
        figsize = (8, 5)
        
        fig = plot_residuals(
            y_true, y_pred,
            title=title,
            figsize=figsize
        )
        
        # Check that custom parameters were applied
        assert fig.get_size_inches()[0] == figsize[0]
        assert fig.get_size_inches()[1] == figsize[1]
        assert fig.axes[0].get_title() == title

    def test_plot_residuals_save(self, sample_data: Tuple[np.ndarray, np.ndarray], temp_dir: str) -> None:
        """Test saving plot_residuals to file."""
        y_true, y_pred = sample_data
        
        # Create save path
        save_path = os.path.join(temp_dir, "residuals.png")
        
        # Create and save plot
        plot_residuals(y_true, y_pred, save_path=save_path)
        
        # Check that file was created
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_plot_residuals_input_validation(self) -> None:
        """Test input validation in plot_residuals."""
        # Test with invalid y_true type
        with pytest.raises(TypeError):
            plot_residuals("not_an_array", np.array([1, 2, 3]))
        
        # Test with invalid y_pred type
        with pytest.raises(TypeError):
            plot_residuals(np.array([1, 2, 3]), "not_an_array")
        
        # Test with incompatible shapes
        with pytest.raises(ValueError):
            plot_residuals(np.array([1, 2, 3]), np.array([1, 2]))

    def test_plot_actual_vs_predicted_basic(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test basic functionality of plot_actual_vs_predicted."""
        y_true, y_pred = sample_data
        
        # Create plot
        fig = plot_actual_vs_predicted(y_true, y_pred)
        
        # Check that a figure was returned
        assert isinstance(fig, Figure)
        
        # Check figure properties
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Actual Values"
        assert ax.get_ylabel() == "Predicted Values"
        assert ax.get_title() == "Actual vs Predicted Values"

    def test_plot_actual_vs_predicted_no_identity_line(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test plot_actual_vs_predicted without identity line."""
        y_true, y_pred = sample_data

        # Create plot without identity line
        fig = plot_actual_vs_predicted(y_true, y_pred, identity_line=False)

        # Check that a figure was returned
        assert isinstance(fig, Figure)

        # Check that the plot was created
        ax = fig.axes[0]
        
        # Verify plot has a title and labels
        assert ax.get_title() != ""
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""

    def test_plot_prediction_error_distribution_basic(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test basic functionality of plot_prediction_error_distribution."""
        y_true, y_pred = sample_data
        
        # Create plot
        fig = plot_prediction_error_distribution(y_true, y_pred)
        
        # Check that a figure was returned
        assert isinstance(fig, Figure)
        
        # Check figure properties
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Prediction Error"
        assert ax.get_ylabel() == "Frequency"
        assert ax.get_title() == "Prediction Error Distribution"

    def test_plot_prediction_error_distribution_no_kde(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test plot_prediction_error_distribution without KDE."""
        y_true, y_pred = sample_data
        
        # Create plot without KDE
        fig = plot_prediction_error_distribution(y_true, y_pred, kde=False)
        
        # Check that a figure was returned
        assert isinstance(fig, Figure)

    def test_plot_metrics_comparison_basic(self) -> None:
        """Test basic functionality of plot_metrics_comparison."""
        # Create sample metrics
        metrics1 = {'mse': 0.1, 'rmse': 0.316, 'mae': 0.25, 'r2': 0.85}
        metrics2 = {'mse': 0.2, 'rmse': 0.447, 'mae': 0.35, 'r2': 0.75}
        
        # Create plot
        fig = plot_metrics_comparison(
            [metrics1, metrics2],
            ['Model A', 'Model B']
        )
        
        # Check that a figure was returned
        assert isinstance(fig, Figure)
        
        # Check figure properties
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Metric"
        assert ax.get_ylabel() == "Value"
        assert ax.get_title() == "Model Performance Comparison"

    def test_plot_metrics_comparison_custom_metrics(self) -> None:
        """Test plot_metrics_comparison with custom metrics selection."""
        # Create sample metrics
        metrics1 = {'mse': 0.1, 'rmse': 0.316, 'mae': 0.25, 'r2': 0.85}
        metrics2 = {'mse': 0.2, 'rmse': 0.447, 'mae': 0.35, 'r2': 0.75}
        
        # Create plot with specific metrics
        fig = plot_metrics_comparison(
            [metrics1, metrics2],
            ['Model A', 'Model B'],
            metrics_to_plot=['mse', 'r2']
        )
        
        # Check that a figure was returned
        assert isinstance(fig, Figure)

    def test_plot_metrics_comparison_validation(self) -> None:
        """Test input validation in plot_metrics_comparison."""
        # Test with incompatible lengths
        with pytest.raises(ValueError):
            plot_metrics_comparison(
                [{'mse': 0.1}],  # One metrics dict
                ['Model A', 'Model B']  # Two model names
            )
        
        # Test with empty metrics list
        with pytest.raises(ValueError):
            plot_metrics_comparison([], [])
        
        # Test with no common metrics
        with pytest.raises(ValueError):
            plot_metrics_comparison(
                [{'metric1': 0.1}, {'metric2': 0.2}],
                ['Model A', 'Model B'],
                metrics_to_plot=['common_metric']  # No common metric
            )

    def test_plot_prediction_intervals_basic(self) -> None:
        """Test basic functionality of plot_prediction_intervals."""
        # Create sample data
        np.random.seed(42)
        y_true = np.random.normal(0, 1, 50)
        y_pred = y_true + np.random.normal(0, 0.2, 50)
        y_std = np.abs(np.random.normal(0.2, 0.05, 50))
        
        # Create plot
        fig = plot_prediction_intervals(y_true, y_pred, y_std)
        
        # Check that a figure was returned
        assert isinstance(fig, Figure)
        
        # Check figure properties
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Sample Index"
        assert ax.get_ylabel() == "Value"
        assert ax.get_title() == "Predictions with Confidence Intervals"

    def test_plot_prediction_intervals_custom_confidence(self) -> None:
        """Test plot_prediction_intervals with custom confidence level."""
        # Create sample data
        np.random.seed(42)
        y_true = np.random.normal(0, 1, 50)
        y_pred = y_true + np.random.normal(0, 0.2, 50)
        y_std = np.abs(np.random.normal(0.2, 0.05, 50))
        
        # Create plot with custom confidence level
        confidence_level = 0.9
        fig = plot_prediction_intervals(
            y_true, y_pred, y_std,
            confidence_level=confidence_level
        )
        
        # Check that a figure was returned
        assert isinstance(fig, Figure)
        
        # Check legend contains the correct confidence level
        ax = fig.axes[0]
        legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
        assert any(f"{int(confidence_level*100)}% Confidence Interval" in text for text in legend_texts)

    def test_plot_prediction_intervals_validation(self) -> None:
        """Test input validation in plot_prediction_intervals."""
        # Create sample data
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 2.1, 3.1])
        y_std = np.array([0.1, 0.1, 0.1])
        
        # Test with invalid confidence level
        with pytest.raises(ValueError):
            plot_prediction_intervals(
                y_true, y_pred, y_std,
                confidence_level=1.5  # Invalid confidence level
            )
        
        # Test with incompatible shapes
        with pytest.raises(ValueError):
            plot_prediction_intervals(
                y_true, y_pred, np.array([0.1, 0.1])  # One fewer std value
            )
        
        # Test with invalid input types
        with pytest.raises(TypeError):
            plot_prediction_intervals(
                "not_an_array", y_pred, y_std
            )

    def test_with_pandas_series(self) -> None:
        """Test visualization functions with pandas Series inputs."""
        # Create pandas Series data
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = pd.Series([1.1, 2.1, 3.1, 4.1, 5.1])
        
        # Test each visualization function
        fig1 = plot_residuals(y_true, y_pred)
        assert isinstance(fig1, Figure)
        
        fig2 = plot_actual_vs_predicted(y_true, y_pred)
        assert isinstance(fig2, Figure)
        
        fig3 = plot_prediction_error_distribution(y_true, y_pred)
        assert isinstance(fig3, Figure)
        
        # For prediction intervals, we need std values
        y_std = pd.Series([0.1, 0.1, 0.1, 0.1, 0.1])
        fig4 = plot_prediction_intervals(y_true, y_pred, y_std)
        assert isinstance(fig4, Figure)
