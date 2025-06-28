"""
Tests for the LinearRegressionModel class.

This module contains tests for the LinearRegressionModel class in ml.regression.linear_regression.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Any, cast, TYPE_CHECKING
from unittest.mock import Mock, patch

from ml.regression.linear_regression import LinearRegressionModel

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestLinearRegressionModel:
    """Tests for the LinearRegressionModel class."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Create sample data for testing."""
        # Create a simple dataset with a linear relationship
        np.random.seed(42)
        X = np.random.rand(100, 3)
        # y = 2*x1 + 3*x2 - 1.5*x3 + 0.5 + noise
        y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + 0.5 + np.random.normal(0, 0.1, 100)
        
        # Convert to pandas DataFrame/Series
        X_df = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])
        y_series = pd.Series(y, name="target")
        
        return {
            "X_train": X_df.iloc[:80],
            "y_train": y_series.iloc[:80],
            "X_test": X_df.iloc[80:],
            "y_test": y_series.iloc[80:],
            "feature_names": ["feature1", "feature2", "feature3"],
            "true_coefficients": [2, 3, -1.5],
            "true_intercept": 0.5
        }
    
    def test_initialization(self) -> None:
        """Test that the model initializes correctly."""
        model = LinearRegressionModel()
        
        assert hasattr(model, "model")
        assert hasattr(model, "feature_names")
        assert hasattr(model, "target_name")
        assert hasattr(model, "is_fitted")
        assert model.feature_names == []
        assert model.target_name == ""
        assert model.is_fitted is False
    
    def test_fit_with_dataframe(self, sample_data: Dict[str, Any]) -> None:
        """Test fitting the model with pandas DataFrame/Series."""
        model = LinearRegressionModel()
        result = model.fit(
            sample_data["X_train"],
            sample_data["y_train"]
        )
        
        # Check that fit returns self for method chaining
        assert result is model
        
        # Check that model is fitted
        assert model.is_fitted is True
        
        # Check that feature names are set correctly
        assert model.feature_names == sample_data["feature_names"]
        
        # Check that target name is set correctly
        assert model.target_name == "target"
        
        # Check that coefficients are close to true values
        coefficients = model.model.coef_
        for i, (coef, true_coef) in enumerate(zip(coefficients, sample_data["true_coefficients"])):
            assert np.isclose(coef, true_coef, rtol=0.2), f"Coefficient {i} is not close to true value"
        
        # Check that intercept is close to true value
        assert np.isclose(model.model.intercept_, sample_data["true_intercept"], rtol=0.2)
    
    def test_fit_with_numpy_arrays(self, sample_data: Dict[str, Any]) -> None:
        """Test fitting the model with numpy arrays."""
        model = LinearRegressionModel()
        X_train_np = sample_data["X_train"].values
        y_train_np = sample_data["y_train"].values
        
        model.fit(
            X_train_np,
            y_train_np,
            feature_names=sample_data["feature_names"]
        )
        
        # Check that model is fitted
        assert model.is_fitted is True
        
        # Check that feature names are set correctly
        assert model.feature_names == sample_data["feature_names"]
        
        # Check that coefficients are close to true values
        coefficients = model.model.coef_
        for i, (coef, true_coef) in enumerate(zip(coefficients, sample_data["true_coefficients"])):
            assert np.isclose(coef, true_coef, rtol=0.2), f"Coefficient {i} is not close to true value"
    
    def test_fit_with_invalid_inputs(self) -> None:
        """Test fitting the model with invalid inputs."""
        model = LinearRegressionModel()
        
        # Test with non-DataFrame X
        with pytest.raises(TypeError):
            model.fit("not a dataframe", pd.Series([1, 2, 3]))
        
        # Test with non-Series y
        with pytest.raises(TypeError):
            model.fit(pd.DataFrame({"a": [1, 2, 3]}), "not a series")
        
        # Test with mismatched feature names
        with pytest.raises(ValueError):
            model.fit(
                pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
                pd.Series([1, 2, 3]),
                feature_names=["a", "b", "c"]  # Too many feature names
            )
    
    def test_predict(self, sample_data: Dict[str, Any]) -> None:
        """Test making predictions with the model."""
        model = LinearRegressionModel()
        model.fit(
            sample_data["X_train"],
            sample_data["y_train"]
        )
        
        # Test with DataFrame
        predictions_df = model.predict(sample_data["X_test"])
        assert isinstance(predictions_df, np.ndarray)
        assert len(predictions_df) == len(sample_data["X_test"])
        
        # Test with numpy array
        predictions_np = model.predict(sample_data["X_test"].values)
        assert isinstance(predictions_np, np.ndarray)
        assert len(predictions_np) == len(sample_data["X_test"])
        
        # Check that predictions are reasonable (close to true values)
        X_test = sample_data["X_test"].values
        expected_y = (
            sample_data["true_coefficients"][0] * X_test[:, 0] +
            sample_data["true_coefficients"][1] * X_test[:, 1] +
            sample_data["true_coefficients"][2] * X_test[:, 2] +
            sample_data["true_intercept"]
        )
        assert np.allclose(predictions_df, expected_y, rtol=0.3)
    
    def test_predict_without_fitting(self) -> None:
        """Test that predict raises an error if model is not fitted."""
        model = LinearRegressionModel()
        
        with pytest.raises(ValueError):
            model.predict(pd.DataFrame({"a": [1, 2, 3]}))
    
    def test_predict_with_invalid_inputs(self, sample_data: Dict[str, Any]) -> None:
        """Test predict with invalid inputs."""
        model = LinearRegressionModel()
        model.fit(
            sample_data["X_train"],
            sample_data["y_train"]
        )
        
        # Test with non-DataFrame/ndarray
        with pytest.raises(TypeError):
            model.predict("not a dataframe")
        
        # Test with wrong number of features
        with pytest.raises(ValueError):
            model.predict(pd.DataFrame({"a": [1, 2, 3]}))  # Only one feature
    
    def test_evaluate(self, sample_data: Dict[str, Any]) -> None:
        """Test evaluating the model."""
        model = LinearRegressionModel()
        model.fit(
            sample_data["X_train"],
            sample_data["y_train"]
        )
        
        # Evaluate with DataFrame/Series
        metrics = model.evaluate(
            sample_data["X_test"],
            sample_data["y_test"]
        )
        
        # Check that metrics are returned
        assert isinstance(metrics, dict)
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "explained_variance" in metrics
        
        # Check that metrics are reasonable
        assert metrics["mse"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["r2"] <= 1.0
        assert metrics["explained_variance"] <= 1.0
        
        # For a good model on this data, R² should be high
        assert metrics["r2"] > 0.8
        
        # Evaluate with numpy arrays
        metrics_np = model.evaluate(
            sample_data["X_test"].values,
            sample_data["y_test"].values
        )
        
        # Check that metrics are the same
        assert np.isclose(metrics["mse"], metrics_np["mse"])
        assert np.isclose(metrics["r2"], metrics_np["r2"])
    
    def test_evaluate_without_fitting(self) -> None:
        """Test that evaluate raises an error if model is not fitted."""
        model = LinearRegressionModel()
        
        with pytest.raises(ValueError):
            model.evaluate(
                pd.DataFrame({"a": [1, 2, 3]}),
                pd.Series([1, 2, 3])
            )
    
    def test_evaluate_with_invalid_inputs(self, sample_data: Dict[str, Any]) -> None:
        """Test evaluate with invalid inputs."""
        model = LinearRegressionModel()
        model.fit(
            sample_data["X_train"],
            sample_data["y_train"]
        )
        
        # Test with non-DataFrame X
        with pytest.raises(TypeError):
            model.evaluate(
                "not a dataframe",
                sample_data["y_test"]
            )
        
        # Test with non-Series y
        with pytest.raises(TypeError):
            model.evaluate(
                sample_data["X_test"],
                "not a series"
            )
    
    def test_get_coefficients(self, sample_data: Dict[str, Any]) -> None:
        """Test getting model coefficients."""
        model = LinearRegressionModel()
        model.fit(
            sample_data["X_train"],
            sample_data["y_train"]
        )
        
        coef_df = model.get_coefficients()
        
        # Check that DataFrame has the right structure
        assert isinstance(coef_df, pd.DataFrame)
        assert "feature" in coef_df.columns
        assert "coefficient" in coef_df.columns
        
        # Check that all features are included
        features_in_df = set(coef_df["feature"].tolist())
        expected_features = set(sample_data["feature_names"] + ["intercept"])
        assert features_in_df == expected_features
        
        # Check that coefficients are sorted by absolute value
        abs_coefs = coef_df["coefficient"].abs().tolist()
        assert all(abs_coefs[i] >= abs_coefs[i+1] for i in range(len(abs_coefs)-1))
        
        # Check that coefficients match the model
        for feature, coef in zip(coef_df["feature"], coef_df["coefficient"]):
            if feature == "intercept":
                assert np.isclose(coef, model.model.intercept_)
            else:
                idx = sample_data["feature_names"].index(feature)
                assert np.isclose(coef, model.model.coef_[idx])
    
    def test_get_coefficients_without_fitting(self) -> None:
        """Test that get_coefficients raises an error if model is not fitted."""
        model = LinearRegressionModel()
        
        with pytest.raises(ValueError):
            model.get_coefficients()
    
    def test_summary(self, sample_data: Dict[str, Any]) -> None:
        """Test getting model summary."""
        model = LinearRegressionModel()
        model.fit(
            sample_data["X_train"],
            sample_data["y_train"]
        )
        
        summary = model.summary()
        
        # Check that summary has the right structure
        assert isinstance(summary, dict)
        assert "model_type" in summary
        assert "num_features" in summary
        assert "feature_names" in summary
        assert "target_name" in summary
        assert "coefficients" in summary
        assert "intercept" in summary
        
        # Check that values are correct
        assert summary["model_type"] == "Linear Regression"
        assert summary["num_features"] == len(sample_data["feature_names"])
        assert summary["feature_names"] == sample_data["feature_names"]
        assert summary["target_name"] == "target"
        assert len(summary["coefficients"]) == len(sample_data["feature_names"])
        assert isinstance(summary["intercept"], float)
    
    def test_summary_without_fitting(self) -> None:
        """Test that summary raises an error if model is not fitted."""
        model = LinearRegressionModel()
        
        with pytest.raises(ValueError):
            model.summary()
    
    def test_with_single_feature(self) -> None:
        """Test model with a single feature."""
        # Create simple dataset with one feature
        np.random.seed(42)
        X = np.random.rand(50, 1)
        y = 2 * X[:, 0] + 0.5 + np.random.normal(0, 0.1, 50)
        
        X_df = pd.DataFrame(X, columns=["feature1"])
        y_series = pd.Series(y, name="target")
        
        model = LinearRegressionModel()
        model.fit(X_df, y_series)
        
        # Check that coefficient is close to true value
        assert np.isclose(model.model.coef_[0], 2, rtol=0.2)
        assert np.isclose(model.model.intercept_, 0.5, rtol=0.2)
        
        # Test predictions
        X_test = pd.DataFrame({"feature1": [0.1, 0.5, 0.9]})
        predictions = model.predict(X_test)
        
        expected = 2 * X_test["feature1"].values + 0.5
        assert np.allclose(predictions, expected, rtol=0.3)
    
    def test_with_perfect_fit(self) -> None:
        """Test model with data that can be perfectly fit."""
        # Create dataset with exact linear relationship (no noise)
        # Use more diverse data points to ensure unique solution
        X = np.array([
            [1, 1],
            [2, 1],
            [1, 2],
            [2, 2],
            [3, 1],
            [1, 3]
        ])
        y = 2 * X[:, 0] + 3 * X[:, 1] + 1  # y = 2*x1 + 3*x2 + 1
        
        X_df = pd.DataFrame(X, columns=["feature1", "feature2"])
        y_series = pd.Series(y, name="target")
        
        model = LinearRegressionModel()
        model.fit(X_df, y_series)
        
        # Check that coefficients are close to true values
        assert np.allclose(model.model.coef_, [2, 3], rtol=1e-10, atol=1e-10)
        assert np.isclose(model.model.intercept_, 1)
        
        # Evaluate model - should have perfect metrics
        metrics = model.evaluate(X_df, y_series)
        assert np.isclose(metrics["mse"], 0)
        assert np.isclose(metrics["rmse"], 0)
        assert np.isclose(metrics["mae"], 0)
        assert np.isclose(metrics["r2"], 1.0)
        assert np.isclose(metrics["explained_variance"], 1.0)
        
        # For perfect predictions, rmse_to_stdev should be very close to 0
        # Use isclose to handle floating point precision issues
        assert np.isclose(metrics["rmse_to_stdev"], 0.0, atol=1e-10)
    
    def test_with_constant_target(self) -> None:
        """Test model with constant target values."""
        # Create dataset with constant target
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.ones(5) * 5  # All targets are 5
        
        X_df = pd.DataFrame(X, columns=["feature1", "feature2"])
        y_series = pd.Series(y, name="target")
        
        model = LinearRegressionModel()
        model.fit(X_df, y_series)
        
        # Coefficients should be zero (or very close)
        assert np.allclose(model.model.coef_, [0, 0], atol=1e-10)
        
        # Intercept should be equal to the constant target
        assert np.isclose(model.model.intercept_, 5)
        
        # Predictions should all be the constant value
        predictions = model.predict(X_df)
        assert np.allclose(predictions, 5)
        
        # For constant targets, metrics can behave differently across implementations
        metrics = model.evaluate(X_df, y_series)
        assert np.isclose(metrics["mse"], 0) or np.isnan(metrics["mse"])
        
        # For constant targets, R² can be 1.0 (perfect prediction) or 0.0/NaN (no variance explained)
        # Both are mathematically valid interpretations
        assert (np.isclose(metrics["r2"], 0) or 
                np.isclose(metrics["r2"], 1.0) or 
                np.isnan(metrics["r2"]))
        
        # Similarly for explained variance
        assert (np.isclose(metrics["explained_variance"], 0) or 
                np.isclose(metrics["explained_variance"], 1.0) or 
                np.isnan(metrics["explained_variance"]))
        
        # For constant targets, rmse_to_stdev should be very close to 0
        # Use isclose to handle floating point precision issues
        assert np.isclose(metrics["rmse_to_stdev"], 0.0, atol=1e-10)
