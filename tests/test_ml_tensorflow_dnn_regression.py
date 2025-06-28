"""
Unit tests for TensorFlow DNN Regressor model.

This module tests the TensorFlowDNNRegressor class from ml.regression.tensorflow_dnn.
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Union, Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from ml.regression import TensorFlowDNNRegressor

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestTensorFlowDNNRegressor:
    """Test cases for TensorFlowDNNRegressor."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test environment."""
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Suppress TensorFlow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')

    def test_initialization(self) -> None:
        """Test model initialization with default and custom parameters."""
        # Test with default parameters
        model = TensorFlowDNNRegressor()
        assert model.hidden_layers == [64, 32]
        assert model.activation == 'relu'
        assert model.dropout_rate == 0.2
        assert model.learning_rate == 0.001
        assert model.batch_size == 32
        assert model.epochs == 100
        assert model.is_fitted is False
        assert model.model is None

        # Test with custom parameters
        model = TensorFlowDNNRegressor(
            hidden_layers=[128, 64, 32],
            activation='tanh',
            dropout_rate=0.3,
            learning_rate=0.01,
            batch_size=64,
            epochs=50,
            random_state=42
        )
        assert model.hidden_layers == [128, 64, 32]
        assert model.activation == 'tanh'
        assert model.dropout_rate == 0.3
        assert model.learning_rate == 0.01
        assert model.batch_size == 64
        assert model.epochs == 50
        assert model.is_fitted is False

    def test_build_model(self) -> None:
        """Test model architecture building."""
        model = TensorFlowDNNRegressor(
            hidden_layers=[64, 32],
            activation='relu',
            dropout_rate=0.2,
            use_batch_norm=True
        )
        
        # Build model with 5 input features
        tf_model = model._build_model(5)
        
        # Check model structure
        assert isinstance(tf_model, tf.keras.Sequential)
        
        # Count layers
        # For each hidden layer we have: Dense + BatchNorm + Dropout
        # Plus output layer (Dense)
        expected_layers = (3 * len(model.hidden_layers)) + 1
        assert len(tf_model.layers) == expected_layers
        
        # Check input shape
        assert tf_model.layers[0].input_shape == (None, 5)
        
        # Check output layer
        assert tf_model.layers[-1].units == 1
        assert tf_model.layers[-1].activation.__name__ == 'linear'

    def test_build_model_without_batch_norm(self) -> None:
        """Test model architecture building without batch normalization."""
        model = TensorFlowDNNRegressor(
            hidden_layers=[64, 32],
            activation='relu',
            dropout_rate=0.2,
            use_batch_norm=False
        )
        
        # Build model with 5 input features
        tf_model = model._build_model(5)
        
        # Check model structure
        assert isinstance(tf_model, tf.keras.Sequential)
        
        # Count layers
        # For each hidden layer: Dense + Dropout
        # Plus output layer
        expected_layers = (2 * len(model.hidden_layers)) + 1
        assert len(tf_model.layers) == expected_layers

    def test_fit_with_dataframe(self) -> None:
        """Test model fitting with pandas DataFrame and Series."""
        # Create simple linear data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        })
        y = pd.Series([2*x1 + 3*x2 for x1, x2 in zip(X['feature1'], X['feature2'])], name='target')

        # Fit model with minimal epochs for test speed
        model = TensorFlowDNNRegressor(
            epochs=5,
            batch_size=2,
            validation_split=0.2,
            random_state=42
        )
        model.fit(X, y, verbose=0)

        # Check model state
        assert model.is_fitted is True
        assert model.feature_names == ['feature1', 'feature2']
        assert model.target_name == 'target'
        assert model.n_features == 2
        assert model.model is not None
        assert model.history is not None
        assert len(model.history.epoch) <= 5  # May stop early due to early stopping

    def test_fit_with_numpy(self) -> None:
        """Test model fitting with numpy arrays."""
        # Create simple linear data
        X = np.array([[1, 10], [2, 9], [3, 8], [4, 7], [5, 6], 
                      [6, 5], [7, 4], [8, 3], [9, 2], [10, 1]])
        y = np.array([2*x[0] + 3*x[1] for x in X])

        # Fit model with feature names
        model = TensorFlowDNNRegressor(
            epochs=5,
            batch_size=2,
            validation_split=0.2,
            random_state=42
        )
        feature_names = ['feature1', 'feature2']
        model.fit(X, y, feature_names=feature_names, verbose=0)

        # Check model state
        assert model.is_fitted is True
        assert model.feature_names == feature_names
        assert model.target_name == 'target'  # Default target name
        assert model.n_features == 2
        assert model.model is not None

        # Fit model without feature names
        model = TensorFlowDNNRegressor(
            epochs=5,
            batch_size=2,
            validation_split=0.2,
            random_state=42
        )
        model.fit(X, y, verbose=0)

        # Check auto-generated feature names
        assert model.is_fitted is True
        assert model.feature_names == ['feature_0', 'feature_1']

    def test_fit_with_invalid_inputs(self) -> None:
        """Test model fitting with invalid inputs."""
        model = TensorFlowDNNRegressor()

        # Test with invalid X type
        with pytest.raises(TypeError):
            model.fit("not_a_dataframe", np.array([1, 2, 3]), verbose=0)

        # Test with invalid y type
        with pytest.raises(TypeError):
            model.fit(np.array([[1, 2], [3, 4]]), "not_an_array", verbose=0)

        # Test with mismatched feature_names length
        with pytest.raises(ValueError):
            model.fit(
                np.array([[1, 2], [3, 4]]),
                np.array([1, 2]),
                feature_names=['feature1'],  # Should be 2 features
                verbose=0
            )

    def test_predict(self) -> None:
        """Test model prediction."""
        # Create and fit model with simple data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })
        y = pd.Series([2*x1 + 3*x2 for x1, x2 in zip(X['feature1'], X['feature2'])])

        model = TensorFlowDNNRegressor(
            epochs=5,
            batch_size=2,
            validation_split=0.2,
            random_state=42
        )
        model.fit(X, y, verbose=0)

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
        model = TensorFlowDNNRegressor()
        X = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError):
            model.predict(X)

    def test_predict_with_invalid_inputs(self) -> None:
        """Test prediction with invalid inputs."""
        # Create and fit model
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])

        model = TensorFlowDNNRegressor(
            epochs=5,
            batch_size=2,
            validation_split=0.2,
            random_state=42
        )
        model.fit(X, y, verbose=0)

        # Test with invalid X type
        with pytest.raises(TypeError):
            model.predict("not_a_dataframe")

        # Test with wrong number of features
        with pytest.raises(ValueError):
            model.predict(np.array([[1, 2, 3], [4, 5, 6]]))  # 3 features instead of 2

    def test_evaluate(self) -> None:
        """Test model evaluation."""
        # Create and fit model with simple data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        })
        y = pd.Series([2*x1 + 3*x2 for x1, x2 in zip(X['feature1'], X['feature2'])])

        model = TensorFlowDNNRegressor(
            epochs=10,
            batch_size=2,
            validation_split=0.2,
            random_state=42
        )
        model.fit(X, y, verbose=0)

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

    def test_get_training_history(self) -> None:
        """Test getting training history."""
        # Create and fit model
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([5, 11, 17, 23, 29])  # y = 2*x1 + 3*x2

        model = TensorFlowDNNRegressor(
            epochs=5,
            batch_size=2,
            validation_split=0.2,
            random_state=42
        )
        model.fit(X, y, verbose=0)

        # Get history
        history = model.get_training_history()

        # Check history structure
        assert isinstance(history, dict)
        assert 'loss' in history
        assert 'val_loss' in history
        assert 'mae' in history
        assert 'val_mae' in history
        assert 'mse' in history
        assert 'val_mse' in history

        # Check history values
        assert len(history['loss']) <= 5  # May stop early due to early stopping
        assert all(isinstance(val, float) for val in history['loss'])

    def test_get_training_history_unfitted(self) -> None:
        """Test getting history with unfitted model."""
        model = TensorFlowDNNRegressor()

        with pytest.raises(ValueError):
            model.get_training_history()

    def test_save_and_load_model(self) -> None:
        """Test saving and loading model."""
        # Create and fit model
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([5, 11, 17, 23, 29])  # y = 2*x1 + 3*x2

        model = TensorFlowDNNRegressor(
            epochs=5,
            batch_size=2,
            validation_split=0.2,
            random_state=42
        )
        model.fit(X, y, verbose=0)

        # Make predictions with original model
        original_predictions = model.predict(X)

        # Save model to temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.keras")
            model.save_model(model_path)

            # Create new model with same architecture
            new_model = TensorFlowDNNRegressor(
                epochs=5,
                batch_size=2,
                validation_split=0.2,
                random_state=42
            )
            
            # Set the necessary attributes before loading
            new_model.n_features = 2
            new_model.feature_names = ['feature_0', 'feature_1']
            
            # Load the model
            new_model.load_model(model_path)

            # Check model state
            assert new_model.is_fitted is True
            assert new_model.model is not None

            # Make predictions with loaded model
            loaded_predictions = new_model.predict(X)

            # Check that predictions are the same
            assert np.allclose(original_predictions, loaded_predictions)

    def test_save_model_unfitted(self) -> None:
        """Test saving unfitted model."""
        model = TensorFlowDNNRegressor()

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.h5")
            with pytest.raises(ValueError):
                model.save_model(model_path)

    def test_load_model_nonexistent(self) -> None:
        """Test loading model from nonexistent file."""
        model = TensorFlowDNNRegressor()

        with pytest.raises(Exception):
            model.load_model("/nonexistent/path/model.h5")

    def test_summary(self) -> None:
        """Test model summary."""
        # Create and fit model
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })
        y = pd.Series([2*x1 + 3*x2 for x1, x2 in zip(X['feature1'], X['feature2'])])

        model = TensorFlowDNNRegressor(
            hidden_layers=[32, 16],
            activation='relu',
            dropout_rate=0.2,
            epochs=5,
            batch_size=2,
            random_state=42
        )
        model.fit(X, y, verbose=0)

        # Get summary
        summary = model.summary()

        # Check summary content
        assert isinstance(summary, dict)
        assert summary['model_type'] == 'TensorFlowDNNRegressor'
        assert summary['hidden_layers'] == [32, 16]
        assert summary['activation'] == 'relu'
        assert summary['dropout_rate'] == 0.2
        assert summary['n_features'] == 2
        assert summary['feature_names'] == ['feature1', 'feature2']
        assert summary['is_fitted'] is True
        assert 'model_architecture' in summary
        assert isinstance(summary['model_architecture'], str)

    def test_summary_unfitted(self) -> None:
        """Test summary with unfitted model."""
        model = TensorFlowDNNRegressor()

        with pytest.raises(ValueError):
            model.summary()

    def test_with_nonlinear_data(self) -> None:
        """Test model with nonlinear data where DNN should excel."""
        # Create nonlinear data
        np.random.seed(42)
        X = np.random.rand(200, 2)  # 200 samples, 2 features
        # Nonlinear function: y = x1^2 + sin(x2) + noise
        y = X[:, 0]**2 + np.sin(4 * X[:, 1]) + np.random.normal(0, 0.05, 200)

        X_df = pd.DataFrame(X, columns=['feature1', 'feature2'])
        y_series = pd.Series(y, name='target')

        # Fit DNN model with more epochs for better learning
        dnn_model = TensorFlowDNNRegressor(
            hidden_layers=[64, 32, 16],  # Larger network
            epochs=50,                   # More epochs
            batch_size=16,
            dropout_rate=0.1,            # Less dropout
            validation_split=0.2,
            random_state=42
        )
        dnn_model.fit(X_df, y_series, verbose=0)

        # Evaluate model
        metrics = dnn_model.evaluate(X_df, y_series)

        # For test purposes, we'll just check that the model learned something
        # In a real test, we might compare against a baseline or use a fixed threshold
        assert metrics['r2'] > 0.0  # Just ensure it's better than predicting the mean

    def test_logging(self, caplog: "LogCaptureFixture") -> None:
        """Test that the model logs important events."""
        caplog.set_level(logging.INFO)

        # Create and fit model
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })
        y = pd.Series([2*x1 + 3*x2 for x1, x2 in zip(X['feature1'], X['feature2'])])

        model = TensorFlowDNNRegressor(
            epochs=2,  # Minimal epochs for test speed
            batch_size=2,
            random_state=42
        )
        model.fit(X, y, verbose=0)
        model.predict(X)
        model.evaluate(X, y)
        model.get_training_history()
        model.summary()

        # Check that logs were created
        assert "Initialized TensorFlowDNNRegressor" in caplog.text
        assert "Built TensorFlow DNN model" in caplog.text
        assert "Fitted TensorFlowDNNRegressor" in caplog.text
        assert "Made predictions" in caplog.text
        assert "Evaluated model" in caplog.text
        assert "Generated model summary" in caplog.text
