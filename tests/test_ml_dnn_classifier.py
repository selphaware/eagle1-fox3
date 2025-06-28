"""
Unit tests for the TensorFlow DNN Classifier implementation.
"""

__author__ = "Usman Ahmad"

import pytest
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import TYPE_CHECKING, Dict, Any, List, Tuple

from ml.classification import DNNClassifier

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def sample_binary_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a simple binary classification dataset for testing.
    
    Returns:
        Tuple containing features DataFrame and target Series.
    """
    # Create features with two informative features
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100)
    })
    
    # Create target with linear relationship to features
    # y = 1 if feature1 + feature2 > 0 else 0
    y = pd.Series((X['feature1'] + X['feature2'] > 0).astype(int))
    
    return X, y


@pytest.fixture
def sample_multiclass_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a simple multiclass classification dataset for testing.
    
    Returns:
        Tuple containing features DataFrame and target Series.
    """
    # Create features with three informative features
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 150),
        'feature2': np.random.normal(0, 1, 150),
        'feature3': np.random.normal(0, 1, 150)
    })
    
    # Create target with 3 classes based on feature combinations
    y = pd.Series(np.zeros(150, dtype=int))
    y[X['feature1'] + X['feature2'] > 1] = 1
    y[X['feature1'] - X['feature2'] + X['feature3'] > 1.5] = 2
    
    return X, y


class TestDNNClassifier:
    """Tests for the DNNClassifier class."""
    
    def test_initialization(self) -> None:
        """Test that DNNClassifier initializes correctly with default and custom parameters."""
        # Test with default parameters
        dnn = DNNClassifier()
        assert dnn.hidden_layers == [128, 64, 32]
        assert dnn.activation == 'relu'
        assert dnn.dropout_rate == 0.2
        assert dnn.use_batch_norm is True
        assert dnn.learning_rate == 0.001
        assert dnn.batch_size == 32
        assert dnn.epochs == 50
        assert dnn.early_stopping_patience == 10
        assert dnn.is_trained is False
        assert dnn.model is None
        
        # Test with custom parameters
        dnn = DNNClassifier(
            hidden_layers=[64, 32],
            activation='tanh',
            dropout_rate=0.3,
            use_batch_norm=False,
            learning_rate=0.01,
            batch_size=64,
            epochs=100,
            early_stopping_patience=5,
            random_state=42
        )
        assert dnn.hidden_layers == [64, 32]
        assert dnn.activation == 'tanh'
        assert dnn.dropout_rate == 0.3
        assert dnn.use_batch_norm is False
        assert dnn.learning_rate == 0.01
        assert dnn.batch_size == 64
        assert dnn.epochs == 100
        assert dnn.early_stopping_patience == 5
        assert dnn.is_trained is False
        assert dnn.model is None
    
    def test_model_architecture(self) -> None:
        """Test that the model architecture is built correctly."""
        dnn = DNNClassifier(
            hidden_layers=[64, 32],
            activation='relu',
            dropout_rate=0.2,
            use_batch_norm=True,
            random_state=42
        )
        
        # Build model for binary classification
        model = dnn._build_model(input_dim=10, num_classes=2)
        
        # Check model type and structure
        assert isinstance(model, tf.keras.Sequential)
        
        # Check input shape
        assert model.input_shape == (None, 10)
        
        # Check output shape for binary classification
        assert model.output_shape == (None, 1)
        
        # Build model for multi-class classification
        model = dnn._build_model(input_dim=10, num_classes=3)
        
        # Check output shape for multi-class classification
        assert model.output_shape == (None, 3)
        
        # Check that the model contains expected layer types
        layer_types = [type(layer).__name__ for layer in model.layers]
        
        # Check for expected layer types
        assert 'Dense' in layer_types
        assert 'BatchNormalization' in layer_types
        assert 'Dropout' in layer_types
    
    def test_train_with_invalid_inputs(self) -> None:
        """Test that train method raises appropriate errors for invalid inputs."""
        dnn = DNNClassifier(epochs=1)  # Use small epochs for faster tests
        
        # Test with invalid X type
        with pytest.raises(TypeError, match="X must be a pandas DataFrame or numpy array"):
            dnn.train("invalid_x", [0, 1, 0])
        
        # Test with invalid y type
        with pytest.raises(TypeError, match="y must be a pandas Series, numpy array, or list"):
            dnn.train(pd.DataFrame({'a': [1, 2, 3]}), "invalid_y")
        
        # Test with incompatible shapes
        with pytest.raises(ValueError, match="X and y have incompatible shapes"):
            dnn.train(
                pd.DataFrame({'a': [1, 2, 3]}),
                pd.Series([0, 1])
            )
    
    @pytest.mark.parametrize("use_batch_norm", [True, False])
    @pytest.mark.parametrize("dropout_rate", [0.0, 0.2])
    def test_train_binary_classification(self, sample_binary_data: Tuple[pd.DataFrame, pd.Series], 
                                        use_batch_norm: bool, dropout_rate: float) -> None:
        """Test training on binary classification data with different configurations."""
        X, y = sample_binary_data
        
        # Create a small model for faster testing
        dnn = DNNClassifier(
            hidden_layers=[16, 8],
            epochs=2,  # Use very few epochs for testing
            batch_size=32,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            random_state=42
        )
        
        # Train model
        results = dnn.train(X, y, validation_split=0.2)
        
        # Check that model is trained
        assert dnn.is_trained is True
        assert dnn.model is not None
        assert 'model' in results
        assert 'history' in results
        assert 'feature_names' in results
        
        # Check that history contains expected metrics
        history = results['history']
        assert 'loss' in history
        assert 'accuracy' in history
        assert 'val_loss' in history
        assert 'val_accuracy' in history
        
        # Check feature names
        assert results['feature_names'] == X.columns.tolist()
        
        # Check number of classes
        assert dnn.num_classes == 2
    
    def test_train_multiclass_classification(self, sample_multiclass_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test training on multiclass classification data."""
        X, y = sample_multiclass_data
        
        # Create a small model for faster testing
        dnn = DNNClassifier(
            hidden_layers=[16, 8],
            epochs=2,  # Use very few epochs for testing
            batch_size=32,
            random_state=42
        )
        
        # Train model
        results = dnn.train(X, y)
        
        # Check that model is trained
        assert dnn.is_trained is True
        assert dnn.model is not None
        
        # Check number of classes
        assert dnn.num_classes == 3
        
        # Check output layer shape
        assert dnn.model.output_shape == (None, 3)
    
    def test_predict_without_training(self) -> None:
        """Test that predict raises error if model is not trained."""
        dnn = DNNClassifier()
        X = pd.DataFrame({'feature1': [0.5, -0.5], 'feature2': [0.2, -0.2]})
        
        with pytest.raises(ValueError, match="Model has not been trained"):
            dnn.predict(X)
    
    def test_predict_binary_classification(self, sample_binary_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test prediction on binary classification data."""
        X, y = sample_binary_data
        
        # Create and train a small model
        dnn = DNNClassifier(
            hidden_layers=[16, 8],
            epochs=5,  # Use few epochs for testing
            batch_size=32,
            random_state=42
        )
        dnn.train(X, y)
        
        # Make predictions
        y_pred = dnn.predict(X)
        
        # Check predictions shape and type
        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == (len(X),)
        assert np.all(np.isin(y_pred, [0, 1]))
    
    def test_predict_proba(self, sample_binary_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test probability prediction."""
        X, y = sample_binary_data
        
        # Create and train a small model
        dnn = DNNClassifier(
            hidden_layers=[16, 8],
            epochs=5,  # Use few epochs for testing
            batch_size=32,
            random_state=42
        )
        dnn.train(X, y)
        
        # Test without training
        with pytest.raises(ValueError, match="Model has not been trained"):
            DNNClassifier().predict_proba(X)
        
        # Make probability predictions
        y_proba = dnn.predict_proba(X)
        
        # Check predictions shape and properties
        assert isinstance(y_proba, np.ndarray)
        assert y_proba.shape == (len(X), 2)  # Binary classification, 2 classes
        assert np.all(y_proba >= 0)
        assert np.all(y_proba <= 1)
        assert np.allclose(np.sum(y_proba, axis=1), 1.0)
    
    def test_evaluate_without_training(self) -> None:
        """Test that evaluate raises error if model is not trained."""
        dnn = DNNClassifier()
        X = pd.DataFrame({'feature1': [0.5, -0.5], 'feature2': [0.2, -0.2]})
        y = pd.Series([1, 0])
        
        with pytest.raises(ValueError, match="Model has not been trained"):
            dnn.evaluate(X, y)
    
    def test_evaluate_binary_classification(self, sample_binary_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test evaluation on binary classification data."""
        X, y = sample_binary_data
        
        # Create and train a small model
        dnn = DNNClassifier(
            hidden_layers=[16, 8],
            epochs=5,  # Use few epochs for testing
            batch_size=32,
            random_state=42
        )
        dnn.train(X, y)
        
        # Evaluate model
        metrics = dnn.evaluate(X, y)
        
        # Check metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        assert 'confusion_matrix' in metrics
        assert 'loss' in metrics
        assert 'accuracy_tf' in metrics
    
    def test_get_feature_importance_without_training(self) -> None:
        """Test that get_feature_importance raises error if model is not trained."""
        dnn = DNNClassifier()
        
        with pytest.raises(ValueError, match="Model has not been trained"):
            dnn.get_feature_importance(['feature1', 'feature2'])
    
    def test_get_feature_importance(self, sample_binary_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test feature importance extraction."""
        X, y = sample_binary_data
        
        # Create and train a small model
        dnn = DNNClassifier(
            hidden_layers=[16, 8],
            epochs=5,  # Use few epochs for testing
            batch_size=32,
            random_state=42
        )
        dnn.train(X, y)
        
        # Test 1: Get feature importance with feature names
        feature_names = X.columns.tolist()
        importance_df = dnn.get_feature_importance(
            feature_names=feature_names,
            X_eval=X,  # Use training data for evaluation
            n_repeats=2,  # Use fewer repeats for testing
            random_state=42
        )
        
        # Check feature importance DataFrame
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == len(feature_names)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert set(importance_df['feature']) == set(feature_names)
        assert np.isclose(importance_df['importance'].sum(), 1.0) or importance_df['importance'].sum() == 0
        
        # Test 2: Get feature importance with default parameters
        importance_df2 = dnn.get_feature_importance(feature_names=feature_names)
        assert isinstance(importance_df2, pd.DataFrame)
        assert len(importance_df2) == len(feature_names)
        
        # Test 3: With mismatched feature names length
        with pytest.raises(ValueError, match="Length mismatch between feature_names"):
            dnn.get_feature_importance(['single_feature'])
    
    def test_get_model_summary(self) -> None:
        """Test getting model summary."""
        dnn = DNNClassifier()
        
        # Test without building model
        with pytest.raises(ValueError, match="Model has not been built yet"):
            dnn.get_model_summary()
        
        # Train model
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'feature2': np.random.normal(0, 1, 50)
        })
        y = pd.Series((X['feature1'] + X['feature2'] > 0).astype(int))
        
        dnn.train(X, y, validation_split=0.2)
        
        # Get summary
        summary = dnn.get_model_summary()
        
        # Check summary content
        assert isinstance(summary, str)
        assert "Model:" in summary or "Layer" in summary
        assert "Dense" in summary
