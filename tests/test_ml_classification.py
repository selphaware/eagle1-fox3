"""
Unit tests for the ML classification module.
"""

__author__ = "Usman Ahmad"

import pytest
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING, Dict, Any, List, Tuple

from ml.classification import LogisticRegressionModel

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
    
    # Create target with 3 classes
    # Class 0: feature1 > 0.5
    # Class 1: feature1 <= 0.5 and feature2 > 0
    # Class 2: feature1 <= 0.5 and feature2 <= 0
    conditions = [
        X['feature1'] > 0.5,
        (X['feature1'] <= 0.5) & (X['feature2'] > 0)
    ]
    choices = [0, 1]
    y = pd.Series(np.select(conditions, choices, default=2))
    
    return X, y


class TestLogisticRegressionModel:
    """Tests for the LogisticRegressionModel class."""
    
    def test_initialization(self) -> None:
        """Test that LogisticRegressionModel initializes correctly with default and custom parameters."""
        # Test with default parameters
        model = LogisticRegressionModel()
        assert model.C == 1.0
        assert model.penalty == 'l2'
        assert model.solver == 'lbfgs'
        assert model.max_iter == 1000
        assert model.random_state is None
        assert model.class_weight is None
        assert model.multi_class == 'auto'
        assert model.is_trained is False
        
        # Test with custom parameters
        model = LogisticRegressionModel(
            C=0.5,
            penalty='l2',
            solver='newton-cg',
            max_iter=2000,
            random_state=42,
            class_weight='balanced',
            multi_class='ovr'
        )
        assert model.C == 0.5
        assert model.penalty == 'l2'
        assert model.solver == 'newton-cg'
        assert model.max_iter == 2000
        assert model.random_state == 42
        assert model.class_weight == 'balanced'
        assert model.multi_class == 'ovr'
        assert model.is_trained is False
    
    def test_train_with_invalid_inputs(self) -> None:
        """Test that train method raises appropriate errors for invalid inputs."""
        model = LogisticRegressionModel()
        
        # Invalid X type
        with pytest.raises(TypeError, match="X must be a pandas DataFrame or numpy array"):
            model.train([1, 2, 3], [0, 1, 0])  # type: ignore
        
        # Invalid y type
        X = pd.DataFrame({'feature': [1, 2, 3]})
        with pytest.raises(TypeError, match="y must be a pandas Series, numpy array, or list"):
            model.train(X, 123)  # type: ignore
        
        # Incompatible shapes
        X = pd.DataFrame({'feature': [1, 2, 3, 4]})
        y = pd.Series([0, 1, 0])
        with pytest.raises(ValueError, match="X and y have incompatible shapes"):
            model.train(X, y)
    
    def test_train_binary_classification(self, sample_binary_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test training on binary classification data."""
        X, y = sample_binary_data
        
        # Create and train model
        model = LogisticRegressionModel(random_state=42)
        results = model.train(X, y)
        
        # Check results
        assert 'model' in results
        assert model.is_trained is True
        
        # Test with hyperparameter tuning (with reduced CV to speed up tests)
        model = LogisticRegressionModel(random_state=42)
        results = model.train(X, y, tune_hyperparameters=True, cv_folds=2)
        
        # Check tuning results
        assert 'model' in results
        assert 'best_params' in results
        assert 'cv_results' in results
        assert model.is_trained is True
    
    def test_train_multiclass_classification(self, sample_multiclass_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test training on multiclass classification data."""
        X, y = sample_multiclass_data
        
        # Create and train model
        model = LogisticRegressionModel(random_state=42, multi_class='multinomial')
        results = model.train(X, y)
        
        # Check results
        assert 'model' in results
        assert model.is_trained is True
        
        # Verify that model handles multiple classes
        unique_classes = len(np.unique(y))
        assert unique_classes > 2  # Ensure we have a multiclass problem
    
    def test_predict_without_training(self) -> None:
        """Test that predict raises error if model is not trained."""
        model = LogisticRegressionModel()
        X = pd.DataFrame({'feature': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Model has not been trained"):
            model.predict(X)
    
    def test_predict_binary_classification(self, sample_binary_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test prediction on binary classification data."""
        X, y = sample_binary_data
        
        # Split data into train and test
        train_size = int(0.7 * len(X))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train = y.iloc[:train_size]
        
        # Train model
        model = LogisticRegressionModel(random_state=42)
        model.train(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Check predictions
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
        assert set(np.unique(predictions)).issubset({0, 1})  # Binary predictions
    
    def test_predict_proba(self, sample_binary_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test probability prediction."""
        X, y = sample_binary_data
        
        # Split data into train and test
        train_size = int(0.7 * len(X))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train = y.iloc[:train_size]
        
        # Train model
        model = LogisticRegressionModel(random_state=42)
        model.train(X_train, y_train)
        
        # Make probability predictions
        probabilities = model.predict_proba(X_test)
        
        # Check probabilities
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (len(X_test), 2)  # Two classes
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)  # Valid probabilities
        assert np.allclose(np.sum(probabilities, axis=1), 1.0)  # Sum to 1
    
    def test_evaluate_without_training(self) -> None:
        """Test that evaluate raises error if model is not trained."""
        model = LogisticRegressionModel()
        X = pd.DataFrame({'feature': [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        
        with pytest.raises(ValueError, match="Model has not been trained"):
            model.evaluate(X, y)
    
    def test_evaluate_binary_classification(self, sample_binary_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test evaluation on binary classification data."""
        X, y = sample_binary_data
        
        # Split data into train and test
        train_size = int(0.7 * len(X))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Train model
        model = LogisticRegressionModel(random_state=42)
        model.train(X_train, y_train)
        
        # Evaluate model
        metrics = model.evaluate(X_test, y_test)
        
        # Check metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics  # Should be present for binary classification
        assert 'confusion_matrix' in metrics
        
        # Check metric values
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_evaluate_multiclass_classification(self, sample_multiclass_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test evaluation on multiclass classification data."""
        X, y = sample_multiclass_data
        
        # Split data into train and test
        train_size = int(0.7 * len(X))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Train model
        model = LogisticRegressionModel(random_state=42, multi_class='multinomial')
        model.train(X_train, y_train)
        
        # Evaluate model with macro averaging
        metrics = model.evaluate(X_test, y_test, average='macro')
        
        # Check metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'confusion_matrix' in metrics
        
        # Check metric values
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    def test_get_feature_importance_without_training(self) -> None:
        """Test that get_feature_importance raises error if model is not trained."""
        model = LogisticRegressionModel()
        
        with pytest.raises(ValueError, match="Model has not been trained"):
            model.get_feature_importance()
    
    def test_get_feature_importance(self, sample_binary_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test feature importance extraction."""
        X, y = sample_binary_data
        
        # Train model
        model = LogisticRegressionModel(random_state=42)
        model.train(X, y)
        
        # Get feature importance without feature names
        importance_df = model.get_feature_importance()
        
        # Check result
        assert isinstance(importance_df, pd.DataFrame)
        assert list(importance_df.columns) == ['feature', 'importance']
        assert len(importance_df) == X.shape[1]
        assert (importance_df['importance'] >= 0).all()
        
        # Get feature importance with feature names
        feature_names = list(X.columns)
        importance_df = model.get_feature_importance(feature_names)
        
        # Check result with feature names
        assert set(importance_df['feature']) == set(feature_names)
        
        # Check with mismatched feature names (should use generic names)
        wrong_names = ['wrong1', 'wrong2', 'wrong3']  # One extra name
        importance_df = model.get_feature_importance(wrong_names)
        
        # Should fall back to generic names
        assert len(importance_df) == X.shape[1]
        assert importance_df['feature'][0].startswith('feature_')
