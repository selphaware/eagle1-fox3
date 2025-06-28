"""
Unit tests for the Random Forest Classifier implementation.
"""

__author__ = "Usman Ahmad"

import pytest
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING, Dict, Any, List, Tuple

from ml.classification import RandomForestClassifier

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


class TestRandomForestClassifier:
    """Tests for the RandomForestClassifier class."""
    
    def test_initialization(self) -> None:
        """Test that RandomForestClassifier initializes correctly with default and custom parameters."""
        # Test with default parameters
        rf = RandomForestClassifier()
        assert rf.n_estimators == 100
        assert rf.max_depth is None
        assert rf.min_samples_split == 2
        assert rf.min_samples_leaf == 1
        assert rf.max_features == 'sqrt'
        assert rf.bootstrap is True
        assert rf.criterion == 'gini'
        assert rf.random_state is None
        assert rf.class_weight is None
        assert rf.is_trained is False
        
        # Test with custom parameters
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='log2',
            bootstrap=False,
            criterion='entropy',
            random_state=42,
            class_weight='balanced'
        )
        assert rf.n_estimators == 50
        assert rf.max_depth == 10
        assert rf.min_samples_split == 5
        assert rf.min_samples_leaf == 2
        assert rf.max_features == 'log2'
        assert rf.bootstrap is False
        assert rf.criterion == 'entropy'
        assert rf.random_state == 42
        assert rf.class_weight == 'balanced'
        assert rf.is_trained is False
    
    def test_train_with_invalid_inputs(self) -> None:
        """Test that train method raises appropriate errors for invalid inputs."""
        rf = RandomForestClassifier()
        
        # Test with invalid X type
        with pytest.raises(TypeError, match="X must be a pandas DataFrame or numpy array"):
            rf.train("invalid_x", [0, 1, 0])
        
        # Test with invalid y type
        with pytest.raises(TypeError, match="y must be a pandas Series, numpy array, or list"):
            rf.train(pd.DataFrame({'a': [1, 2, 3]}), "invalid_y")
        
        # Test with incompatible shapes
        with pytest.raises(ValueError, match="X and y have incompatible shapes"):
            rf.train(
                pd.DataFrame({'a': [1, 2, 3]}),
                pd.Series([0, 1])
            )
    
    def test_train_binary_classification(self, sample_binary_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test training on binary classification data."""
        X, y = sample_binary_data
        rf = RandomForestClassifier(random_state=42)
        
        # Train model
        results = rf.train(X, y)
        
        # Check that model is trained
        assert rf.is_trained is True
        assert 'model' in results
        
        # Check feature importance
        assert 'feature_importance' in results
        feature_importance = results['feature_importance']
        assert isinstance(feature_importance, pd.DataFrame)
        assert len(feature_importance) == X.shape[1]
        assert 'feature' in feature_importance.columns
        assert 'importance' in feature_importance.columns
        
        # Check that feature importance sums to approximately 1
        assert np.isclose(feature_importance['importance'].sum(), 1.0, atol=1e-5)
    
    def test_train_with_hyperparameter_tuning(self, sample_binary_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test training with hyperparameter tuning."""
        X, y = sample_binary_data
        rf = RandomForestClassifier(random_state=42)
        
        # Train model with hyperparameter tuning
        results = rf.train(X, y, tune_hyperparameters=True, cv_folds=3)
        
        # Check that model is trained
        assert rf.is_trained is True
        assert 'model' in results
        
        # Check tuning results
        assert 'best_params' in results
        assert 'cv_results' in results
        
        # Check that best_params contains expected keys
        best_params = results['best_params']
        assert 'classifier__n_estimators' in best_params
        assert 'classifier__max_depth' in best_params
        assert 'classifier__min_samples_split' in best_params
        assert 'classifier__min_samples_leaf' in best_params
        assert 'classifier__max_features' in best_params
    
    def test_train_multiclass_classification(self, sample_multiclass_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test training on multiclass classification data."""
        X, y = sample_multiclass_data
        rf = RandomForestClassifier(random_state=42)
        
        # Train model
        results = rf.train(X, y)
        
        # Check that model is trained
        assert rf.is_trained is True
        assert 'model' in results
        
        # Check feature importance
        assert 'feature_importance' in results
        feature_importance = results['feature_importance']
        assert isinstance(feature_importance, pd.DataFrame)
        assert len(feature_importance) == X.shape[1]
    
    def test_predict_without_training(self) -> None:
        """Test that predict raises error if model is not trained."""
        rf = RandomForestClassifier()
        X = pd.DataFrame({'feature1': [0.5, -0.5], 'feature2': [0.2, -0.2]})
        
        with pytest.raises(ValueError, match="Model has not been trained"):
            rf.predict(X)
    
    def test_predict_binary_classification(self, sample_binary_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test prediction on binary classification data."""
        X, y = sample_binary_data
        rf = RandomForestClassifier(random_state=42)
        
        # Train model
        rf.train(X, y)
        
        # Make predictions
        y_pred = rf.predict(X)
        
        # Check predictions shape and type
        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == (len(X),)
        assert np.all(np.isin(y_pred, [0, 1]))
        
        # Check accuracy (should be good on training data)
        accuracy = np.mean(y_pred == y)
        assert accuracy > 0.7
    
    def test_predict_proba(self, sample_binary_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test probability prediction."""
        X, y = sample_binary_data
        rf = RandomForestClassifier(random_state=42)
        
        # Train model
        rf.train(X, y)
        
        # Test without training
        with pytest.raises(ValueError, match="Model has not been trained"):
            RandomForestClassifier().predict_proba(X)
        
        # Make probability predictions
        y_proba = rf.predict_proba(X)
        
        # Check predictions shape and properties
        assert isinstance(y_proba, np.ndarray)
        assert y_proba.shape == (len(X), 2)  # Binary classification, 2 classes
        assert np.all(y_proba >= 0)
        assert np.all(y_proba <= 1)
        assert np.allclose(np.sum(y_proba, axis=1), 1.0)
    
    def test_evaluate_without_training(self) -> None:
        """Test that evaluate raises error if model is not trained."""
        rf = RandomForestClassifier()
        X = pd.DataFrame({'feature1': [0.5, -0.5], 'feature2': [0.2, -0.2]})
        y = pd.Series([1, 0])
        
        with pytest.raises(ValueError, match="Model has not been trained"):
            rf.evaluate(X, y)
    
    def test_evaluate_binary_classification(self, sample_binary_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test evaluation on binary classification data."""
        X, y = sample_binary_data
        rf = RandomForestClassifier(random_state=42)
        
        # Train model
        rf.train(X, y)
        
        # Evaluate model
        metrics = rf.evaluate(X, y)
        
        # Check metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        assert 'confusion_matrix' in metrics
        
        # Check metric values (should be good on training data)
        assert metrics['accuracy'] > 0.7
        assert metrics['precision'] > 0.7
        assert metrics['recall'] > 0.7
        assert metrics['f1'] > 0.7
        assert metrics['roc_auc'] > 0.7
    
    def test_evaluate_multiclass_classification(self, sample_multiclass_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test evaluation on multiclass classification data."""
        X, y = sample_multiclass_data
        rf = RandomForestClassifier(random_state=42)
        
        # Train model
        rf.train(X, y)
        
        # Evaluate model with macro averaging
        metrics_macro = rf.evaluate(X, y, average='macro')
        
        # Check metrics
        assert 'accuracy' in metrics_macro
        assert 'precision' in metrics_macro
        assert 'recall' in metrics_macro
        assert 'f1' in metrics_macro
        assert 'confusion_matrix' in metrics_macro
        
        # Evaluate model with weighted averaging
        metrics_weighted = rf.evaluate(X, y, average='weighted')
        
        # Check metrics
        assert 'accuracy' in metrics_weighted
        assert 'precision' in metrics_weighted
        assert 'recall' in metrics_weighted
        assert 'f1' in metrics_weighted
        
        # Check that accuracy is the same regardless of averaging method
        assert metrics_macro['accuracy'] == metrics_weighted['accuracy']
    
    def test_get_feature_importance_without_training(self) -> None:
        """Test that get_feature_importance raises error if model is not trained."""
        rf = RandomForestClassifier()
        
        with pytest.raises(ValueError, match="Model has not been trained"):
            rf.get_feature_importance(['feature1', 'feature2'])
    
    def test_get_feature_importance(self, sample_binary_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        """Test feature importance extraction."""
        X, y = sample_binary_data
        rf = RandomForestClassifier(random_state=42)
        
        # Train model
        rf.train(X, y)
        
        # Get feature importance with feature names
        feature_names = X.columns.tolist()
        importance_df = rf.get_feature_importance(feature_names)
        
        # Check feature importance DataFrame
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == len(feature_names)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert set(importance_df['feature']) == set(feature_names)
        assert importance_df['importance'].sum() > 0.99  # Should sum to approximately 1
        
        # Get feature importance without feature names
        importance_df_no_names = rf.get_feature_importance()
        
        # Check feature importance DataFrame
        assert isinstance(importance_df_no_names, pd.DataFrame)
        assert len(importance_df_no_names) == len(feature_names)
        assert 'feature' in importance_df_no_names.columns
        assert 'importance' in importance_df_no_names.columns
        assert all(f.startswith('feature_') for f in importance_df_no_names['feature'])
        
        # Test with mismatched feature names length
        with pytest.raises(ValueError):
            rf.get_feature_importance(['single_feature'])
