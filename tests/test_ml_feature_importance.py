"""
Tests for feature importance functionality.

This module contains tests for the feature importance functions in ml.metrics.feature_importance.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Any, cast, TYPE_CHECKING
from unittest.mock import Mock
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier

from ml.metrics.feature_importance import get_feature_importance

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestFeatureImportance:
    """Tests for feature importance functions."""

    def test_tree_based_model(self) -> None:
        """Test feature importance extraction from tree-based models."""
        # Create a simple dataset
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        y = np.array([0, 1, 0, 1])
        
        # Train a random forest classifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        feature_names = ["feature1", "feature2", "feature3"]
        
        importance_df = get_feature_importance(model, feature_names)
        
        # Check that the DataFrame has the correct structure
        assert isinstance(importance_df, pd.DataFrame)
        assert list(importance_df.columns) == ["feature", "importance"]
        assert len(importance_df) == 3
        
        # Check that all feature names are present
        assert set(importance_df["feature"]) == set(feature_names)
        
        # Check that importances sum to approximately 1
        assert np.isclose(importance_df["importance"].sum(), 1.0, atol=1e-6)

    def test_linear_model_binary(self) -> None:
        """Test feature importance extraction from linear models (binary classification)."""
        # Create a simple dataset
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        y = np.array([0, 1, 0, 1])
        
        # Train a logistic regression model
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        feature_names = ["feature1", "feature2", "feature3"]
        
        importance_df = get_feature_importance(model, feature_names)
        
        # Check that the DataFrame has the correct structure
        assert isinstance(importance_df, pd.DataFrame)
        assert list(importance_df.columns) == ["feature", "importance"]
        assert len(importance_df) == 3
        
        # Check that all feature names are present
        assert set(importance_df["feature"]) == set(feature_names)
        
        # Check that importances are positive (absolute values of coefficients)
        assert (importance_df["importance"] >= 0).all()

    def test_linear_model_multiclass(self) -> None:
        """Test feature importance extraction from linear models (multiclass)."""
        # Create a simple dataset
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
        y = np.array([0, 1, 2, 0, 1, 2])
        
        # Train a logistic regression model for multiclass
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
        model.fit(X, y)
        
        feature_names = ["feature1", "feature2", "feature3"]
        
        importance_df = get_feature_importance(model, feature_names)
        
        # Check that the DataFrame has the correct structure
        assert isinstance(importance_df, pd.DataFrame)
        assert list(importance_df.columns) == ["feature", "importance"]
        assert len(importance_df) == 3
        
        # Check that all feature names are present
        assert set(importance_df["feature"]) == set(feature_names)
        
        # Check that importances are positive (absolute values of coefficients)
        assert (importance_df["importance"] >= 0).all()

    def test_ensemble_model(self) -> None:
        """Test feature importance extraction from ensemble models."""
        # Create a simple dataset
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        y = np.array([0, 1, 0, 1])
        
        # Create a base estimator
        tree_estimator = DecisionTreeClassifier(max_depth=1)
        
        # Train an AdaBoost ensemble model using 'estimator' parameter instead of deprecated 'base_estimator'
        model = AdaBoostClassifier(estimator=tree_estimator, n_estimators=3, random_state=42)
        model.fit(X, y)
        
        feature_names = ["feature1", "feature2", "feature3"]
        
        importance_df = get_feature_importance(model, feature_names)
        
        # Check that the DataFrame has the correct structure
        assert isinstance(importance_df, pd.DataFrame)
        assert list(importance_df.columns) == ["feature", "importance"]
        assert len(importance_df) == 3
        
        # Check that all feature names are present
        assert set(importance_df["feature"]) == set(feature_names)
        
        # Check that importances sum to approximately 1
        assert np.isclose(importance_df["importance"].sum(), 1.0, atol=1e-6)

    def test_unsupported_model(self) -> None:
        """Test that unsupported models raise ValueError."""
        # Create a simple class without feature_importances_ or coef_ attributes
        class UnsupportedModel:
            def __init__(self):
                pass
                
        model = UnsupportedModel()
        feature_names = ["feature1", "feature2", "feature3"]
        
        with pytest.raises(ValueError):
            get_feature_importance(model, feature_names)

    def test_feature_count_mismatch(self) -> None:
        """Test that feature count mismatch raises ValueError."""
        # Create a simple dataset and model
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        y = np.array([0, 1, 0, 1])
        
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        model.fit(X, y)
        
        # Feature names list has wrong length
        feature_names = ["feature1", "feature2", "feature3", "feature4"]
        
        with pytest.raises(ValueError, match="Number of features in model .* does not match"):
            get_feature_importance(model, feature_names)

    def test_invalid_feature_names(self) -> None:
        """Test that invalid feature names raise TypeError."""
        # Create a simple dataset and model
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        y = np.array([0, 1, 0, 1])
        
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        model.fit(X, y)
        
        # Feature names is not a list
        feature_names = "feature1,feature2,feature3"
        
        with pytest.raises(TypeError):
            get_feature_importance(model, feature_names)  # type: ignore
        
        # Feature names contains non-string elements
        feature_names = ["feature1", 2, "feature3"]
        
        with pytest.raises(TypeError):
            get_feature_importance(model, feature_names)  # type: ignore

    def test_zero_feature_importance(self) -> None:
        """Test model with zero feature importance."""
        # Create a simple dataset
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        y = np.array([0, 1, 0, 1])
        
        # Create a custom model class with zero feature importances
        class ZeroImportanceModel:
            def __init__(self):
                self.feature_importances_ = np.array([0.0, 0.0, 0.0])
        
        model = ZeroImportanceModel()
        feature_names = ["feature1", "feature2", "feature3"]
        
        importance_df = get_feature_importance(model, feature_names)
        
        # Check that all importances are zero
        assert all(importance_df["importance"] == 0.0)
        
        # Check that all feature names are present
        assert set(importance_df["feature"]) == set(feature_names)

    def test_single_feature(self) -> None:
        """Test model with a single feature."""
        # Create a simple dataset with a single feature
        X = np.array([[1], [4], [7], [10]])
        y = np.array([0, 1, 0, 1])
        
        # Train a random forest classifier
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        model.fit(X, y)
        
        feature_names = ["feature1"]
        
        importance_df = get_feature_importance(model, feature_names)
        
        # Check that the DataFrame has the correct structure
        assert isinstance(importance_df, pd.DataFrame)
        assert list(importance_df.columns) == ["feature", "importance"]
        assert len(importance_df) == 1
        
        # Check the single feature
        assert importance_df.iloc[0]["feature"] == "feature1"
        assert importance_df.iloc[0]["importance"] == 1.0  # Should be 1.0 since it's the only feature

    def test_uneven_feature_importance(self) -> None:
        """Test model with highly uneven feature importance distribution."""
        # Create a simple dataset where one feature is highly predictive
        X = np.array([
            [1, 0, 0],  # Feature 1 perfectly predicts class
            [1, 5, 6],
            [0, 7, 8],
            [0, 10, 12]
        ])
        y = np.array([1, 1, 0, 0])  # Perfectly correlated with feature 1
        
        # Create a custom model class with uneven feature importances
        class UnevenImportanceModel:
            def __init__(self):
                self.feature_importances_ = np.array([0.98, 0.01, 0.01])
        
        model = UnevenImportanceModel()
        feature_names = ["feature1", "feature2", "feature3"]
        
        importance_df = get_feature_importance(model, feature_names)
        
        # Check that the DataFrame has the correct structure
        assert isinstance(importance_df, pd.DataFrame)
        assert list(importance_df.columns) == ["feature", "importance"]
        assert len(importance_df) == 3
        
        # Check that features are sorted by importance in descending order
        assert importance_df.iloc[0]["feature"] == "feature1"
        assert importance_df.iloc[0]["importance"] == 0.98
        
        # Check that the remaining features have lower importance
        assert importance_df.iloc[1]["importance"] < importance_df.iloc[0]["importance"]
        assert importance_df.iloc[2]["importance"] < importance_df.iloc[0]["importance"]
