"""
Unit tests for the ML data splitting module.
"""

__author__ = "Usman Ahmad"

import pytest
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

from ml.base import split_data

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestSplitData:
    """Tests for the split_data function."""
    
    def test_split_data_with_invalid_df_type(self) -> None:
        """Test that split_data raises TypeError for invalid DataFrame type."""
        with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
            split_data([1, 2, 3], "target")  # type: ignore
        
        with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
            split_data({"a": 1, "b": 2}, "target")  # type: ignore
    
    def test_split_data_with_invalid_target_type(self) -> None:
        """Test that split_data raises TypeError for invalid target type."""
        df = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
        
        with pytest.raises(TypeError, match="target must be a string"):
            split_data(df, 123)  # type: ignore
    
    def test_split_data_with_missing_target_column(self) -> None:
        """Test that split_data raises ValueError when target column is not in DataFrame."""
        df = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
        
        with pytest.raises(ValueError, match="Target column 'missing_column' not found"):
            split_data(df, "missing_column")
    
    def test_split_data_with_invalid_test_size(self) -> None:
        """Test that split_data raises ValueError for invalid test_size."""
        df = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
        
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            split_data(df, "target", test_size=0)
        
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            split_data(df, "target", test_size=1)
        
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            split_data(df, "target", test_size=-0.5)
        
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            split_data(df, "target", test_size=1.5)
    
    def test_split_data_with_valid_inputs(self) -> None:
        """Test that split_data correctly splits data with default parameters."""
        # Create test DataFrame
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        # Call function with default parameters
        X_train, X_test, y_train, y_test = split_data(df, "target")
        
        # Verify split sizes (default test_size=0.2)
        assert len(X_train) == 8  # 80% of 10 = 8
        assert len(X_test) == 2   # 20% of 10 = 2
        assert len(y_train) == 8
        assert len(y_test) == 2
        
        # Verify column structure
        assert "target" not in X_train.columns
        assert "target" not in X_test.columns
        assert "feature1" in X_train.columns
        assert "feature2" in X_train.columns
        assert "feature1" in X_test.columns
        assert "feature2" in X_test.columns
        
        # Verify that y contains only the target column values
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
    
    def test_split_data_with_different_test_sizes(self) -> None:
        """Test that split_data correctly handles different test sizes."""
        # Create test DataFrame
        df = pd.DataFrame({
            "feature": list(range(100)),
            "target": [i % 2 for i in range(100)]  # Alternating 0s and 1s
        })
        
        # Test with test_size=0.3
        X_train_30, X_test_30, y_train_30, y_test_30 = split_data(df, "target", test_size=0.3)
        assert len(X_train_30) == 70  # 70% of 100 = 70
        assert len(X_test_30) == 30   # 30% of 100 = 30
        
        # Test with test_size=0.5
        X_train_50, X_test_50, y_train_50, y_test_50 = split_data(df, "target", test_size=0.5)
        assert len(X_train_50) == 50  # 50% of 100 = 50
        assert len(X_test_50) == 50   # 50% of 100 = 50
        
        # Test with test_size=0.8
        X_train_80, X_test_80, y_train_80, y_test_80 = split_data(df, "target", test_size=0.8)
        assert len(X_train_80) == 20  # 20% of 100 = 20
        assert len(X_test_80) == 80   # 80% of 100 = 80
    
    def test_split_data_with_stratification(self) -> None:
        """Test that split_data correctly stratifies based on target column."""
        # Create imbalanced test DataFrame (80% class 0, 20% class 1)
        df = pd.DataFrame({
            "feature": list(range(100)),
            "target": [0] * 80 + [1] * 20
        })
        
        # Split with stratification
        X_train, X_test, y_train, y_test = split_data(
            df, "target", test_size=0.25, stratify=True
        )
        
        # Verify split sizes
        assert len(X_train) == 75  # 75% of 100 = 75
        assert len(X_test) == 25   # 25% of 100 = 25
        
        # Verify stratification - class distribution should be preserved
        train_class_0_ratio = (y_train == 0).sum() / len(y_train)
        test_class_0_ratio = (y_test == 0).sum() / len(y_test)
        
        # Both should be close to 0.8 (original distribution)
        assert abs(train_class_0_ratio - 0.8) < 0.05
        assert abs(test_class_0_ratio - 0.8) < 0.05
    
    def test_split_data_with_random_state(self) -> None:
        """Test that split_data produces reproducible results with fixed random_state."""
        # Create test DataFrame
        df = pd.DataFrame({
            "feature": list(range(100)),
            "target": [i % 2 for i in range(100)]
        })
        
        # Split with fixed random_state
        X_train1, X_test1, y_train1, y_test1 = split_data(
            df, "target", random_state=42
        )
        
        # Split again with same random_state
        X_train2, X_test2, y_train2, y_test2 = split_data(
            df, "target", random_state=42
        )
        
        # Verify that splits are identical
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_test1, y_test2)
        
        # Split with different random_state
        X_train3, X_test3, y_train3, y_test3 = split_data(
            df, "target", random_state=24
        )
        
        # Verify that splits are different
        assert not X_train1.equals(X_train3)
        assert not X_test1.equals(X_test3)
    
    def test_split_data_with_single_class_stratification(self) -> None:
        """Test that split_data falls back to random split when stratification is not possible."""
        # Create DataFrame with only one class
        df = pd.DataFrame({
            "feature": list(range(10)),
            "target": [0] * 10  # Only class 0
        })
        
        # Split with stratification
        X_train, X_test, y_train, y_test = split_data(
            df, "target", test_size=0.3, stratify=True
        )
        
        # Verify split sizes
        assert len(X_train) == 7  # 70% of 10 = 7
        assert len(X_test) == 3   # 30% of 10 = 3
        
        # Verify all are class 0
        assert (y_train == 0).all()
        assert (y_test == 0).all()
