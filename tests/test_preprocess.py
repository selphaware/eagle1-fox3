"""Unit tests for the preprocess module."""

__author__ = "Usman Ahmad"

import pytest
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

from data.preprocess import impute_missing

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestImputeMissing:
    """Tests for the impute_missing function."""
    
    def test_impute_missing_with_invalid_df_type(self) -> None:
        """Test that impute_missing raises TypeError for invalid DataFrame type."""
        with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
            impute_missing([1, 2, 3])  # type: ignore
        
        with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
            impute_missing({"a": 1, "b": 2})  # type: ignore
    
    def test_impute_missing_with_invalid_method(self) -> None:
        """Test that impute_missing raises ValueError for invalid method."""
        df = pd.DataFrame({'A': [1, 2, np.nan]})
        
        with pytest.raises(ValueError, match="Invalid method 'invalid'. Must be 'mean', 'median', or 'drop'"):
            impute_missing(df, "invalid")  # type: ignore
    
    def test_impute_missing_with_mean_method(self) -> None:
        """Test that impute_missing correctly imputes missing values using mean."""
        # Create test DataFrame with missing values
        df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0],
            'B': [5.0, np.nan, 7.0, 8.0]
        })
        
        # Expected result: mean of A = (1+2+4)/3 = 2.33, mean of B = (5+7+8)/3 = 6.67
        expected = pd.DataFrame({
            'A': [1.0, 2.0, 2.33333, 4.0],
            'B': [5.0, 6.66667, 7.0, 8.0]
        })
        
        # Call function with mean method
        result = impute_missing(df, "mean")
        
        # Verify result
        pd.testing.assert_frame_equal(result, expected, rtol=1e-5)
    
    def test_impute_missing_with_median_method(self) -> None:
        """Test that impute_missing correctly imputes missing values using median."""
        # Create test DataFrame with missing values
        df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 10.0],  # median = 2.0
            'B': [5.0, np.nan, 7.0, 8.0]    # median = 7.0
        })
        
        # Expected result: median of A = 2.0, median of B = 7.0
        expected = pd.DataFrame({
            'A': [1.0, 2.0, 2.0, 10.0],
            'B': [5.0, 7.0, 7.0, 8.0]
        })
        
        # Call function with median method
        result = impute_missing(df, "median")
        
        # Verify result
        pd.testing.assert_frame_equal(result, expected)
    
    def test_impute_missing_with_drop_method(self) -> None:
        """Test that impute_missing correctly drops rows with missing values."""
        # Create test DataFrame with missing values
        df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0],
            'B': [5.0, np.nan, 7.0, 8.0]
        })
        
        # Call function with drop method
        result = impute_missing(df, "drop")
        
        # Verify result - should only have rows without any NaN values
        assert len(result) == 2  # Two rows should remain (index 0 and 3)
        assert 0 in result.index  # Row with index 0 should remain (no NaN in this row)
        assert 3 in result.index  # Row with index 3 should remain (no NaN in this row)
        assert not result.isna().any().any()  # No NaN values should remain
    
    def test_impute_missing_with_no_missing_values(self) -> None:
        """Test that impute_missing returns original DataFrame when no missing values."""
        # Create test DataFrame without missing values
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0],
            'B': [5.0, 6.0, 7.0, 8.0]
        })
        
        # Call function
        result = impute_missing(df)
        
        # Verify result is same as input
        pd.testing.assert_frame_equal(result, df)
    
    def test_impute_missing_with_non_numeric_columns(self) -> None:
        """Test that impute_missing handles DataFrames with non-numeric columns."""
        # Create test DataFrame with mixed column types
        df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0],
            'B': ['a', 'b', np.nan, 'd']  # string column
        })
        
        # Expected result: only numeric columns imputed
        expected = pd.DataFrame({
            'A': [1.0, 2.0, 2.33333, 4.0],
            'B': ['a', 'b', np.nan, 'd']  # string column still has NaN
        })
        
        # Call function
        result = impute_missing(df, "mean")
        
        # Verify numeric column was imputed
        assert result['A'].isna().sum() == 0
        # Verify non-numeric column still has NaN
        assert result['B'].isna().sum() == 1
        
        # Check numeric values
        pd.testing.assert_series_equal(result['A'], expected['A'], rtol=1e-5)
    
    def test_impute_missing_with_no_numeric_columns(self) -> None:
        """Test that impute_missing raises ValueError when no numeric columns for imputation."""
        # Create test DataFrame with only non-numeric columns
        df = pd.DataFrame({
            'A': ['a', 'b', np.nan, 'd'],
            'B': ['e', 'f', np.nan, 'h']
        })
        
        # Test with mean method
        with pytest.raises(ValueError, match="No numeric columns found for imputation"):
            impute_missing(df, "mean")
        
        # Test with median method
        with pytest.raises(ValueError, match="No numeric columns found for imputation"):
            impute_missing(df, "median")
        
        # Drop method should still work
        result = impute_missing(df, "drop")
        # Only rows without NaN values should remain
        assert len(result) == 3  # Based on actual function behavior
        assert not result.isna().any().any()  # No NaN values should remain
