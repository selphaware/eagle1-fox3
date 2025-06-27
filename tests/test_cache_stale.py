"""Unit tests for the is_cache_stale function in cache module."""

__author__ = "Usman Ahmad"

import os
import pickle
import shutil
import pytest
from datetime import datetime
import pandas as pd
from unittest.mock import patch, mock_open
from typing import TYPE_CHECKING

from data.cache import save_to_cache, is_cache_stale

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def cleanup_cache() -> None:
    """Fixture to clean up cache directory after tests."""
    # Setup: ensure any existing cache directory is removed
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    
    # Let the test run
    yield
    
    # Teardown: clean up after test
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)


class TestIsCacheStale:
    """Tests for the is_cache_stale function."""
    
    def test_is_cache_stale_with_invalid_key(self) -> None:
        """Test that is_cache_stale raises ValueError for invalid keys."""
        with pytest.raises(ValueError, match="Key must be a non-empty string"):
            is_cache_stale("", 60)
        
        with pytest.raises(ValueError, match="Key must be a non-empty string"):
            is_cache_stale("   ", 60)
        
        with pytest.raises(ValueError, match="Key must be a non-empty string"):
            is_cache_stale(123, 60)  # type: ignore
    
    def test_is_cache_stale_with_invalid_minutes(self) -> None:
        """Test that is_cache_stale raises ValueError for invalid minutes."""
        with pytest.raises(ValueError, match="Minutes must be a positive integer"):
            is_cache_stale("test_key", 0)
        
        with pytest.raises(ValueError, match="Minutes must be a positive integer"):
            is_cache_stale("test_key", -10)
        
        with pytest.raises(ValueError, match="Minutes must be a positive integer"):
            is_cache_stale("test_key", "60")  # type: ignore
    
    def test_is_cache_stale_with_nonexistent_key(self, cleanup_cache: None) -> None:
        """Test that is_cache_stale returns True for non-existent keys."""
        # Test with a key that doesn't exist
        result = is_cache_stale("nonexistent_key", 60)
        assert result is True
    
    def test_is_cache_stale_with_fresh_cache(self, cleanup_cache: None) -> None:
        """Test that is_cache_stale returns False for fresh cache."""
        # First save some data to cache
        key = "test_fresh_key"
        df = pd.DataFrame({'A': [1, 2, 3]})
        save_to_cache(key, df)
        
        # Then check if it's stale (it shouldn't be)
        result = is_cache_stale(key, 60)
        
        # Verify result
        assert result is False
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    @patch('datetime.datetime')
    def test_is_cache_stale_with_stale_cache(
        self, mock_datetime: pytest.MonkeyPatch, mock_load: pytest.MonkeyPatch, 
        mock_file: pytest.MonkeyPatch, mock_exists: pytest.MonkeyPatch
    ) -> None:
        """Test that is_cache_stale returns True for stale cache."""
        # Setup mocks
        mock_exists.return_value = True
        
        # Create a timestamp that's older than our threshold
        current_time = datetime(2025, 6, 27, 12, 0, 0)
        old_time = datetime(2025, 6, 27, 10, 0, 0)  # 2 hours old
        
        mock_datetime.now.return_value = current_time
        mock_load.return_value = {
            "timestamp": old_time,
            "data": pd.DataFrame()
        }
        
        # Test with 60 minutes threshold (our cache is 120 minutes old)
        result = is_cache_stale("stale_key", 60)
        
        # Verify result
        assert result is True
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    def test_is_cache_stale_with_corrupted_structure(
        self, mock_load: pytest.MonkeyPatch, mock_file: pytest.MonkeyPatch, mock_exists: pytest.MonkeyPatch
    ) -> None:
        """Test that is_cache_stale handles corrupted cache structure."""
        # Setup mocks
        mock_exists.return_value = True
        mock_load.return_value = {"wrong_key": "wrong_value"}  # Missing timestamp
        
        # Test
        result = is_cache_stale("corrupted_key", 60)
        
        # Verify result
        assert result is True
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    def test_is_cache_stale_with_invalid_timestamp_type(
        self, mock_load: pytest.MonkeyPatch, mock_file: pytest.MonkeyPatch, mock_exists: pytest.MonkeyPatch
    ) -> None:
        """Test that is_cache_stale handles invalid timestamp type."""
        # Setup mocks
        mock_exists.return_value = True
        mock_load.return_value = {
            "timestamp": "2025-06-27",  # String instead of datetime
            "data": pd.DataFrame()
        }
        
        # Test
        result = is_cache_stale("invalid_timestamp_key", 60)
        
        # Verify result
        assert result is True
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    def test_is_cache_stale_with_unpickling_error(
        self, mock_load: pytest.MonkeyPatch, mock_file: pytest.MonkeyPatch, mock_exists: pytest.MonkeyPatch
    ) -> None:
        """Test that is_cache_stale handles pickle unpickling errors."""
        # Setup mocks
        mock_exists.return_value = True
        mock_load.side_effect = pickle.UnpicklingError("Corrupted pickle file")
        
        # Test
        result = is_cache_stale("unpickling_error_key", 60)
        
        # Verify result
        assert result is True
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_is_cache_stale_with_general_exception(
        self, mock_file: pytest.MonkeyPatch, mock_exists: pytest.MonkeyPatch
    ) -> None:
        """Test that is_cache_stale handles general exceptions."""
        # Setup mocks
        mock_exists.return_value = True
        mock_file.side_effect = Exception("Unexpected error")
        
        # Test
        result = is_cache_stale("exception_key", 60)
        
        # Verify result
        assert result is True
