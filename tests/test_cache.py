"""Unit tests for the cache module."""

__author__ = "Usman Ahmad"

import os
import pickle
import shutil
import pytest
from datetime import datetime
import pandas as pd
from unittest.mock import patch, mock_open
from typing import TYPE_CHECKING

from data.cache import save_to_cache

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


class TestSaveToCache:
    """Tests for the save_to_cache function."""
    
    def test_save_to_cache_with_valid_inputs(self, cleanup_cache: None) -> None:
        """Test that save_to_cache works with valid inputs."""
        # Create test data
        key = "test_key"
        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        
        # Call function
        save_to_cache(key, df)
        
        # Verify cache directory was created
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
        assert os.path.exists(cache_dir)
        
        # Verify file was created
        cache_path = os.path.join(cache_dir, f"{key}.pkl")
        assert os.path.exists(cache_path)
        
        # Verify file contents
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        assert 'timestamp' in cache_data
        assert isinstance(cache_data['timestamp'], datetime)
        assert 'data' in cache_data
        pd.testing.assert_frame_equal(cache_data['data'], df)
    
    def test_save_to_cache_with_special_characters_in_key(self, cleanup_cache: None) -> None:
        """Test that save_to_cache sanitizes keys with special characters."""
        # Create test data
        key = "test/key:with*special?chars"
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        # Call function
        save_to_cache(key, df)
        
        # Verify file was created with sanitized name
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
        sanitized_key = "test_key_with_special_chars"
        cache_path = os.path.join(cache_dir, f"{sanitized_key}.pkl")
        assert os.path.exists(cache_path)
    
    def test_save_to_cache_with_empty_key(self) -> None:
        """Test that save_to_cache raises ValueError for empty key."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Key must be a non-empty string"):
            save_to_cache("", df)
        
        with pytest.raises(ValueError, match="Key must be a non-empty string"):
            save_to_cache("   ", df)
    
    def test_save_to_cache_with_invalid_key_type(self) -> None:
        """Test that save_to_cache raises ValueError for non-string key."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Key must be a non-empty string"):
            save_to_cache(123, df)  # type: ignore
    
    def test_save_to_cache_with_invalid_df_type(self) -> None:
        """Test that save_to_cache raises TypeError for non-DataFrame data."""
        with pytest.raises(TypeError, match="Data must be a pandas DataFrame"):
            save_to_cache("test_key", [1, 2, 3])  # type: ignore
        
        with pytest.raises(TypeError, match="Data must be a pandas DataFrame"):
            save_to_cache("test_key", {"a": 1, "b": 2})  # type: ignore
    
    @patch('data.cache.open', new_callable=mock_open)
    @patch('data.cache.pickle.dump')
    def test_save_to_cache_with_file_operation_error(
        self, mock_dump: pytest.MonkeyPatch, mock_file: pytest.MonkeyPatch
    ) -> None:
        """Test that save_to_cache handles file operation errors."""
        # Setup
        mock_dump.side_effect = IOError("Disk full")
        
        # Test data
        key = "test_key"
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        # Test
        with pytest.raises(OSError, match="Failed to save data to cache"):
            save_to_cache(key, df)
