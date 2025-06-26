"""Tests for the fetch_data module."""

from typing import TYPE_CHECKING, Any, Dict
import pytest
from unittest.mock import patch, MagicMock

# Import the function, not the module, to avoid actual API calls during import
with patch('yfinance.Ticker'):
    from data.fetch_data import validate_ticker

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestValidateTicker:
    """Tests for the validate_ticker function."""

    def test_validate_ticker_with_invalid_input(self) -> None:
        """Test that validate_ticker returns False for invalid inputs."""
        # Test with empty string
        assert validate_ticker("") is False
        
        # Test with whitespace string
        assert validate_ticker("   ") is False

    @patch('yfinance.Ticker')
    def test_validate_ticker_with_valid_ticker(self, mock_ticker: MagicMock) -> None:
        """Test that validate_ticker returns True for valid tickers."""
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {'regularMarketPrice': 150.25}
        mock_ticker.return_value = mock_ticker_instance
        
        # Test
        result = validate_ticker('AAPL')
        
        # Assertions
        assert result is True
        mock_ticker.assert_called_once_with('AAPL')

    @patch('yfinance.Ticker')
    def test_validate_ticker_with_invalid_ticker(self, mock_ticker: MagicMock) -> None:
        """Test that validate_ticker returns False for invalid tickers."""
        # Setup mock for invalid ticker (missing regularMarketPrice)
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {'symbol': 'INVALID'}
        mock_ticker.return_value = mock_ticker_instance
        
        # Test
        result = validate_ticker('INVALID')
        
        # Assertions
        assert result is False
        mock_ticker.assert_called_once_with('INVALID')

    @patch('yfinance.Ticker')
    def test_validate_ticker_with_exception(self, mock_ticker: MagicMock) -> None:
        """Test that validate_ticker handles exceptions gracefully."""
        # Setup mock to raise exception
        mock_ticker.side_effect = Exception("API Error")
        
        # Test
        result = validate_ticker('AAPL')
        
        # Assertions
        assert result is False
        mock_ticker.assert_called_once_with('AAPL')
