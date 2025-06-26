"""Tests for the fetch_data module."""

__author__ = "Usman Ahmad"

from typing import TYPE_CHECKING, Any, Dict
import pytest
from unittest.mock import patch, MagicMock

# Import the functions, not the module, to avoid actual API calls during import
with patch('yfinance.Ticker'):
    from data.fetch_data import validate_ticker, get_financials, get_13f_holdings, get_mutual_fund_holdings

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


class TestGetFinancials:
    """Tests for the get_financials function."""
    
    def test_get_financials_with_invalid_input(self) -> None:
        """Test that get_financials raises TypeError for non-string inputs."""
        with pytest.raises(TypeError):
            get_financials(123)  # type: ignore
    
    @patch('data.fetch_data.validate_ticker')
    def test_get_financials_with_invalid_ticker(self, mock_validate: MagicMock) -> None:
        """Test that get_financials returns empty DataFrame for invalid tickers."""
        # Setup mock
        mock_validate.return_value = False
        
        # Test
        result = get_financials('INVALID')
        
        # Assertions
        assert result.empty
        mock_validate.assert_called_once_with('INVALID')
    
    @patch('yfinance.Ticker')
    @patch('data.fetch_data.validate_ticker')
    def test_get_financials_with_valid_ticker(self, mock_validate: MagicMock, mock_ticker: MagicMock) -> None:
        """Test that get_financials returns financial data for valid tickers."""
        import pandas as pd
        import numpy as np
        
        # Setup mocks
        mock_validate.return_value = True
        
        # Create mock financial data
        mock_income_stmt = pd.DataFrame({
            'Revenue': [100, 120],
            'NetIncome': [10, 15]
        }, index=[pd.Timestamp('2022-12-31'), pd.Timestamp('2021-12-31')])
        
        mock_balance_sheet = pd.DataFrame({
            'TotalAssets': [500, 450],
            'TotalLiabilities': [300, 280]
        }, index=[pd.Timestamp('2022-12-31'), pd.Timestamp('2021-12-31')])
        
        mock_cash_flow = pd.DataFrame({
            'OperatingCashFlow': [50, 45],
            'InvestingCashFlow': [-30, -25]
        }, index=[pd.Timestamp('2022-12-31'), pd.Timestamp('2021-12-31')])
        
        # Setup mock ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.income_stmt = mock_income_stmt
        mock_ticker_instance.balance_sheet = mock_balance_sheet
        mock_ticker_instance.cashflow = mock_cash_flow
        mock_ticker.return_value = mock_ticker_instance
        
        # Test
        result = get_financials('AAPL')
        
        # Assertions
        assert not result.empty
        assert 'income_statement' in result.columns.levels[0]
        assert 'balance_sheet' in result.columns.levels[0]
        assert 'cash_flow' in result.columns.levels[0]
        mock_validate.assert_called_once_with('AAPL')
        mock_ticker.assert_called_once_with('AAPL')
    
    @patch('yfinance.Ticker')
    @patch('data.fetch_data.validate_ticker')
    def test_get_financials_with_empty_response(self, mock_validate: MagicMock, mock_ticker: MagicMock) -> None:
        """Test that get_financials handles empty responses gracefully."""
        import pandas as pd
        
        # Setup mocks
        mock_validate.return_value = True
        
        # Create empty DataFrames
        empty_df = pd.DataFrame()
        
        # Setup mock ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.income_stmt = empty_df
        mock_ticker_instance.balance_sheet = empty_df
        mock_ticker_instance.cashflow = empty_df
        mock_ticker.return_value = mock_ticker_instance
        
        # Test
        result = get_financials('AAPL')
        
        # Assertions
        assert result.empty
        mock_validate.assert_called_once_with('AAPL')
        mock_ticker.assert_called_once_with('AAPL')
    
    @patch('yfinance.Ticker')
    @patch('data.fetch_data.validate_ticker')
    def test_get_financials_with_exception(self, mock_validate: MagicMock, mock_ticker: MagicMock) -> None:
        """Test that get_financials handles exceptions gracefully."""
        # Setup mocks
        mock_validate.return_value = True
        mock_ticker.side_effect = Exception("API Error")
        
        # Test
        result = get_financials('AAPL')
        
        # Assertions
        assert result.empty
        mock_validate.assert_called_once_with('AAPL')
        mock_ticker.assert_called_once_with('AAPL')


class TestGet13FHoldings:
    """Tests for the get_13f_holdings function."""
    
    def test_get_13f_holdings_with_invalid_input(self) -> None:
        """Test that get_13f_holdings raises TypeError for non-string inputs."""
        with pytest.raises(TypeError):
            get_13f_holdings(123)  # type: ignore
    
    @patch('data.fetch_data.validate_ticker')
    def test_get_13f_holdings_with_invalid_ticker(self, mock_validate: MagicMock) -> None:
        """Test that get_13f_holdings returns empty DataFrame for invalid tickers."""
        # Setup mock
        mock_validate.return_value = False
        
        # Test
        result = get_13f_holdings('INVALID')
        
        # Assertions
        assert result.empty
        mock_validate.assert_called_once_with('INVALID')
    
    @patch('yfinance.Ticker')
    @patch('data.fetch_data.validate_ticker')
    def test_get_13f_holdings_with_valid_ticker(self, mock_validate: MagicMock, mock_ticker: MagicMock) -> None:
        """Test that get_13f_holdings returns institutional holders data for valid tickers."""
        import pandas as pd
        
        # Setup mocks
        mock_validate.return_value = True
        
        # Create mock institutional holders data
        mock_inst_holders = pd.DataFrame({
            'Holder': ['BlackRock Inc', 'Vanguard Group Inc'],
            'Shares': [1000000, 950000],
            'Date Reported': ['2022-12-31', '2022-12-31'],
            'Value': [150000000, 142500000],
            '% Out': [0.0625, 0.0594]
        })
        
        # Create mock major holders data
        mock_major_holders = pd.DataFrame({
            0: ['% of Shares Held by All Insider', '% of Shares Held by Institutions'],
            1: ['0.06%', '60.95%']
        })
        
        # Setup mock ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.institutional_holders = mock_inst_holders
        mock_ticker_instance.major_holders = mock_major_holders
        mock_ticker.return_value = mock_ticker_instance
        
        # Test
        result = get_13f_holdings('AAPL')
        
        # Assertions
        assert not result.empty
        assert 'Holder' in result.columns
        assert 'Shares' in result.columns
        assert 'Ticker' in result.columns  # We add this column in our implementation
        mock_validate.assert_called_once_with('AAPL')
        mock_ticker.assert_called_once_with('AAPL')
    
    @patch('yfinance.Ticker')
    @patch('data.fetch_data.validate_ticker')
    def test_get_13f_holdings_with_only_major_holders(self, mock_validate: MagicMock, mock_ticker: MagicMock) -> None:
        """Test that get_13f_holdings can use major_holders when institutional_holders is empty."""
        import pandas as pd
        
        # Setup mocks
        mock_validate.return_value = True
        
        # Create empty institutional holders data
        mock_inst_holders = pd.DataFrame()
        
        # Create mock major holders data
        mock_major_holders = pd.DataFrame({
            0: ['% of Shares Held by All Insider', '% of Shares Held by Institutions'],
            1: ['0.06%', '60.95%']
        })
        
        # Setup mock ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.institutional_holders = mock_inst_holders
        mock_ticker_instance.major_holders = mock_major_holders
        mock_ticker.return_value = mock_ticker_instance
        
        # Test
        result = get_13f_holdings('AAPL')
        
        # Assertions
        assert not result.empty
        assert 'Holder Type' in result.columns
        assert 'Percentage' in result.columns
        assert 'Ticker' in result.columns
        mock_validate.assert_called_once_with('AAPL')
        mock_ticker.assert_called_once_with('AAPL')
    
    @patch('yfinance.Ticker')
    @patch('data.fetch_data.validate_ticker')
    def test_get_13f_holdings_with_empty_response(self, mock_validate: MagicMock, mock_ticker: MagicMock) -> None:
        """Test that get_13f_holdings handles empty responses gracefully."""
        import pandas as pd
        
        # Setup mocks
        mock_validate.return_value = True
        
        # Create empty DataFrames
        empty_df = pd.DataFrame()
        
        # Setup mock ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.institutional_holders = empty_df
        mock_ticker_instance.major_holders = empty_df
        mock_ticker.return_value = mock_ticker_instance
        
        # Test
        result = get_13f_holdings('AAPL')
        
        # Assertions
        assert result.empty
        mock_validate.assert_called_once_with('AAPL')
        mock_ticker.assert_called_once_with('AAPL')
    
    @patch('yfinance.Ticker')
    @patch('data.fetch_data.validate_ticker')
    def test_get_13f_holdings_with_exception(self, mock_validate: MagicMock, mock_ticker: MagicMock) -> None:
        """Test that get_13f_holdings handles exceptions gracefully."""
        # Setup mocks
        mock_validate.return_value = True
        mock_ticker.side_effect = Exception("API Error")
        
        # Test
        result = get_13f_holdings('AAPL')
        
        # Assertions
        assert result.empty
        mock_validate.assert_called_once_with('AAPL')
        mock_ticker.assert_called_once_with('AAPL')


class TestGetMutualFundHoldings:
    """Tests for the get_mutual_fund_holdings function."""
    
    def test_get_mutual_fund_holdings_with_invalid_input(self) -> None:
        """Test that get_mutual_fund_holdings raises TypeError for non-string inputs."""
        with pytest.raises(TypeError):
            get_mutual_fund_holdings(123)  # type: ignore
    
    @patch('data.fetch_data.validate_ticker')
    def test_get_mutual_fund_holdings_with_invalid_ticker(self, mock_validate: MagicMock) -> None:
        """Test that get_mutual_fund_holdings returns empty DataFrame for invalid tickers."""
        # Setup mock
        mock_validate.return_value = False
        
        # Test
        result = get_mutual_fund_holdings('INVALID')
        
        # Assertions
        assert result.empty
        mock_validate.assert_called_once_with('INVALID')
    
    @patch('yfinance.Ticker')
    @patch('data.fetch_data.validate_ticker')
    def test_get_mutual_fund_holdings_with_valid_ticker(self, mock_validate: MagicMock, mock_ticker: MagicMock) -> None:
        """Test that get_mutual_fund_holdings returns mutual fund holders data for valid tickers."""
        import pandas as pd
        
        # Setup mocks
        mock_validate.return_value = True
        
        # Create mock mutual fund holders data
        mock_mf_holders = pd.DataFrame({
            'Holder': ['Vanguard Total Stock Market Index Fund', 'Vanguard 500 Index Fund'],
            'Shares': [120000000, 110000000],
            'Date Reported': ['2022-12-31', '2022-12-31'],
            'Value': [18000000000, 16500000000],
            '% Out': [0.0755, 0.0692]
        })
        
        # Setup mock ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.mutualfund_holders = mock_mf_holders
        mock_ticker.return_value = mock_ticker_instance
        
        # Test
        result = get_mutual_fund_holdings('AAPL')
        
        # Assertions
        assert not result.empty
        assert 'Holder' in result.columns
        assert 'Shares' in result.columns
        assert 'Ticker' in result.columns  # We add this column in our implementation
        mock_validate.assert_called_once_with('AAPL')
        mock_ticker.assert_called_once_with('AAPL')
    
    @patch('yfinance.Ticker')
    @patch('data.fetch_data.validate_ticker')
    def test_get_mutual_fund_holdings_with_empty_response(self, mock_validate: MagicMock, mock_ticker: MagicMock) -> None:
        """Test that get_mutual_fund_holdings handles empty responses gracefully."""
        import pandas as pd
        
        # Setup mocks
        mock_validate.return_value = True
        
        # Create empty DataFrame
        empty_df = pd.DataFrame()
        
        # Setup mock ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.mutualfund_holders = empty_df
        mock_ticker.return_value = mock_ticker_instance
        
        # Test
        result = get_mutual_fund_holdings('AAPL')
        
        # Assertions
        assert result.empty
        mock_validate.assert_called_once_with('AAPL')
        mock_ticker.assert_called_once_with('AAPL')
    
    @patch('yfinance.Ticker')
    @patch('data.fetch_data.validate_ticker')
    def test_get_mutual_fund_holdings_with_exception(self, mock_validate: MagicMock, mock_ticker: MagicMock) -> None:
        """Test that get_mutual_fund_holdings handles exceptions gracefully."""
        # Setup mocks
        mock_validate.return_value = True
        mock_ticker.side_effect = Exception("API Error")
        
        # Test
        result = get_mutual_fund_holdings('AAPL')
        
        # Assertions
        assert result.empty
        mock_validate.assert_called_once_with('AAPL')
        mock_ticker.assert_called_once_with('AAPL')
