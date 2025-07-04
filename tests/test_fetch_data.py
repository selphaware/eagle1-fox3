"""Tests for the fetch_data module."""

__author__ = "Usman Ahmad"

from typing import TYPE_CHECKING, Any, Dict
import pytest
from unittest.mock import patch, MagicMock

# Import the functions, not the module, to avoid actual API calls during import
with patch('yfinance.Ticker'):
    from data.fetch_data import validate_ticker, get_financials, get_13f_holdings, get_mutual_fund_holdings, get_corporate_actions, retry_api_call

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


class TestGetCorporateActions:
    """Tests for the get_corporate_actions function."""
    
    def test_get_corporate_actions_with_invalid_input(self) -> None:
        """Test that get_corporate_actions raises TypeError for non-string inputs."""
        with pytest.raises(TypeError):
            get_corporate_actions(123)  # type: ignore
    
    @patch('data.fetch_data.validate_ticker')
    def test_get_corporate_actions_with_invalid_ticker(self, mock_validate: MagicMock) -> None:
        """Test that get_corporate_actions returns empty DataFrame for invalid tickers."""
        # Setup mock
        mock_validate.return_value = False
        
        # Test
        result = get_corporate_actions('INVALID')
        
        # Assertions
        assert result.empty
        mock_validate.assert_called_once_with('INVALID')
    
    @patch('yfinance.Ticker')
    @patch('data.fetch_data.validate_ticker')
    def test_get_corporate_actions_with_valid_ticker_dividends_only(self, mock_validate: MagicMock, mock_ticker: MagicMock) -> None:
        """Test that get_corporate_actions returns dividend data when only dividends are available."""
        import pandas as pd
        import numpy as np
        
        # Setup mocks
        mock_validate.return_value = True
        
        # Create mock dividends Series
        dates = pd.DatetimeIndex(['2022-12-01', '2022-09-01', '2022-06-01', '2022-03-01'])
        dividends = pd.Series([0.23, 0.23, 0.23, 0.22], index=dates)
        
        # Create empty splits Series
        splits = pd.Series(dtype='float64')
        
        # Setup mock ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.dividends = dividends
        mock_ticker_instance.splits = splits
        mock_ticker.return_value = mock_ticker_instance
        
        # Test
        result = get_corporate_actions('AAPL')
        
        # Assertions
        assert not result.empty
        assert 'Date' in result.columns
        assert 'Dividend' in result.columns
        assert 'Action' in result.columns
        assert 'Ticker' in result.columns
        assert len(result) == 4  # 4 dividend entries
        assert (result['Action'] == 'Dividend').all()
        mock_validate.assert_called_once_with('AAPL')
        mock_ticker.assert_called_once_with('AAPL')
    
    @patch('yfinance.Ticker')
    @patch('data.fetch_data.validate_ticker')
    def test_get_corporate_actions_with_valid_ticker_splits_only(self, mock_validate: MagicMock, mock_ticker: MagicMock) -> None:
        """Test that get_corporate_actions returns split data when only splits are available."""
        import pandas as pd
        
        # Setup mocks
        mock_validate.return_value = True
        
        # Create empty dividends Series
        dividends = pd.Series(dtype='float64')
        
        # Create mock splits Series
        dates = pd.DatetimeIndex(['2020-08-31', '2014-06-09', '2005-02-28'])
        splits = pd.Series([4.0, 7.0, 2.0], index=dates)  # 4:1, 7:1, 2:1 splits
        
        # Setup mock ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.dividends = dividends
        mock_ticker_instance.splits = splits
        mock_ticker.return_value = mock_ticker_instance
        
        # Test
        result = get_corporate_actions('AAPL')
        
        # Assertions
        assert not result.empty
        assert 'Date' in result.columns
        assert 'Split Ratio' in result.columns
        assert 'Action' in result.columns
        assert 'Ticker' in result.columns
        assert len(result) == 3  # 3 split entries
        assert (result['Action'] == 'Stock Split').all()
        mock_validate.assert_called_once_with('AAPL')
        mock_ticker.assert_called_once_with('AAPL')
    
    @patch('yfinance.Ticker')
    @patch('data.fetch_data.validate_ticker')
    def test_get_corporate_actions_with_valid_ticker_both_actions(self, mock_validate: MagicMock, mock_ticker: MagicMock) -> None:
        """Test that get_corporate_actions returns both dividend and split data when both are available."""
        import pandas as pd
        
        # Setup mocks
        mock_validate.return_value = True
        
        # Create mock dividends Series
        div_dates = pd.DatetimeIndex(['2022-12-01', '2022-09-01'])
        dividends = pd.Series([0.23, 0.23], index=div_dates)
        
        # Create mock splits Series
        split_dates = pd.DatetimeIndex(['2020-08-31', '2014-06-09'])
        splits = pd.Series([4.0, 7.0], index=split_dates)
        
        # Setup mock ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.dividends = dividends
        mock_ticker_instance.splits = splits
        mock_ticker.return_value = mock_ticker_instance
        
        # Test
        result = get_corporate_actions('AAPL')
        
        # Assertions
        assert not result.empty
        assert 'Date' in result.columns
        assert 'Action' in result.columns
        assert 'Ticker' in result.columns
        assert len(result) == 4  # 2 dividend + 2 split entries
        assert sum(result['Action'] == 'Dividend') == 2
        assert sum(result['Action'] == 'Stock Split') == 2
        mock_validate.assert_called_once_with('AAPL')
        mock_ticker.assert_called_once_with('AAPL')
    
    @patch('yfinance.Ticker')
    @patch('data.fetch_data.validate_ticker')
    def test_get_corporate_actions_with_empty_response(self, mock_validate: MagicMock, mock_ticker: MagicMock) -> None:
        """Test that get_corporate_actions handles empty responses gracefully."""
        import pandas as pd
        
        # Setup mocks
        mock_validate.return_value = True
        
        # Create empty Series
        empty_series = pd.Series(dtype='float64')
        
        # Setup mock ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.dividends = empty_series
        mock_ticker_instance.splits = empty_series
        mock_ticker.return_value = mock_ticker_instance
        
        # Test
        result = get_corporate_actions('AAPL')
        
        # Assertions
        assert result.empty
        mock_validate.assert_called_once_with('AAPL')
        mock_ticker.assert_called_once_with('AAPL')
    
    @patch('yfinance.Ticker')
    @patch('data.fetch_data.validate_ticker')
    def test_get_corporate_actions_with_exception(self, mock_validate: MagicMock, mock_ticker: MagicMock) -> None:
        """Test that get_corporate_actions handles exceptions gracefully."""
        # Setup mocks
        mock_validate.return_value = True
        mock_ticker.side_effect = Exception("API Error")
        
        # Test
        result = get_corporate_actions('AAPL')
        
        # Assertions
        assert result.empty
        mock_validate.assert_called_once_with('AAPL')
        mock_ticker.assert_called_once_with('AAPL')


class TestRetryApiCall:
    """Tests for the retry_api_call decorator."""
    
    def test_successful_execution(self) -> None:
        """Test that the decorator works with successful function execution."""
        # Define a test function
        @retry_api_call
        def test_func(value: str) -> str:
            return f"Success: {value}"
        
        # Test
        result = test_func("test")
        
        # Assertions
        assert result == "Success: test"
    
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_with_retries_needed(self, mock_sleep: MagicMock) -> None:
        """Test that the decorator retries when exceptions are raised."""
        # Setup counter for tracking calls
        call_count = 0
        
        # Define a test function that fails twice then succeeds
        @retry_api_call
        def test_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # Fail on first two calls
                raise ConnectionError("API Connection Error")
            return "Success after retries"
        
        # Test
        result = test_func()
        
        # Assertions
        assert result == "Success after retries"
        assert call_count == 3  # Function should be called 3 times (2 failures + 1 success)
        assert mock_sleep.call_count == 2  # Sleep should be called twice (after each failure)
        # Verify exponential backoff
        mock_sleep.assert_any_call(1)  # First retry: 1 second
        mock_sleep.assert_any_call(2)  # Second retry: 2 seconds
    
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_max_retries_exceeded(self, mock_sleep: MagicMock) -> None:
        """Test that the decorator raises the exception after max retries."""
        # Define a test function that always fails
        @retry_api_call
        def test_func() -> None:
            raise ValueError("API Error")
        
        # Test
        with pytest.raises(ValueError, match="API Error"):
            test_func()
        
        # Assertions
        # MAX_RETRIES is 3, so sleep should be called 3 times
        assert mock_sleep.call_count == 3
        # Verify exponential backoff
        mock_sleep.assert_any_call(1)  # First retry: 1 second
        mock_sleep.assert_any_call(2)  # Second retry: 2 seconds
        mock_sleep.assert_any_call(4)  # Third retry: 4 seconds
