"""Module for fetching financial data from Yahoo Finance."""

from typing import Dict, List, Optional, Union
import logging
import pandas as pd
import yfinance as yf

# Configure logging
logger = logging.getLogger(__name__)


def validate_ticker(ticker: str) -> bool:
    """
    Validate if a ticker symbol exists in Yahoo Finance.
    
    Args:
        ticker: The ticker symbol to validate.
        
    Returns:
        bool: True if the ticker is valid, False otherwise.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        logger.error("Invalid ticker format: ticker must be a non-empty string")
        return False
    
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
        # Check if we got valid info back (if 'regularMarketPrice' exists, it's likely valid)
        if info and 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
            logger.info(f"Ticker {ticker} validated successfully")
            return True
        else:
            logger.warning(f"Ticker {ticker} not found or returned incomplete data")
            return False
    except Exception as e:
        logger.error(f"Error validating ticker {ticker}: {str(e)}")
        return False


def get_financials(ticker: str) -> pd.DataFrame:
    """
    Fetch company financials (income statement, balance sheet, cash flow).
    
    Args:
        ticker: The ticker symbol to fetch data for.
        
    Returns:
        DataFrame: Financial data for the specified ticker.
        Empty DataFrame if ticker is invalid or data cannot be fetched.
    
    Raises:
        TypeError: If ticker is not a string.
    """
    if not isinstance(ticker, str):
        logger.error("TypeError: ticker must be a string")
        raise TypeError("ticker must be a string")
    
    if not validate_ticker(ticker):
        logger.warning(f"Invalid ticker: {ticker}. Returning empty DataFrame")
        return pd.DataFrame()
    
    try:
        logger.info(f"Fetching financial data for {ticker}")
        ticker_obj = yf.Ticker(ticker)
        
        # Get income statement, balance sheet, and cash flow data
        income_stmt = ticker_obj.income_stmt
        balance_sheet = ticker_obj.balance_sheet
        cash_flow = ticker_obj.cashflow
        
        # Combine all financial data into a dictionary
        financial_data = {
            'income_statement': income_stmt,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow
        }
        
        # Check if we got any data
        if all(df.empty for df in financial_data.values()):
            logger.warning(f"No financial data available for {ticker}")
            return pd.DataFrame()
        
        # Create a combined DataFrame with multi-level columns
        # First level: statement type, Second level: original columns
        combined_data = pd.concat(
            {
                key: df for key, df in financial_data.items() if not df.empty
            },
            axis=1
        )
        
        logger.info(f"Successfully fetched financial data for {ticker}")
        return combined_data
    
    except Exception as e:
        logger.error(f"Error fetching financial data for {ticker}: {str(e)}")
        return pd.DataFrame()


def get_13f_holdings(ticker: str) -> pd.DataFrame:
    """
    Fetch 13F/13D filings (institutional investor holdings).
    
    Args:
        ticker: The ticker symbol to fetch data for.
        
    Returns:
        DataFrame: 13F/13D holdings data for the specified ticker.
    """
    # Placeholder implementation
    pass


def get_mutual_fund_holdings(ticker: str) -> pd.DataFrame:
    """
    Fetch mutual fund holdings.
    
    Args:
        ticker: The ticker symbol to fetch data for.
        
    Returns:
        DataFrame: Mutual fund holdings data for the specified ticker.
    """
    # Placeholder implementation
    pass


def get_corporate_actions(ticker: str) -> pd.DataFrame:
    """
    Fetch corporate actions (dividends, splits, etc.).
    
    Args:
        ticker: The ticker symbol to fetch data for.
        
    Returns:
        DataFrame: Corporate actions data for the specified ticker.
    """
    # Placeholder implementation
    pass


def retry_api_call(func):
    """
    Decorator to retry API calls with exponential backoff.
    
    Args:
        func: The function to decorate.
        
    Returns:
        The decorated function with retry logic.
    """
    # Placeholder implementation
    pass
