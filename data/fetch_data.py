"""Module for fetching financial data from Yahoo Finance."""

__author__ = "Usman Ahmad"

from typing import Any, Callable, Dict, List, Optional, Union
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
        logger.info(f"Fetching 13F holdings data for {ticker}")
        ticker_obj = yf.Ticker(ticker)
        
        # Get institutional holders (13F filers)
        institutional_holders = ticker_obj.institutional_holders
        
        # Get major holders (includes 13D filers)
        major_holders = ticker_obj.major_holders
        
        # Check if we got any data
        if institutional_holders is None or institutional_holders.empty:
            logger.warning(f"No institutional holders data available for {ticker}")
            institutional_holders = pd.DataFrame()
            
        if major_holders is None or major_holders.empty:
            logger.warning(f"No major holders data available for {ticker}")
            major_holders = pd.DataFrame()
        
        # If both are empty, return empty DataFrame
        if institutional_holders.empty and (major_holders is None or major_holders.empty):
            logger.warning(f"No 13F/13D holdings data available for {ticker}")
            return pd.DataFrame()
        
        # Process and return the institutional holders data
        # This contains the most detailed 13F information
        if not institutional_holders.empty:
            # Add ticker column for reference
            institutional_holders['Ticker'] = ticker
            
            # Convert date column to datetime if it exists
            if 'Date Reported' in institutional_holders.columns:
                institutional_holders['Date Reported'] = pd.to_datetime(
                    institutional_holders['Date Reported']
                )
                
            logger.info(f"Successfully fetched 13F holdings data for {ticker}")
            return institutional_holders
        else:
            # If no institutional data but we have major holders
            # Convert major_holders to a proper DataFrame with column names
            if not (major_holders is None or major_holders.empty):
                # Major holders is often returned in a strange format
                # We need to reshape it into a more usable form
                try:
                    # Try to create a meaningful DataFrame from major_holders
                    holders_df = pd.DataFrame({
                        'Holder Type': major_holders.iloc[:, 0],
                        'Percentage': major_holders.iloc[:, 1]
                    })
                    holders_df['Ticker'] = ticker
                    
                    logger.info(f"Successfully fetched major holders data for {ticker}")
                    return holders_df
                except Exception as e:
                    logger.error(f"Error processing major holders data: {str(e)}")
                    return pd.DataFrame()
            
            return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Error fetching 13F/13D holdings data for {ticker}: {str(e)}")
        return pd.DataFrame()


def get_mutual_fund_holdings(ticker: str) -> pd.DataFrame:
    """
    Fetch mutual fund holdings data for a given ticker.
    
    Args:
        ticker: The ticker symbol to fetch data for.
        
    Returns:
        DataFrame: Mutual fund holdings data for the specified ticker.
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
        logger.info(f"Fetching mutual fund holdings data for {ticker}")
        ticker_obj = yf.Ticker(ticker)
        
        # Get mutual fund holders
        mutual_fund_holders = ticker_obj.mutualfund_holders
        
        # Check if we got any data
        if mutual_fund_holders is None or mutual_fund_holders.empty:
            logger.warning(f"No mutual fund holdings data available for {ticker}")
            return pd.DataFrame()
        
        # Add ticker column for reference
        mutual_fund_holders['Ticker'] = ticker
        
        # Convert date column to datetime if it exists
        if 'Date Reported' in mutual_fund_holders.columns:
            mutual_fund_holders['Date Reported'] = pd.to_datetime(
                mutual_fund_holders['Date Reported']
            )
        
        logger.info(f"Successfully fetched mutual fund holdings data for {ticker}")
        return mutual_fund_holders
        
    except Exception as e:
        logger.error(f"Error fetching mutual fund holdings data for {ticker}: {str(e)}")
        return pd.DataFrame()


def get_corporate_actions(ticker: str) -> pd.DataFrame:
    """
    Fetch corporate actions (dividends, splits, etc.) for a given ticker.
    
    Args:
        ticker: The ticker symbol to fetch data for.
        
    Returns:
        DataFrame: Corporate actions data for the specified ticker.
        Empty DataFrame if ticker is invalid or data cannot be fetched.
        The DataFrame contains dividends and stock splits information.
    
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
        logger.info(f"Fetching corporate actions data for {ticker}")
        ticker_obj = yf.Ticker(ticker)
        
        # Get dividends
        dividends = ticker_obj.dividends
        if not isinstance(dividends, pd.Series) or dividends.empty:
            logger.warning(f"No dividend data available for {ticker}")
            dividends = pd.Series(dtype='float64')
        
        # Get stock splits
        splits = ticker_obj.splits
        if not isinstance(splits, pd.Series) or splits.empty:
            logger.warning(f"No stock split data available for {ticker}")
            splits = pd.Series(dtype='float64')
        
        # If both are empty, return empty DataFrame
        if dividends.empty and splits.empty:
            logger.warning(f"No corporate actions data available for {ticker}")
            return pd.DataFrame()
        
        # Create DataFrames from Series
        actions_data = {}
        
        if not dividends.empty:
            # Convert dividends Series to DataFrame
            div_df = dividends.reset_index()
            div_df.columns = ['Date', 'Dividend']
            div_df['Action'] = 'Dividend'
            div_df['Ticker'] = ticker
            actions_data['dividends'] = div_df
        
        if not splits.empty:
            # Convert splits Series to DataFrame
            split_df = splits.reset_index()
            split_df.columns = ['Date', 'Split Ratio']
            split_df['Action'] = 'Stock Split'
            split_df['Ticker'] = ticker
            actions_data['splits'] = split_df
        
        # Combine the DataFrames
        if actions_data:
            result = pd.concat(actions_data.values(), ignore_index=True)
            result = result.sort_values('Date', ascending=False).reset_index(drop=True)
            logger.info(f"Successfully fetched corporate actions data for {ticker}")
            return result
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error fetching corporate actions data for {ticker}: {str(e)}")
        return pd.DataFrame()


def retry_api_call(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to retry API calls with exponential backoff.
    
    This decorator will retry the decorated function if it raises an exception,
    with an exponential backoff delay between retries. It will retry up to
    MAX_RETRIES times before giving up.
    
    Args:
        func: The function to decorate.
        
    Returns:
        The decorated function with retry logic.
        
    Example:
        @retry_api_call
        def fetch_data(ticker):
            # API call that might fail
            return yf.Ticker(ticker).info
    """
    import time
    from functools import wraps
    
    MAX_RETRIES = 3
    BASE_DELAY = 1  # seconds
    
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        retries = 0
        last_exception = None
        
        while retries <= MAX_RETRIES:
            try:
                if retries > 0:
                    logger.warning(
                        f"Retry {retries}/{MAX_RETRIES} for {func.__name__} with args {args} and kwargs {kwargs}"
                    )
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                retries += 1
                
                if retries <= MAX_RETRIES:
                    # Calculate exponential backoff delay: 1s, 2s, 4s, ...
                    delay = BASE_DELAY * (2 ** (retries - 1))
                    logger.warning(
                        f"Exception in {func.__name__}: {str(e)}. "
                        f"Retrying in {delay} seconds... ({retries}/{MAX_RETRIES})"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Max retries ({MAX_RETRIES}) exceeded for {func.__name__}. "
                        f"Last exception: {str(e)}"
                    )
                    raise last_exception
    
    return wrapper
