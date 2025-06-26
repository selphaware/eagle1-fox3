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
