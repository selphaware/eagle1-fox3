"""Module for caching financial data to reduce API calls."""

__author__ = "Usman Ahmad"

from typing import Dict, Optional, Union, Any
import os
import pickle
from datetime import datetime, timedelta
import pandas as pd


def save_to_cache(key: str, df: pd.DataFrame) -> None:
    """
    Save DataFrame to cache using pickle.
    
    Creates a cache directory if it doesn't exist and saves the DataFrame
    along with a timestamp for cache invalidation checks.
    
    Args:
        key: Unique identifier for the cached data.
        df: DataFrame to cache.
    
    Returns:
        None
    
    Raises:
        ValueError: If key is empty or invalid.
        TypeError: If df is not a pandas DataFrame.
        OSError: If there's an issue with file operations.
    """
    import logging
    
    # Configure logging
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not isinstance(key, str) or not key.strip():
        logger.error("Invalid key: key must be a non-empty string")
        raise ValueError("Key must be a non-empty string")
    
    if not isinstance(df, pd.DataFrame):
        logger.error(f"Invalid data type: expected DataFrame, got {type(df).__name__}")
        raise TypeError(f"Data must be a pandas DataFrame, not {type(df).__name__}")
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Sanitize key for filename
    safe_key = ''.join(c if c.isalnum() else '_' for c in key)
    cache_path = os.path.join(cache_dir, f"{safe_key}.pkl")
    
    try:
        # Create cache data structure with timestamp and DataFrame
        cache_data = {
            'timestamp': datetime.now(),
            'data': df
        }
        
        # Save to pickle file
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Successfully cached data for key '{key}'")
    except Exception as e:
        logger.error(f"Error saving data to cache for key '{key}': {str(e)}")
        raise OSError(f"Failed to save data to cache: {str(e)}")


def load_from_cache(key: str) -> Optional[pd.DataFrame]:
    """
    Load DataFrame from cache if available.
    
    Args:
        key: Unique identifier for the cached data.
    
    Returns:
        DataFrame if cache exists and is valid, None otherwise.
        
    Raises:
        ValueError: If key is empty or invalid.
    """
    import logging
    
    # Configure logging
    logger = logging.getLogger(__name__)
    
    # Validate input
    if not isinstance(key, str) or not key.strip():
        logger.error("Invalid key: key must be a non-empty string")
        raise ValueError("Key must be a non-empty string")
    
    # Sanitize key for filename
    safe_key = ''.join(c if c.isalnum() else '_' for c in key)
    
    # Get cache path
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
    cache_path = os.path.join(cache_dir, f"{safe_key}.pkl")
    
    # Check if cache file exists
    if not os.path.exists(cache_path):
        logger.info(f"No cache found for key '{key}'")
        return None
    
    try:
        # Load from pickle file
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Verify cache data structure
        if not isinstance(cache_data, dict) or 'data' not in cache_data or 'timestamp' not in cache_data:
            logger.warning(f"Corrupted cache for key '{key}': invalid structure")
            return None
        
        # Verify DataFrame
        if not isinstance(cache_data['data'], pd.DataFrame):
            logger.warning(f"Corrupted cache for key '{key}': data is not a DataFrame")
            return None
            
        logger.info(f"Successfully loaded cached data for key '{key}'")
        return cache_data['data']
    except (pickle.UnpicklingError, EOFError) as e:
        logger.warning(f"Corrupted cache file for key '{key}': {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error loading data from cache for key '{key}': {str(e)}")
        return None


def is_cache_stale(key: str, minutes: int = 60) -> bool:
    """
    Check if cache is older than specified minutes.
    
    Args:
        key: Unique identifier for the cached data.
        minutes: Number of minutes after which cache is considered stale.
    
    Returns:
        bool: True if cache is stale or doesn't exist, False otherwise.
        
    Raises:
        ValueError: If key is empty or invalid, or if minutes is not a positive integer.
    """
    import logging
    from datetime import datetime, timedelta
    
    # Configure logging
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not isinstance(key, str) or not key.strip():
        logger.error("Invalid key: key must be a non-empty string")
        raise ValueError("Key must be a non-empty string")
        
    if not isinstance(minutes, int) or minutes <= 0:
        logger.error(f"Invalid minutes value: {minutes}. Must be a positive integer")
        raise ValueError("Minutes must be a positive integer")
    
    # Sanitize key for filename
    safe_key = ''.join(c if c.isalnum() else '_' for c in key)
    
    # Get cache path
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
    cache_path = os.path.join(cache_dir, f"{safe_key}.pkl")
    
    # Check if cache file exists
    if not os.path.exists(cache_path):
        logger.info(f"No cache found for key '{key}'")
        return True  # No cache means it's stale
    
    try:
        # Load from pickle file
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Verify cache data structure
        if not isinstance(cache_data, dict) or 'timestamp' not in cache_data:
            logger.warning(f"Corrupted cache for key '{key}': missing timestamp")
            return True  # Corrupted cache is considered stale
        
        # Get timestamp
        timestamp = cache_data.get('timestamp')
        if not isinstance(timestamp, datetime):
            logger.warning(f"Corrupted cache for key '{key}': invalid timestamp type")
            return True  # Invalid timestamp means it's stale
        
        # Check if cache is stale
        current_time = datetime.now()
        max_age = timedelta(minutes=minutes)
        is_stale = (current_time - timestamp) > max_age
        
        if is_stale:
            logger.info(f"Cache for key '{key}' is stale (older than {minutes} minutes)")
        else:
            logger.info(f"Cache for key '{key}' is fresh (less than {minutes} minutes old)")
            
        return is_stale
        
    except (pickle.UnpicklingError, EOFError) as e:
        logger.warning(f"Corrupted cache file for key '{key}': {str(e)}")
        return True  # Corrupted cache is considered stale
    except Exception as e:
        logger.error(f"Error checking cache staleness for key '{key}': {str(e)}")
        return True  # Error means we should consider it stale
