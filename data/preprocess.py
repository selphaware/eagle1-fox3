"""Module for preprocessing financial data."""

__author__ = "Usman Ahmad"

from typing import Dict, List, Optional, Union, Literal
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def impute_missing(df: pd.DataFrame, method: Literal["mean", "median", "drop"] = "mean") -> pd.DataFrame:
    """
    Handle missing values in DataFrame by imputation or dropping.
    
    Args:
        df: Input DataFrame with potentially missing values.
        method: Method for handling missing values:
            - "mean": Replace with column mean
            - "median": Replace with column median
            - "drop": Drop rows with missing values
    
    Returns:
        DataFrame with missing values handled.
        
    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If method is not one of the supported methods.
        ValueError: If there are no numeric columns to impute (for mean/median).
    """
    import logging
    
    # Configure logging
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        logger.error("TypeError: df must be a pandas DataFrame")
        raise TypeError("df must be a pandas DataFrame")
    
    if method not in ["mean", "median", "drop"]:
        logger.error(f"ValueError: Invalid method '{method}'. Must be 'mean', 'median', or 'drop'")
        raise ValueError(f"Invalid method '{method}'. Must be 'mean', 'median', or 'drop'")
    
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Check if there are any missing values
    if not result_df.isna().any().any():
        logger.info("No missing values found in DataFrame")
        return result_df
    
    # Count missing values before processing
    missing_count = result_df.isna().sum().sum()
    logger.info(f"Found {missing_count} missing values in DataFrame")
    
    # Handle missing values based on method
    if method == "drop":
        # Drop rows with any missing values
        original_row_count = len(result_df)
        result_df = result_df.dropna()
        dropped_rows = original_row_count - len(result_df)
        logger.info(f"Dropped {dropped_rows} rows with missing values")
        
    else:  # mean or median imputation
        # Identify numeric columns
        numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            logger.error("ValueError: No numeric columns found for imputation")
            raise ValueError("No numeric columns found for imputation")
        
        # Create imputer with specified method
        imputer = SimpleImputer(strategy=method)
        
        # Only impute numeric columns
        result_df[numeric_cols] = pd.DataFrame(
            imputer.fit_transform(result_df[numeric_cols]),
            columns=numeric_cols,
            index=result_df.index
        )
        
        logger.info(f"Imputed missing values in {len(numeric_cols)} numeric columns using {method} strategy")
    
    # Verify results
    remaining_missing = result_df.isna().sum().sum()
    if remaining_missing > 0:
        logger.warning(f"There are still {remaining_missing} missing values in non-numeric columns")
    else:
        logger.info("Successfully handled all missing values")
    
    return result_df


def scale_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale numeric columns to have mean=0 and std=1 using StandardScaler.
    
    Args:
        df: Input DataFrame with numeric columns.
    
    Returns:
        DataFrame with scaled numeric columns. Non-numeric columns are left unchanged.
        
    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If there are no numeric columns to scale.
    """
    import logging
    
    # Configure logging
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        logger.error("TypeError: df must be a pandas DataFrame")
        raise TypeError("df must be a pandas DataFrame")
    
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Identify numeric columns
    numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
    
    # Check if there are any numeric columns
    if not numeric_cols:
        logger.error("ValueError: No numeric columns found for scaling")
        raise ValueError("No numeric columns found for scaling")
    
    # Log the columns that will be scaled
    logger.info(f"Scaling {len(numeric_cols)} numeric columns: {numeric_cols}")
    
    # Create scaler
    scaler = StandardScaler()
    
    # Scale numeric columns
    result_df[numeric_cols] = pd.DataFrame(
        scaler.fit_transform(result_df[numeric_cols]),
        columns=numeric_cols,
        index=result_df.index
    )
    
    # Log scaling results
    for col in numeric_cols:
        mean = result_df[col].mean()
        std = result_df[col].std()
        logger.debug(f"Column {col} scaled: mean={mean:.6f}, std={std:.6f}")
    
    logger.info("Successfully scaled numeric columns")
    
    return result_df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns using one-hot encoding.
    
    Args:
        df: Input DataFrame with categorical columns.
    
    Returns:
        DataFrame with encoded categorical columns.
    """
    # Placeholder implementation
    pass
