"""
Data splitting utilities for machine learning models.

This module provides functions for splitting datasets into training and testing sets,
with options for stratification and different test sizes.
"""

__author__ = "Usman Ahmad"

from typing import Tuple, Optional, Union, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logger = logging.getLogger(__name__)


def split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    stratify: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a DataFrame into training and testing sets.

    Args:
        df: Input DataFrame containing features and target.
        target: Name of the target column.
        test_size: Proportion of the dataset to include in the test split.
            Must be between 0 and 1. Default is 0.2 (20%).
        random_state: Controls the shuffling applied to the data before applying
            the split. Pass an int for reproducible output.
        stratify: If True, data is split in a stratified fashion, using the target
            column as the class labels. Default is False.

    Returns:
        A tuple containing (X_train, X_test, y_train, y_test) where:
            - X_train: Training features
            - X_test: Testing features
            - y_train: Training target
            - y_test: Testing target

    Raises:
        TypeError: If df is not a pandas DataFrame or target is not a string.
        ValueError: If target is not in df columns, test_size is not between 0 and 1,
            or if stratify is True but target column has only one unique value.
    """
    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        logger.error("TypeError: df must be a pandas DataFrame")
        raise TypeError("df must be a pandas DataFrame")
    
    if not isinstance(target, str):
        logger.error("TypeError: target must be a string")
        raise TypeError("target must be a string")
    
    if target not in df.columns:
        logger.error(f"ValueError: target column '{target}' not found in DataFrame")
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    
    if not 0 < test_size < 1:
        logger.error(f"ValueError: test_size must be between 0 and 1, got {test_size}")
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    
    # Extract features and target
    X = df.drop(columns=[target])
    y = df[target]
    
    logger.info(f"Splitting data with test_size={test_size}, stratify={stratify}")
    
    # Check if stratification is possible
    if stratify and len(y.unique()) <= 1:
        logger.warning("Cannot stratify with only one class. Falling back to random split.")
        stratify_param = None
    else:
        stratify_param = y if stratify else None
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    
    # Log split information
    logger.info(f"Split complete. Training set: {X_train.shape[0]} samples, "
                f"Testing set: {X_test.shape[0]} samples")
    
    if stratify and stratify_param is not None:
        # Log class distribution in training and testing sets
        train_dist = y_train.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)
        logger.debug(f"Training set class distribution:\n{train_dist}")
        logger.debug(f"Testing set class distribution:\n{test_dist}")
    
    return X_train, X_test, y_train, y_test
