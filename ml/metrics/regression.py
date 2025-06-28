"""
Regression metrics utilities for machine learning models.

This module provides functions for calculating various performance metrics
for regression models.
"""

__author__ = "Usman Ahmad"

from typing import Dict, List, Union, Optional, Tuple, Any, cast
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, median_absolute_error,
    mean_absolute_percentage_error, max_error
)
import logging

# Configure logging
logger = logging.getLogger(__name__)


def regression_metrics(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series]
) -> Dict[str, Union[float, str]]:
    """
    Calculate regression metrics for model evaluation.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a regressor.

    Returns:
        Dictionary containing various regression metrics:
            - mse: Mean squared error
            - rmse: Root mean squared error
            - mae: Mean absolute error
            - r2: R-squared (coefficient of determination)
            - explained_variance: Explained variance score
            - median_ae: Median absolute error
            - mape: Mean absolute percentage error (if no zero values in y_true)
            - max_error: Maximum residual error
            - rmse_to_stdev: Ratio of RMSE to standard deviation of y_true

    Raises:
        TypeError: If input types are invalid.
        ValueError: If inputs have incompatible shapes.
    """
    # Validate inputs
    if not isinstance(y_true, (list, np.ndarray, pd.Series)):
        logger.error("TypeError: y_true must be a list, numpy array, or pandas Series")
        raise TypeError("y_true must be a list, numpy array, or pandas Series")
    
    if not isinstance(y_pred, (list, np.ndarray, pd.Series)):
        logger.error("TypeError: y_pred must be a list, numpy array, or pandas Series")
        raise TypeError("y_pred must be a list, numpy array, or pandas Series")
    
    # Convert inputs to numpy arrays for consistency
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    # Check shapes
    if y_true_array.shape != y_pred_array.shape:
        logger.error(f"ValueError: y_true and y_pred have incompatible shapes: "
                    f"{y_true_array.shape} vs {y_pred_array.shape}")
        raise ValueError(f"y_true and y_pred have incompatible shapes: "
                        f"{y_true_array.shape} vs {y_pred_array.shape}")
    
    # Check for empty arrays
    if len(y_true_array) == 0:
        logger.error("ValueError: Empty arrays provided")
        raise ValueError("Empty arrays provided")
    
    # Initialize metrics dictionary
    metrics: Dict[str, Union[float, str]] = {}
    
    try:
        # Calculate basic regression metrics
        mse = mean_squared_error(y_true_array, y_pred_array)
        metrics['mse'] = float(mse)
        metrics['rmse'] = float(np.sqrt(mse))
        metrics['mae'] = float(mean_absolute_error(y_true_array, y_pred_array))
        
        # Handle R² and explained variance for single sample case
        if len(y_true_array) < 2:
            logger.warning("R² and explained variance are not well-defined with less than two samples. "
                          "Setting these metrics to NaN.")
            metrics['r2'] = float('nan')
            metrics['explained_variance'] = float('nan')
        else:
            metrics['r2'] = float(r2_score(y_true_array, y_pred_array))
            metrics['explained_variance'] = float(explained_variance_score(y_true_array, y_pred_array))
        
        # Additional regression metrics
        metrics['median_ae'] = float(median_absolute_error(y_true_array, y_pred_array))
        metrics['max_error'] = float(max_error(y_true_array, y_pred_array))
        
        # Calculate MAPE if no zero values in y_true
        if np.all(y_true_array != 0):
            try:
                metrics['mape'] = float(mean_absolute_percentage_error(y_true_array, y_pred_array))
            except Exception as e:
                logger.warning(f"Could not calculate MAPE: {str(e)}")
                metrics['mape'] = float('nan')
        else:
            logger.warning("MAPE not calculated: y_true contains zero values")
            metrics['mape'] = float('nan')
        
        # Calculate RMSE to standard deviation ratio (if std > 0)
        if len(y_true_array) >= 2:
            std_dev = np.std(y_true_array)
            if std_dev > 0:
                metrics['rmse_to_stdev'] = float(metrics['rmse'] / std_dev)
            else:
                # For perfect predictions or constant targets, set rmse_to_stdev to 0
                # This is more intuitive than NaN for perfect predictions
                logger.warning("rmse_to_stdev set to 0: standard deviation is zero")
                metrics['rmse_to_stdev'] = 0.0
        else:
            # For single sample, set rmse_to_stdev to 0 if rmse is 0, otherwise NaN
            if metrics['rmse'] == 0:
                metrics['rmse_to_stdev'] = 0.0
            else:
                metrics['rmse_to_stdev'] = float('nan')
        
        # Log metrics
        logger.info(f"Regression metrics calculated: mse={metrics['mse']:.4f}, "
                   f"rmse={metrics['rmse']:.4f}, mae={metrics['mae']:.4f}, "
                   f"r2={metrics.get('r2', 'nan')}")
        
    except Exception as e:
        logger.error(f"Error calculating regression metrics: {str(e)}")
        raise
    
    return metrics
