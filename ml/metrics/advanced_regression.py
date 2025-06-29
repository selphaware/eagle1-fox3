"""
Advanced regression metrics for machine learning models.

This module provides functions for calculating additional performance metrics
for regression models beyond the basic metrics in regression.py.
"""

from typing import Dict, List, Union, Optional, Tuple, Any, Callable
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_log_error, mean_poisson_deviance,
    mean_gamma_deviance, mean_tweedie_deviance,
    d2_tweedie_score, d2_absolute_error_score,
    d2_pinball_score
)
import logging
from ml.metrics.regression import regression_metrics

# Configure logging
logger = logging.getLogger(__name__)


def advanced_regression_metrics(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    include_basic_metrics: bool = True
) -> Dict[str, float]:
    """
    Calculate advanced regression metrics for model evaluation.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a regressor.
        include_basic_metrics: Whether to include basic metrics from regression_metrics.

    Returns:
        Dictionary containing various advanced regression metrics:
            - All basic metrics from regression_metrics (if include_basic_metrics=True)
            - msle: Mean squared logarithmic error (if all values are positive)
            - rmsle: Root mean squared logarithmic error (if all values are positive)
            - poisson_deviance: Mean Poisson deviance (for count data)
            - gamma_deviance: Mean Gamma deviance (for positive continuous data)
            - tweedie_deviance: Mean Tweedie deviance (for non-negative data)
            - d2_absolute: D² score based on absolute error
            - d2_pinball: D² score based on pinball loss
            - r2_adjusted: Adjusted R-squared (penalizes additional features)
            - residual_std: Standard deviation of residuals
            - prediction_std: Standard deviation of predictions

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
    if include_basic_metrics:
        metrics = regression_metrics(y_true_array, y_pred_array)
    else:
        metrics = {}
    
    try:
        # Calculate residuals
        residuals = y_true_array - y_pred_array
        
        # Standard deviation of residuals
        if len(residuals) >= 2:
            metrics['residual_std'] = float(np.std(residuals, ddof=1))
        else:
            metrics['residual_std'] = float('nan')
        
        # Standard deviation of predictions
        if len(y_pred_array) >= 2:
            metrics['prediction_std'] = float(np.std(y_pred_array, ddof=1))
        else:
            metrics['prediction_std'] = float('nan')
        
        # Mean squared logarithmic error (for positive values)
        if np.all(y_true_array > 0) and np.all(y_pred_array > 0):
            try:
                msle = mean_squared_log_error(y_true_array, y_pred_array)
                metrics['msle'] = float(msle)
                metrics['rmsle'] = float(np.sqrt(msle))
            except Exception as e:
                logger.warning(f"Could not calculate MSLE: {str(e)}")
                metrics['msle'] = float('nan')
                metrics['rmsle'] = float('nan')
        else:
            logger.warning("MSLE not calculated: inputs contain non-positive values")
            metrics['msle'] = float('nan')
            metrics['rmsle'] = float('nan')
        
        # Distribution-specific metrics (handle with try-except as they have specific requirements)
        try:
            # Poisson deviance (for count data)
            metrics['poisson_deviance'] = float(
                mean_poisson_deviance(y_true_array, y_pred_array)
            )
        except Exception as e:
            logger.warning(f"Could not calculate Poisson deviance: {str(e)}")
            metrics['poisson_deviance'] = float('nan')
        
        try:
            # Gamma deviance (for positive continuous data)
            if np.all(y_true_array > 0) and np.all(y_pred_array > 0):
                metrics['gamma_deviance'] = float(
                    mean_gamma_deviance(y_true_array, y_pred_array)
                )
            else:
                logger.warning("Gamma deviance not calculated: inputs contain non-positive values")
                metrics['gamma_deviance'] = float('nan')
        except Exception as e:
            logger.warning(f"Could not calculate Gamma deviance: {str(e)}")
            metrics['gamma_deviance'] = float('nan')
        
        try:
            # Tweedie deviance (for non-negative data)
            if np.all(y_true_array >= 0) and np.all(y_pred_array > 0):
                metrics['tweedie_deviance'] = float(
                    mean_tweedie_deviance(y_true_array, y_pred_array, power=1.5)
                )
            else:
                logger.warning("Tweedie deviance not calculated: inputs contain negative values")
                metrics['tweedie_deviance'] = float('nan')
        except Exception as e:
            logger.warning(f"Could not calculate Tweedie deviance: {str(e)}")
            metrics['tweedie_deviance'] = float('nan')
        
        # D² scores (alternative to R²)
        try:
            if len(y_true_array) >= 2:
                metrics['d2_absolute'] = float(
                    d2_absolute_error_score(y_true_array, y_pred_array)
                )
                metrics['d2_pinball'] = float(
                    d2_pinball_score(y_true_array, y_pred_array, alpha=0.5)
                )
            else:
                logger.warning("D² scores not calculated: need at least 2 samples")
                metrics['d2_absolute'] = float('nan')
                metrics['d2_pinball'] = float('nan')
        except Exception as e:
            logger.warning(f"Could not calculate D² scores: {str(e)}")
            metrics['d2_absolute'] = float('nan')
            metrics['d2_pinball'] = float('nan')
        
        # Log metrics
        logger.info(f"Advanced regression metrics calculated: "
                   f"residual_std={metrics.get('residual_std', 'nan'):.4f}, "
                   f"rmsle={metrics.get('rmsle', 'nan')}")
        
    except Exception as e:
        logger.error(f"Error calculating advanced regression metrics: {str(e)}")
        raise
    
    return metrics


def regression_metrics_by_group(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    groups: Union[pd.Series, np.ndarray],
    metric_fn: Callable = regression_metrics
) -> pd.DataFrame:
    """
    Calculate regression metrics for each group in the data.
    
    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a regressor.
        groups: Group labels for each sample.
        metric_fn: Function to calculate metrics for each group.
                  Default is regression_metrics.
    
    Returns:
        DataFrame with metrics for each group.
        
    Raises:
        ValueError: If inputs have incompatible shapes.
    """
    # Convert inputs to numpy arrays if needed
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if isinstance(groups, pd.Series):
        groups = groups.values
    
    # Check shapes
    if len(y_true) != len(y_pred) or len(y_true) != len(groups):
        raise ValueError("y_true, y_pred, and groups must have the same length")
    
    # Get unique groups
    unique_groups = np.unique(groups)
    
    # Calculate metrics for each group
    results = []
    for group in unique_groups:
        mask = groups == group
        if sum(mask) > 0:  # Only calculate if group has samples
            try:
                group_metrics = metric_fn(y_true[mask], y_pred[mask])
                group_metrics['group'] = group
                group_metrics['count'] = sum(mask)
                results.append(group_metrics)
            except Exception as e:
                logger.warning(f"Could not calculate metrics for group {group}: {str(e)}")
    
    # Convert to DataFrame
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame()


def confidence_interval_metrics(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    confidence_level: float = 0.95,
    bootstrap_samples: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate regression metrics with confidence intervals using bootstrap.
    
    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a regressor.
        confidence_level: Confidence level for intervals (between 0 and 1).
        bootstrap_samples: Number of bootstrap samples to generate.
        random_state: Random seed for reproducibility.
        
    Returns:
        Dictionary with metrics and their confidence intervals.
        Each metric has 'value', 'lower_bound', and 'upper_bound'.
        
    Raises:
        ValueError: If inputs have incompatible shapes or invalid parameters.
    """
    # Validate inputs
    if not isinstance(y_true, (list, np.ndarray, pd.Series)):
        raise TypeError("y_true must be a list, numpy array, or pandas Series")
    
    if not isinstance(y_pred, (list, np.ndarray, pd.Series)):
        raise TypeError("y_pred must be a list, numpy array, or pandas Series")
    
    # Convert inputs to numpy arrays for consistency
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    # Check shapes
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError(f"y_true and y_pred have incompatible shapes: "
                        f"{y_true_array.shape} vs {y_pred_array.shape}")
    
    # Check confidence level
    if not 0 < confidence_level < 1:
        raise ValueError(f"confidence_level must be between 0 and 1, got {confidence_level}")
    
    # Set random seed
    np.random.seed(random_state)
    
    # Calculate base metrics
    base_metrics = regression_metrics(y_true_array, y_pred_array)
    
    # Initialize results dictionary
    results = {}
    
    # Calculate alpha for confidence interval
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    # Generate bootstrap samples and calculate metrics
    n_samples = len(y_true_array)
    bootstrap_metrics = []
    
    for _ in range(bootstrap_samples):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_y_true = y_true_array[indices]
        bootstrap_y_pred = y_pred_array[indices]
        
        # Calculate metrics for this bootstrap sample
        try:
            metrics = regression_metrics(bootstrap_y_true, bootstrap_y_pred)
            bootstrap_metrics.append(metrics)
        except Exception:
            # Skip failed bootstrap samples
            continue
    
    # Convert list of dictionaries to dictionary of lists
    bootstrap_values = {}
    for metric in base_metrics:
        bootstrap_values[metric] = [
            metrics[metric] for metrics in bootstrap_metrics
            if metric in metrics and not np.isnan(metrics[metric])
        ]
    
    # Calculate confidence intervals
    for metric, values in bootstrap_values.items():
        if len(values) >= 10:  # Only calculate CI if we have enough bootstrap samples
            lower_bound = np.percentile(values, lower_percentile)
            upper_bound = np.percentile(values, upper_percentile)
            
            results[metric] = {
                'value': base_metrics[metric],
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
        else:
            results[metric] = {
                'value': base_metrics[metric],
                'lower_bound': float('nan'),
                'upper_bound': float('nan')
            }
    
    return results
