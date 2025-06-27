"""
Metrics utilities for machine learning models.

This module provides functions for calculating various performance metrics
for classification and regression models.
"""

__author__ = "Usman Ahmad"

from typing import Dict, List, Union, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score
)
import logging

# Configure logging
logger = logging.getLogger(__name__)


def classification_metrics(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    y_prob: Optional[Union[List, np.ndarray, pd.Series]] = None,
    average: str = 'binary'
) -> Dict[str, float]:
    """
    Calculate classification metrics for model evaluation.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a classifier.
        y_prob: Probability estimates for the positive class. Required for ROC AUC.
            If None, ROC AUC will not be calculated.
        average: Parameter for precision, recall, and f1 score.
            Options: 'binary', 'micro', 'macro', 'weighted', 'samples'.
            Default is 'binary' for binary classification.

    Returns:
        Dictionary containing various classification metrics:
            - accuracy: Accuracy score
            - precision: Precision score
            - recall: Recall score
            - f1: F1 score
            - roc_auc: ROC AUC score (only for binary classification and if y_prob is provided)
            - confusion_matrix: Confusion matrix as a nested list

    Raises:
        TypeError: If input types are invalid.
        ValueError: If inputs have incompatible shapes or if average parameter is invalid.
    """
    # Validate inputs
    if not isinstance(y_true, (list, np.ndarray, pd.Series)):
        logger.error("TypeError: y_true must be a list, numpy array, or pandas Series")
        raise TypeError("y_true must be a list, numpy array, or pandas Series")
    
    if not isinstance(y_pred, (list, np.ndarray, pd.Series)):
        logger.error("TypeError: y_pred must be a list, numpy array, or pandas Series")
        raise TypeError("y_pred must be a list, numpy array, or pandas Series")
    
    if y_prob is not None and not isinstance(y_prob, (list, np.ndarray, pd.Series)):
        logger.error("TypeError: y_prob must be a list, numpy array, or pandas Series")
        raise TypeError("y_prob must be a list, numpy array, or pandas Series")
    
    # Convert inputs to numpy arrays for consistency
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    # Check shapes
    if y_true_array.shape != y_pred_array.shape:
        logger.error(f"ValueError: y_true and y_pred have incompatible shapes: "
                    f"{y_true_array.shape} vs {y_pred_array.shape}")
        raise ValueError(f"y_true and y_pred have incompatible shapes: "
                        f"{y_true_array.shape} vs {y_pred_array.shape}")
    
    # Initialize metrics dictionary
    metrics = {}
    
    try:
        # Calculate basic classification metrics
        metrics['accuracy'] = float(accuracy_score(y_true_array, y_pred_array))
        metrics['precision'] = float(precision_score(y_true_array, y_pred_array, average=average, zero_division=0))
        metrics['recall'] = float(recall_score(y_true_array, y_pred_array, average=average, zero_division=0))
        metrics['f1'] = float(f1_score(y_true_array, y_pred_array, average=average, zero_division=0))
        
        # Calculate ROC AUC if probabilities are provided and it's binary classification
        if y_prob is not None:
            y_prob_array = np.array(y_prob)
            
            # Check if binary classification (2 unique classes)
            unique_classes = np.unique(y_true_array)
            if len(unique_classes) == 2:
                try:
                    metrics['roc_auc'] = float(roc_auc_score(y_true_array, y_prob_array))
                except Exception as e:
                    logger.warning(f"Could not calculate ROC AUC: {str(e)}")
                    metrics['roc_auc'] = None
            else:
                logger.info("ROC AUC not calculated: not a binary classification problem")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true_array, y_pred_array)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Log metrics
        logger.info(f"Classification metrics calculated: accuracy={metrics['accuracy']:.4f}, "
                   f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
                   f"f1={metrics['f1']:.4f}")
        
    except Exception as e:
        logger.error(f"Error calculating classification metrics: {str(e)}")
        raise
    
    return metrics


def regression_metrics(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series]
) -> Dict[str, float]:
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
    
    # Initialize metrics dictionary
    metrics = {}
    
    try:
        # Calculate regression metrics
        mse = mean_squared_error(y_true_array, y_pred_array)
        metrics['mse'] = float(mse)
        metrics['rmse'] = float(np.sqrt(mse))
        metrics['mae'] = float(mean_absolute_error(y_true_array, y_pred_array))
        metrics['r2'] = float(r2_score(y_true_array, y_pred_array))
        metrics['explained_variance'] = float(explained_variance_score(y_true_array, y_pred_array))
        
        # Log metrics
        logger.info(f"Regression metrics calculated: mse={metrics['mse']:.4f}, "
                   f"rmse={metrics['rmse']:.4f}, mae={metrics['mae']:.4f}, "
                   f"r2={metrics['r2']:.4f}")
        
    except Exception as e:
        logger.error(f"Error calculating regression metrics: {str(e)}")
        raise
    
    return metrics


def get_feature_importance(
    model: Any,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.

    Args:
        model: Trained model with feature_importances_ attribute
               (e.g., RandomForest, XGBoost, etc.)
        feature_names: List of feature names corresponding to the model's features.

    Returns:
        DataFrame with feature names and their importance scores,
        sorted by importance in descending order.

    Raises:
        TypeError: If model doesn't have feature_importances_ attribute.
        ValueError: If lengths of feature_importances_ and feature_names don't match.
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        logger.error("TypeError: Model does not have feature_importances_ attribute")
        raise TypeError("Model does not have feature_importances_ attribute")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Validate feature names
    if len(importances) != len(feature_names):
        logger.error(f"ValueError: Length mismatch between feature_importances_ "
                    f"({len(importances)}) and feature_names ({len(feature_names)})")
        raise ValueError(f"Length mismatch between feature_importances_ "
                        f"({len(importances)}) and feature_names ({len(feature_names)})")
    
    # Create DataFrame with feature importances
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance in descending order
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    logger.info(f"Feature importance extracted for {len(feature_names)} features")
    
    return importance_df
