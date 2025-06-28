"""
Feature importance utilities for machine learning models.

This module provides functions for extracting and analyzing feature importance
from various types of machine learning models.
"""

__author__ = "Usman Ahmad"

from typing import Dict, List, Union, Optional, Tuple, Any, cast
import numpy as np
import pandas as pd
import logging

# Configure logging
logger = logging.getLogger(__name__)


def get_feature_importance(
    model: Any,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    
    This function works with various model types including scikit-learn models
    with feature_importances_ attribute (e.g., RandomForest, GradientBoosting),
    linear models with coef_ attribute (e.g., LogisticRegression, LinearRegression),
    and other models that expose feature importance.
    
    Args:
        model: Trained model object.
        feature_names: List of feature names corresponding to the features used in the model.
            Must match the number of features the model was trained on.
            
    Returns:
        DataFrame with feature names and their importance scores, sorted by importance
        in descending order.
        
    Raises:
        TypeError: If model or feature_names are invalid types.
        ValueError: If feature_names length doesn't match model's feature count or
            if the model doesn't support feature importance extraction.
    """
    # Validate inputs
    if not isinstance(feature_names, list):
        logger.error("TypeError: feature_names must be a list")
        raise TypeError("feature_names must be a list")
    
    if not all(isinstance(name, str) for name in feature_names):
        logger.error("TypeError: All elements in feature_names must be strings")
        raise TypeError("All elements in feature_names must be strings")
        
    if len(feature_names) == 0:
        logger.error("ValueError: feature_names cannot be empty")
        raise ValueError("feature_names cannot be empty")
    
    # Initialize importance array
    importance = None
    
    try:
        # Try to extract feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            # Tree-based models like RandomForest, XGBoost, etc.
            importance = model.feature_importances_
            
        elif hasattr(model, 'coef_'):
            # Linear models like LogisticRegression, LinearRegression, etc.
            coef = model.coef_
            
            # Handle multi-class case where coef_ is a 2D array
            if hasattr(coef, 'shape') and len(coef.shape) > 1:
                # For multi-class, take the mean absolute value across classes
                importance = np.mean(np.abs(coef), axis=0)
            else:
                # For binary classification or regression
                importance = np.abs(coef)
                
        elif hasattr(model, 'estimators_') and hasattr(model, 'estimator_weights_'):
            # Ensemble models like AdaBoost
            importances = []
            for estimator, weight in zip(model.estimators_, model.estimator_weights_):
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(weight * estimator.feature_importances_)
                elif hasattr(estimator, 'coef_'):
                    coef = estimator.coef_
                    if hasattr(coef, 'shape') and len(coef.shape) > 1:
                        importances.append(weight * np.mean(np.abs(coef), axis=0))
                    else:
                        importances.append(weight * np.abs(coef))
            
            if importances:
                importance = np.mean(importances, axis=0)
                
        # Check if we successfully extracted feature importance
        if importance is None:
            logger.error("ValueError: Model does not support feature importance extraction")
            raise ValueError("Model does not support feature importance extraction")
        
        # Validate feature count
        try:
            importance_len = len(importance)
        except TypeError:
            # Convert to numpy array if it's not already one
            importance = np.array(importance)
            importance_len = len(importance)
            
        if importance_len != len(feature_names):
            logger.error(f"ValueError: Number of features in model ({importance_len}) "
                        f"does not match length of feature_names ({len(feature_names)})")
            raise ValueError(f"Number of features in model ({importance_len}) "
                            f"does not match length of feature_names ({len(feature_names)})")
            
        # Ensure importance is a numpy array for consistent handling
        if not isinstance(importance, np.ndarray):
            importance = np.array(importance)
        
        # Create DataFrame with feature names and importance scores
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        # Sort by importance in descending order
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        # Log successful extraction
        logger.info(f"Feature importance extracted for {len(feature_names)} features")
        
        return importance_df
    
    except Exception as e:
        logger.error(f"Error extracting feature importance: {str(e)}")
        # Re-raise with more context
        raise ValueError(f"Failed to extract feature importance: {str(e)}") from e
