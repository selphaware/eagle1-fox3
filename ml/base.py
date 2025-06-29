"""
Base machine learning utilities.

This module provides core machine learning functionality including data splitting,
metrics calculation, and other common ML operations.
"""

__author__ = "Usman Ahmad"

from typing import Tuple, Optional, Union, List
import pandas as pd
import numpy as np
import logging

# Import functionality from submodules
from ml.data_splitting import split_data

# Import metrics functionality
from ml.metrics import (
    classification_metrics,
    binary_classification_metrics,
    multiclass_classification_metrics,
    precision_recall_curve_data,
    roc_curve_data,
    classification_report_dict,
    regression_metrics,
    get_feature_importance,
    # Advanced regression metrics
    advanced_regression_metrics,
    regression_metrics_by_group,
    confidence_interval_metrics,
    # Visualization functions
    plot_residuals,
    plot_actual_vs_predicted,
    plot_prediction_error_distribution,
    plot_metrics_comparison,
    plot_prediction_intervals
)

# Import classification models
from ml.classification import LogisticRegressionModel, RandomForestClassifier, DNNClassifier

# Configure logging
logger = logging.getLogger(__name__)

# Re-export functions for easier imports
__all__ = [
    # Data splitting
    'split_data',
    
    # Classification metrics
    'classification_metrics',
    'binary_classification_metrics',
    'multiclass_classification_metrics',
    'precision_recall_curve_data',
    'roc_curve_data',
    'classification_report_dict',
    
    # Regression metrics
    'regression_metrics',
    'advanced_regression_metrics',
    'regression_metrics_by_group',
    'confidence_interval_metrics',
    
    # Visualization functions
    'plot_residuals',
    'plot_actual_vs_predicted',
    'plot_prediction_error_distribution',
    'plot_metrics_comparison',
    'plot_prediction_intervals',
    
    # Feature importance
    'get_feature_importance',
    
    # Classification models
    'LogisticRegressionModel',
    'RandomForestClassifier',
    'DNNClassifier'
]
