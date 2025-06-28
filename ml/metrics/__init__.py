"""
Metrics package for machine learning model evaluation.

This package provides functions for calculating various performance metrics
for classification and regression models.
"""

__author__ = "Usman Ahmad"

from ml.metrics.classification import (
    binary_classification_metrics,
    multiclass_classification_metrics,
    precision_recall_curve_data,
    roc_curve_data,
    classification_report_dict
)

from ml.metrics.regression import (
    regression_metrics
)

from ml.metrics.feature_importance import (
    get_feature_importance
)

# Re-export for backward compatibility
from ml.metrics.base import classification_metrics

__all__ = [
    'classification_metrics',
    'binary_classification_metrics',
    'multiclass_classification_metrics',
    'precision_recall_curve_data',
    'roc_curve_data',
    'classification_report_dict',
    'regression_metrics',
    'get_feature_importance'
]
