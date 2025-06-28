"""
Regression models for machine learning.

This package provides implementations of various regression models
for financial data analysis.
"""

__author__ = "Usman Ahmad"

from ml.regression.linear_regression import LinearRegressionModel
from ml.regression.random_forest import RandomForestRegressionModel
from ml.regression.tensorflow_dnn import TensorFlowDNNRegressor

__all__ = [
    'LinearRegressionModel',
    'RandomForestRegressionModel',
    'TensorFlowDNNRegressor',
]
