"""
Classification models for machine learning.

This package provides implementations of various classification models
with consistent interfaces for training, prediction, and evaluation.
"""

__author__ = "Usman Ahmad"

from ml.classification.logistic_regression import LogisticRegressionModel
from ml.classification.random_forest import RandomForestClassifier
from ml.classification.tensorflow.dnn_classifier import DNNClassifier

__all__ = [
    'LogisticRegressionModel',
    'RandomForestClassifier',
    'DNNClassifier'
]
