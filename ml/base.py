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
from ml.metrics import (
    classification_metrics,
    regression_metrics,
    get_feature_importance
)
from ml.classification import LogisticRegressionModel

# Configure logging
logger = logging.getLogger(__name__)

# Re-export functions for easier imports
__all__ = [
    'split_data',
    'classification_metrics',
    'regression_metrics',
    'get_feature_importance',
    'LogisticRegressionModel'
]
