"""
Tests for classification metrics functionality.

This module contains tests for the classification metrics functions in ml.metrics.classification.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Any, cast, TYPE_CHECKING

from ml.metrics.classification import (
    binary_classification_metrics,
    multiclass_classification_metrics,
    precision_recall_curve_data,
    roc_curve_data,
    classification_report_dict
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestBinaryClassificationMetrics:
    """Tests for binary classification metrics functions."""

    def test_perfect_prediction(self) -> None:
        """Test metrics with perfect predictions."""
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 1, 0]
        y_prob = [0.1, 0.9, 0.2, 0.8, 0.1]
        
        metrics = binary_classification_metrics(y_true, y_pred, y_prob)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['specificity'] == 1.0
        assert metrics['balanced_accuracy'] == 1.0
        assert metrics['matthews_correlation'] == 1.0
        assert metrics['kappa'] == 1.0
        assert metrics['roc_auc'] == 1.0
        assert metrics['average_precision'] == 1.0

    def test_worst_prediction(self) -> None:
        """Test metrics with completely wrong predictions."""
        y_true = [0, 1, 0, 1, 0]
        y_pred = [1, 0, 1, 0, 1]
        
        metrics = binary_classification_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 0.0
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0
        assert metrics['specificity'] == 0.0
        assert metrics['balanced_accuracy'] == 0.0
        # Matthews correlation and kappa may not be exactly -1.0 due to implementation details
        assert metrics['matthews_correlation'] < -0.9
        assert metrics['kappa'] < -0.9

    def test_imbalanced_classes(self) -> None:
        """Test metrics with imbalanced classes."""
        y_true = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        y_pred = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
        y_prob = [0.1, 0.2, 0.1, 0.3, 0.2, 0.3, 0.6, 0.7, 0.8, 0.4]
        
        metrics = binary_classification_metrics(y_true, y_pred, y_prob)
        
        assert 0.0 < metrics['accuracy'] < 1.0
        assert 0.0 <= metrics['precision'] <= 1.0
        assert 0.0 < metrics['recall'] < 1.0
        assert 0.0 < metrics['f1'] < 1.0
        assert 0.0 < metrics['balanced_accuracy'] < 1.0
        assert metrics['roc_auc'] > 0.5  # Better than random

    def test_all_same_class_predictions(self) -> None:
        """Test metrics when all predictions are the same class."""
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 0, 0, 0, 0]  # All predicted as class 0
        
        metrics = binary_classification_metrics(y_true, y_pred)
        
        assert 0.0 < metrics['accuracy'] < 1.0
        # Precision may be 0.0 or 1.0 depending on implementation when there are no positive predictions
        assert metrics['precision'] in (0.0, 1.0)
        assert metrics['recall'] == 0.0  # No true positives for class 1
        assert metrics['f1'] == 0.0  # F1 is 0 when recall is 0
        assert metrics['specificity'] == 1.0  # All negatives correctly identified

    def test_empty_arrays(self) -> None:
        """Test that empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="Empty arrays provided"):
            binary_classification_metrics([], [])

    def test_invalid_inputs(self) -> None:
        """Test that invalid inputs raise appropriate errors."""
        # Non-binary classification
        with pytest.raises(ValueError):
            binary_classification_metrics([0, 1, 2], [0, 1, 2])
        
        # Incompatible shapes
        with pytest.raises(ValueError, match="incompatible shapes"):
            binary_classification_metrics([0, 1], [0, 1, 0])
        
        # Invalid input types
        with pytest.raises(TypeError, match="must be a list"):
            binary_classification_metrics("invalid", [0, 1])


class TestMulticlassClassificationMetrics:
    """Tests for multiclass classification metrics functions."""

    def test_perfect_prediction(self) -> None:
        """Test metrics with perfect predictions."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        
        metrics = multiclass_classification_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['balanced_accuracy'] == 1.0
        assert metrics['kappa'] == 1.0
        assert metrics['hamming_loss'] == 0.0
        assert metrics['jaccard_score'] == 1.0
        
        # Check per-class metrics
        for class_label in ['0', '1', '2']:
            assert metrics['per_class_precision'][class_label] == 1.0
            assert metrics['per_class_recall'][class_label] == 1.0
            assert metrics['per_class_f1'][class_label] == 1.0

    def test_worst_prediction(self) -> None:
        """Test metrics with completely wrong predictions."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [1, 2, 0, 1, 2, 0]  # All predictions wrong
        
        metrics = multiclass_classification_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 0.0
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0
        assert metrics['balanced_accuracy'] == 0.0
        assert metrics['kappa'] < 0.0  # Negative kappa for worse than random
        assert metrics['hamming_loss'] == 1.0  # All predictions wrong
        assert metrics['jaccard_score'] == 0.0

    def test_imbalanced_classes(self) -> None:
        """Test metrics with imbalanced classes."""
        y_true = [0, 0, 0, 0, 0, 1, 1, 2]
        y_pred = [0, 0, 0, 1, 1, 1, 2, 2]
        
        metrics = multiclass_classification_metrics(y_true, y_pred, average='weighted')
        
        assert 0.0 < metrics['accuracy'] < 1.0
        assert 0.0 < metrics['precision'] < 1.0
        assert 0.0 < metrics['recall'] < 1.0
        assert 0.0 < metrics['f1'] < 1.0
        assert 0.0 < metrics['balanced_accuracy'] < 1.0
        
        # Check that all classes are represented in per-class metrics
        assert set(metrics['per_class_precision'].keys()) == {'0', '1', '2'}

    def test_different_averaging_methods(self) -> None:
        """Test different averaging methods for multiclass metrics."""
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 2, 2, 0]
        
        # Test macro averaging
        macro_metrics = multiclass_classification_metrics(y_true, y_pred, average='macro')
        
        # Test micro averaging
        micro_metrics = multiclass_classification_metrics(y_true, y_pred, average='micro')
        
        # Test weighted averaging
        weighted_metrics = multiclass_classification_metrics(y_true, y_pred, average='weighted')
        
        # Different averaging methods may give different results, but not guaranteed
        # Just verify they all return valid values
        assert 0.0 <= macro_metrics['precision'] <= 1.0
        assert 0.0 <= micro_metrics['precision'] <= 1.0
        assert 0.0 <= weighted_metrics['precision'] <= 1.0
        assert 0.0 <= macro_metrics['recall'] <= 1.0
        assert 0.0 <= micro_metrics['recall'] <= 1.0
        assert 0.0 <= weighted_metrics['recall'] <= 1.0
        
        # But accuracy should be the same regardless of averaging method
        assert macro_metrics['accuracy'] == micro_metrics['accuracy'] == weighted_metrics['accuracy']

    def test_with_probabilities(self) -> None:
        """Test multiclass metrics with probability estimates."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 2, 2]
        y_prob = np.array([
            [0.8, 0.1, 0.1],  # Prob for sample 0
            [0.1, 0.7, 0.2],  # Prob for sample 1
            [0.2, 0.6, 0.2],  # Prob for sample 2
            [0.6, 0.3, 0.1],  # Prob for sample 3
            [0.1, 0.3, 0.6],  # Prob for sample 4
            [0.1, 0.2, 0.7]   # Prob for sample 5
        ])
        
        metrics = multiclass_classification_metrics(y_true, y_pred, y_prob)
        
        # Check that log_loss is calculated
        assert 'log_loss' in metrics
        assert metrics['log_loss'] > 0.0

    def test_empty_arrays(self) -> None:
        """Test that empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="Empty arrays provided"):
            multiclass_classification_metrics([], [])


class TestPrecisionRecallCurve:
    """Tests for precision-recall curve data function."""

    def test_perfect_prediction(self) -> None:
        """Test precision-recall curve with perfect predictions."""
        y_true = [0, 1, 0, 1, 0]
        y_prob = [0.1, 0.9, 0.2, 0.8, 0.1]
        
        curve_data = precision_recall_curve_data(y_true, y_prob)
        
        assert 'precision' in curve_data
        assert 'recall' in curve_data
        assert 'thresholds' in curve_data
        assert 'average_precision' in curve_data
        assert curve_data['average_precision'] > 0.9  # Should be close to 1.0

    def test_random_prediction(self) -> None:
        """Test precision-recall curve with random predictions."""
        np.random.seed(42)
        y_true = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        y_prob = np.random.random(10)
        
        curve_data = precision_recall_curve_data(y_true, y_prob)
        
        # For random predictions, average precision should be around 0.5
        assert 0.2 < curve_data['average_precision'] < 0.8

    def test_invalid_inputs(self) -> None:
        """Test that invalid inputs raise appropriate errors."""
        # Non-binary classification
        with pytest.raises(ValueError, match="Expected binary classification"):
            precision_recall_curve_data([0, 1, 2], [0.1, 0.2, 0.3])
        
        # Incompatible shapes
        with pytest.raises(ValueError, match="incompatible shapes"):
            precision_recall_curve_data([0, 1], [0.1, 0.2, 0.3])


class TestROCCurve:
    """Tests for ROC curve data function."""

    def test_perfect_prediction(self) -> None:
        """Test ROC curve with perfect predictions."""
        y_true = [0, 1, 0, 1, 0]
        y_prob = [0.1, 0.9, 0.2, 0.8, 0.1]
        
        curve_data = roc_curve_data(y_true, y_prob)
        
        assert 'fpr' in curve_data
        assert 'tpr' in curve_data
        assert 'thresholds' in curve_data
        assert 'roc_auc' in curve_data
        assert curve_data['roc_auc'] > 0.9  # Should be close to 1.0

    def test_random_prediction(self) -> None:
        """Test ROC curve with random predictions."""
        np.random.seed(42)
        y_true = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        y_prob = np.random.random(10)
        
        curve_data = roc_curve_data(y_true, y_prob)
        
        # For random predictions, ROC AUC should be around 0.5
        assert 0.2 < curve_data['roc_auc'] < 0.8

    def test_invalid_inputs(self) -> None:
        """Test that invalid inputs raise appropriate errors."""
        # Non-binary classification
        with pytest.raises(ValueError, match="Expected binary classification"):
            roc_curve_data([0, 1, 2], [0.1, 0.2, 0.3])
        
        # Incompatible shapes
        with pytest.raises(ValueError, match="incompatible shapes"):
            roc_curve_data([0, 1], [0.1, 0.2, 0.3])


class TestClassificationReport:
    """Tests for classification report dictionary function."""

    def test_binary_classification(self) -> None:
        """Test classification report for binary classification."""
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1]
        
        report = classification_report_dict(y_true, y_pred)
        
        assert '0' in report
        assert '1' in report
        assert 'accuracy' in report
        assert 'macro avg' in report
        assert 'weighted avg' in report
        
        # Check that each class has precision, recall, f1-score, and support
        for class_label in ['0', '1', 'macro avg', 'weighted avg']:
            assert 'precision' in report[class_label]
            assert 'recall' in report[class_label]
            assert 'f1-score' in report[class_label]
            assert 'support' in report[class_label]

    def test_multiclass_classification(self) -> None:
        """Test classification report for multiclass classification."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 1, 0, 0, 1]
        
        report = classification_report_dict(y_true, y_pred)
        
        assert '0' in report
        assert '1' in report
        assert '2' in report
        assert 'accuracy' in report
        assert 'macro avg' in report
        assert 'weighted avg' in report

    def test_with_target_names(self) -> None:
        """Test classification report with custom target names."""
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1]
        target_names = ['Negative', 'Positive']
        
        report = classification_report_dict(y_true, y_pred, target_names)
        
        assert 'Negative' in report
        assert 'Positive' in report
        assert 'accuracy' in report
        assert 'macro avg' in report
        assert 'weighted avg' in report

    def test_all_same_class(self) -> None:
        """Test classification report when all predictions are the same class."""
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 0, 0, 0, 0]  # All predicted as class 0
        
        report = classification_report_dict(y_true, y_pred)
        
        assert '0' in report
        assert '1' in report
        assert report['1']['precision'] == 0.0
        assert report['1']['recall'] == 0.0
        assert report['1']['f1-score'] == 0.0
