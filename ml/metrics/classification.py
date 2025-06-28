"""
Classification metrics utilities for machine learning models.

This module provides specialized functions for calculating various performance metrics
for classification models, including binary and multiclass classification scenarios.
"""

__author__ = "Usman Ahmad"

from typing import Dict, List, Union, Optional, Tuple, Any, cast
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score,
    fbeta_score, hamming_loss, jaccard_score, log_loss
)
import logging
from ml.metrics.base import classification_metrics

# Configure logging
logger = logging.getLogger(__name__)


def binary_classification_metrics(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    y_prob: Optional[Union[List, np.ndarray, pd.Series]] = None,
    pos_label: int = 1,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for binary classification.
    
    This function extends the base classification_metrics with additional metrics
    specific to binary classification problems.
    
    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a classifier.
        y_prob: Probability estimates for the positive class. Required for ROC AUC
            and precision-recall curve metrics. If None, these metrics will not be calculated.
        pos_label: The label of the positive class (default: 1).
        threshold: Decision threshold for converting probabilities to binary predictions (default: 0.5).
        
    Returns:
        Dictionary containing various binary classification metrics:
            - All metrics from classification_metrics
            - specificity: True negative rate
            - balanced_accuracy: Balanced accuracy (average of sensitivity and specificity)
            - matthews_correlation: Matthews correlation coefficient
            - kappa: Cohen's kappa score
            - f2: F2 score (emphasizes recall over precision)
            - f0.5: F0.5 score (emphasizes precision over recall)
            - average_precision: Area under the precision-recall curve
            - pos_pred_value: Positive predictive value (same as precision)
            - neg_pred_value: Negative predictive value
            
    Raises:
        TypeError: If input types are invalid.
        ValueError: If inputs have incompatible shapes or if not a binary classification problem.
    """
    # First get base metrics
    metrics = classification_metrics(y_true, y_pred, y_prob, average='binary')
    
    # Convert inputs to numpy arrays for consistency
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    # Check if binary classification
    unique_classes = np.unique(y_true_array)
    if len(unique_classes) != 2:
        logger.error(f"ValueError: Expected binary classification problem, got {len(unique_classes)} classes")
        raise ValueError(f"Expected binary classification problem, got {len(unique_classes)} classes")
    
    try:
        # Calculate confusion matrix elements
        cm = confusion_matrix(y_true_array, y_pred_array)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate additional metrics
        # Specificity (true negative rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['specificity'] = float(specificity)
        
        # Balanced accuracy
        metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true_array, y_pred_array))
        
        # Matthews correlation coefficient
        metrics['matthews_correlation'] = float(matthews_corrcoef(y_true_array, y_pred_array))
        
        # Cohen's kappa
        metrics['kappa'] = float(cohen_kappa_score(y_true_array, y_pred_array))
        
        # F-beta scores
        metrics['f2'] = float(fbeta_score(y_true_array, y_pred_array, beta=2, zero_division=0))
        metrics['f0.5'] = float(fbeta_score(y_true_array, y_pred_array, beta=0.5, zero_division=0))
        
        # Negative predictive value
        metrics['neg_pred_value'] = float(tn / (tn + fn) if (tn + fn) > 0 else 0.0)
        
        # Positive predictive value (same as precision, but included for completeness)
        metrics['pos_pred_value'] = float(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        
        # Calculate probability-based metrics if probabilities are provided
        if y_prob is not None:
            y_prob_array = np.array(y_prob)
            
            # Average precision score (area under precision-recall curve)
            metrics['average_precision'] = float(average_precision_score(y_true_array, y_prob_array))
            
            # Log loss
            metrics['log_loss'] = float(log_loss(y_true_array, y_prob_array))
            
        # Log additional metrics
        logger.info(f"Binary classification metrics calculated: specificity={metrics['specificity']:.4f}, "
                   f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
                   f"matthews_correlation={metrics['matthews_correlation']:.4f}")
        
    except Exception as e:
        logger.error(f"Error calculating binary classification metrics: {str(e)}")
        raise
    
    return metrics


def multiclass_classification_metrics(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    y_prob: Optional[Union[List, np.ndarray, pd.Series]] = None,
    average: str = 'macro'
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for multiclass classification.
    
    This function extends the base classification_metrics with additional metrics
    specific to multiclass classification problems.
    
    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a classifier.
        y_prob: Probability estimates for each class. If provided, should be a 2D array
            with shape (n_samples, n_classes).
        average: Parameter for averaging metrics. Options: 'micro', 'macro', 'weighted', 'samples'.
            Default is 'macro'.
        
    Returns:
        Dictionary containing various multiclass classification metrics:
            - All metrics from classification_metrics
            - balanced_accuracy: Balanced accuracy
            - kappa: Cohen's kappa score
            - hamming_loss: Hamming loss
            - jaccard_score: Jaccard similarity coefficient
            - per_class_precision: Precision for each class
            - per_class_recall: Recall for each class
            - per_class_f1: F1 score for each class
            
    Raises:
        TypeError: If input types are invalid.
        ValueError: If inputs have incompatible shapes.
    """
    # First get base metrics
    metrics = classification_metrics(y_true, y_pred, None, average=average)
    
    # Convert inputs to numpy arrays for consistency
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    # Get unique classes
    unique_classes = np.unique(np.concatenate((y_true_array, y_pred_array)))
    n_classes = len(unique_classes)
    
    try:
        # Calculate additional metrics
        # Balanced accuracy
        metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true_array, y_pred_array))
        
        # Cohen's kappa
        metrics['kappa'] = float(cohen_kappa_score(y_true_array, y_pred_array))
        
        # Hamming loss
        metrics['hamming_loss'] = float(hamming_loss(y_true_array, y_pred_array))
        
        # Jaccard score
        metrics['jaccard_score'] = float(jaccard_score(y_true_array, y_pred_array, average=average, zero_division=0))
        
        # Per-class metrics
        per_class_precision = precision_score(y_true_array, y_pred_array, average=None, zero_division=0)
        per_class_recall = recall_score(y_true_array, y_pred_array, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true_array, y_pred_array, average=None, zero_division=0)
        
        # Create dictionaries for per-class metrics
        metrics['per_class_precision'] = {str(unique_classes[i]): float(per_class_precision[i]) 
                                         for i in range(len(unique_classes))}
        metrics['per_class_recall'] = {str(unique_classes[i]): float(per_class_recall[i]) 
                                      for i in range(len(unique_classes))}
        metrics['per_class_f1'] = {str(unique_classes[i]): float(per_class_f1[i]) 
                                  for i in range(len(unique_classes))}
        
        # Calculate probability-based metrics if probabilities are provided
        if y_prob is not None:
            y_prob_array = np.array(y_prob)
            
            # Check shape of y_prob
            if y_prob_array.shape[1] != n_classes:
                logger.warning(f"y_prob shape {y_prob_array.shape} does not match number of classes {n_classes}")
            else:
                # Log loss
                metrics['log_loss'] = float(log_loss(y_true_array, y_prob_array))
        
        # Log additional metrics
        logger.info(f"Multiclass classification metrics calculated: balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
                   f"kappa={metrics['kappa']:.4f}, hamming_loss={metrics['hamming_loss']:.4f}")
        
    except Exception as e:
        logger.error(f"Error calculating multiclass classification metrics: {str(e)}")
        raise
    
    return metrics


def precision_recall_curve_data(
    y_true: Union[List, np.ndarray, pd.Series],
    y_prob: Union[List, np.ndarray, pd.Series],
    pos_label: int = 1
) -> Dict[str, np.ndarray]:
    """
    Calculate precision-recall curve data for binary classification.
    
    Args:
        y_true: Ground truth (correct) target values.
        y_prob: Probability estimates for the positive class.
        pos_label: The label of the positive class (default: 1).
        
    Returns:
        Dictionary containing precision-recall curve data:
            - precision: Precision values
            - recall: Recall values
            - thresholds: Thresholds used to compute precision and recall
            - average_precision: Average precision score
            
    Raises:
        TypeError: If input types are invalid.
        ValueError: If inputs have incompatible shapes or if not a binary classification problem.
    """
    # Validate inputs
    if not isinstance(y_true, (list, np.ndarray, pd.Series)):
        logger.error("TypeError: y_true must be a list, numpy array, or pandas Series")
        raise TypeError("y_true must be a list, numpy array, or pandas Series")
    
    if not isinstance(y_prob, (list, np.ndarray, pd.Series)):
        logger.error("TypeError: y_prob must be a list, numpy array, or pandas Series")
        raise TypeError("y_prob must be a list, numpy array, or pandas Series")
    
    # Convert inputs to numpy arrays for consistency
    y_true_array = np.array(y_true)
    y_prob_array = np.array(y_prob)
    
    # Check shapes
    if len(y_true_array) != len(y_prob_array):
        logger.error(f"ValueError: y_true and y_prob have incompatible shapes: "
                    f"{y_true_array.shape} vs {y_prob_array.shape}")
        raise ValueError(f"y_true and y_prob have incompatible shapes: "
                        f"{y_true_array.shape} vs {y_prob_array.shape}")
    
    # Check if binary classification
    unique_classes = np.unique(y_true_array)
    if len(unique_classes) != 2:
        logger.error(f"ValueError: Expected binary classification problem, got {len(unique_classes)} classes")
        raise ValueError(f"Expected binary classification problem, got {len(unique_classes)} classes")
    
    try:
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true_array, y_prob_array, pos_label=pos_label)
        
        # Calculate average precision
        average_precision = average_precision_score(y_true_array, y_prob_array, pos_label=pos_label)
        
        # Create result dictionary
        result = {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'average_precision': float(average_precision)
        }
        
        logger.info(f"Precision-recall curve calculated with {len(thresholds)} threshold points, "
                   f"average precision: {average_precision:.4f}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error calculating precision-recall curve: {str(e)}")
        raise


def roc_curve_data(
    y_true: Union[List, np.ndarray, pd.Series],
    y_prob: Union[List, np.ndarray, pd.Series],
    pos_label: int = 1
) -> Dict[str, np.ndarray]:
    """
    Calculate ROC curve data for binary classification.
    
    Args:
        y_true: Ground truth (correct) target values.
        y_prob: Probability estimates for the positive class.
        pos_label: The label of the positive class (default: 1).
        
    Returns:
        Dictionary containing ROC curve data:
            - fpr: False positive rate
            - tpr: True positive rate
            - thresholds: Thresholds used to compute TPR and FPR
            - roc_auc: Area under the ROC curve
            
    Raises:
        TypeError: If input types are invalid.
        ValueError: If inputs have incompatible shapes or if not a binary classification problem.
    """
    # Validate inputs
    if not isinstance(y_true, (list, np.ndarray, pd.Series)):
        logger.error("TypeError: y_true must be a list, numpy array, or pandas Series")
        raise TypeError("y_true must be a list, numpy array, or pandas Series")
    
    if not isinstance(y_prob, (list, np.ndarray, pd.Series)):
        logger.error("TypeError: y_prob must be a list, numpy array, or pandas Series")
        raise TypeError("y_prob must be a list, numpy array, or pandas Series")
    
    # Convert inputs to numpy arrays for consistency
    y_true_array = np.array(y_true)
    y_prob_array = np.array(y_prob)
    
    # Check shapes
    if len(y_true_array) != len(y_prob_array):
        logger.error(f"ValueError: y_true and y_prob have incompatible shapes: "
                    f"{y_true_array.shape} vs {y_prob_array.shape}")
        raise ValueError(f"y_true and y_prob have incompatible shapes: "
                        f"{y_true_array.shape} vs {y_prob_array.shape}")
    
    # Check if binary classification
    unique_classes = np.unique(y_true_array)
    if len(unique_classes) != 2:
        logger.error(f"ValueError: Expected binary classification problem, got {len(unique_classes)} classes")
        raise ValueError(f"Expected binary classification problem, got {len(unique_classes)} classes")
    
    try:
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_array, y_prob_array, pos_label=pos_label)
        
        # Calculate AUC
        roc_auc = auc(fpr, tpr)
        
        # Create result dictionary
        result = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'roc_auc': float(roc_auc)
        }
        
        logger.info(f"ROC curve calculated with {len(thresholds)} threshold points, "
                   f"ROC AUC: {roc_auc:.4f}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error calculating ROC curve: {str(e)}")
        raise


def classification_report_dict(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate a classification report as a dictionary.
    
    This is a wrapper around sklearn's classification_report with output as a dictionary
    instead of a string, with additional error handling.
    
    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a classifier.
        target_names: Display names for the classes (optional).
        
    Returns:
        Dictionary containing classification report data:
            - For each class: precision, recall, f1-score, support
            - Macro avg: precision, recall, f1-score, support
            - Weighted avg: precision, recall, f1-score, support
            
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
    
    try:
        # Generate classification report as dictionary
        report = classification_report(y_true_array, y_pred_array, 
                                      target_names=target_names, 
                                      output_dict=True,
                                      zero_division=0)
        
        # Log report generation
        logger.info(f"Classification report generated with {len(report) - 3} classes")
        
        return report
    
    except Exception as e:
        logger.error(f"Error generating classification report: {str(e)}")
        raise
