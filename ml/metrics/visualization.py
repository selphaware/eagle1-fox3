"""
Visualization utilities for regression metrics.

This module provides functions for visualizing regression model performance
through various plots and charts.
"""

from typing import Dict, List, Union, Optional, Tuple, Any, Callable
import numpy as np
import pandas as pd
import logging

# Use try-except for visualization libraries to avoid breaking tests if not installed
try:
    # Set non-interactive backend to avoid requiring a GUI
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    # Define Figure type for type hints even if matplotlib is not available
    class Figure:  # type: ignore
        """Placeholder for matplotlib Figure class when matplotlib is not available."""
        pass

# Configure logging
logger = logging.getLogger(__name__)


def plot_residuals(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    title: str = "Residual Plot",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Figure:
    """Create a residual plot for regression model evaluation.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a regressor.
        title: Title for the plot.
        figsize: Figure size as (width, height) in inches.
        save_path: Path to save the figure. If None, figure is not saved.

    Returns:
        Matplotlib Figure object.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If inputs have incompatible shapes.
        ImportError: If matplotlib is not available.
    """
    if not VISUALIZATION_AVAILABLE:
        logger.error("Matplotlib and/or seaborn is not available. Cannot create plot.")
        raise ImportError("Matplotlib and/or seaborn is not available. Cannot create plot.")
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

    # Calculate residuals
    residuals = y_true_array - y_pred_array

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create scatter plot of predicted values vs residuals
    ax.scatter(y_pred_array, residuals, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)

    # Add labels and title
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title(title)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Save figure if path is provided
    if save_path:
        try:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Residual plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving residual plot: {str(e)}")

    return fig


def plot_actual_vs_predicted(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    title: str = "Actual vs Predicted Values",
    figsize: Tuple[int, int] = (10, 6),
    identity_line: bool = True,
    save_path: Optional[str] = None
) -> Figure:
    """Create a scatter plot of actual vs predicted values.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a regressor.
        title: Title for the plot.
        figsize: Figure size as (width, height) in inches.
        identity_line: Whether to draw the identity line (y=x).
        save_path: Path to save the figure. If None, figure is not saved.

    Returns:
        Matplotlib Figure object.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If inputs have incompatible shapes.
        ImportError: If matplotlib is not available.
    """
    if not VISUALIZATION_AVAILABLE:
        logger.error("Matplotlib and/or seaborn is not available. Cannot create plot.")
        raise ImportError("Matplotlib and/or seaborn is not available. Cannot create plot.")
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

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create scatter plot
    ax.scatter(y_true_array, y_pred_array, alpha=0.6)

    # Add identity line if requested
    if identity_line:
        min_val = min(np.min(y_true_array), np.min(y_pred_array))
        max_val = max(np.max(y_true_array), np.max(y_pred_array))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

    # Add labels and title
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Save figure if path is provided
    if save_path:
        try:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Actual vs predicted plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving actual vs predicted plot: {str(e)}")

    return fig


def plot_prediction_error_distribution(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    title: str = "Prediction Error Distribution",
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 30,
    kde: bool = True,
    save_path: Optional[str] = None
) -> Figure:
    """Create a histogram of prediction errors (residuals).

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a regressor.
        title: Title for the plot.
        figsize: Figure size as (width, height) in inches.
        bins: Number of bins for the histogram.
        kde: Whether to draw a kernel density estimate.
        save_path: Path to save the figure. If None, figure is not saved.

    Returns:
        Matplotlib Figure object.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If inputs have incompatible shapes.
        ImportError: If matplotlib is not available.
    """
    if not VISUALIZATION_AVAILABLE:
        logger.error("Matplotlib and/or seaborn is not available. Cannot create plot.")
        raise ImportError("Matplotlib and/or seaborn is not available. Cannot create plot.")
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

    # Calculate errors
    errors = y_true_array - y_pred_array

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create histogram with KDE
    sns.histplot(errors, bins=bins, kde=kde, ax=ax)

    # Add vertical line at zero
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)

    # Add labels and title
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Frequency")
    ax.set_title(title)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Save figure if path is provided
    if save_path:
        try:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Error distribution plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving error distribution plot: {str(e)}")

    return fig


def plot_metrics_comparison(
    metrics_list: List[Dict[str, float]],
    model_names: List[str],
    metrics_to_plot: Optional[List[str]] = None,
    title: str = "Model Performance Comparison",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> Figure:
    """Create a bar chart comparing metrics across different models.

    Args:
        metrics_list: List of metrics dictionaries, one per model.
        model_names: List of model names corresponding to metrics_list.
        metrics_to_plot: List of metric names to include in the plot.
                        If None, plots common metrics across all models.
        title: Title for the plot.
        figsize: Figure size as (width, height) in inches.
        save_path: Path to save the figure. If None, figure is not saved.

    Returns:
        Matplotlib Figure object.

    Raises:
        ValueError: If inputs have incompatible shapes.
        ImportError: If matplotlib is not available.
    """
    if not VISUALIZATION_AVAILABLE:
        logger.error("Matplotlib and/or seaborn is not available. Cannot create plot.")
        raise ImportError("Matplotlib and/or seaborn is not available. Cannot create plot.")
    # Validate inputs
    if len(metrics_list) != len(model_names):
        raise ValueError("metrics_list and model_names must have the same length")

    if len(metrics_list) == 0:
        raise ValueError("metrics_list cannot be empty")

    # Determine which metrics to plot
    if metrics_to_plot is None:
        # Find common metrics across all models
        common_metrics = set(metrics_list[0].keys())
        for metrics in metrics_list[1:]:
            common_metrics = common_metrics.intersection(set(metrics.keys()))
        metrics_to_plot = list(common_metrics)

    # Filter out metrics that don't exist in all models
    valid_metrics = []
    for metric in metrics_to_plot:
        if all(metric in metrics for metrics in metrics_list):
            valid_metrics.append(metric)

    if not valid_metrics:
        raise ValueError("No common metrics found across all models")

    # Create DataFrame for plotting
    data = []
    for i, metrics in enumerate(metrics_list):
        for metric in valid_metrics:
            data.append({
                'Model': model_names[i],
                'Metric': metric,
                'Value': metrics[metric]
            })
    df = pd.DataFrame(data)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create grouped bar chart
    sns.barplot(x='Metric', y='Value', hue='Model', data=df, ax=ax)

    # Add labels and title
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title(title)

    # Rotate x-axis labels if there are many metrics
    if len(valid_metrics) > 4:
        plt.xticks(rotation=45, ha='right')

    # Adjust layout
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        try:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Metrics comparison plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving metrics comparison plot: {str(e)}")

    return fig


def plot_prediction_intervals(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    y_std: Union[List, np.ndarray, pd.Series],
    confidence_level: float = 0.95,
    title: str = "Predictions with Confidence Intervals",
    figsize: Tuple[int, int] = (12, 6),
    max_points: int = 100,
    save_path: Optional[str] = None
) -> Figure:
    """Plot predictions with confidence intervals.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a regressor.
        y_std: Standard deviation of predictions.
        confidence_level: Confidence level for intervals (between 0 and 1).
        title: Title for the plot.
        figsize: Figure size as (width, height) in inches.
        max_points: Maximum number of points to plot (to avoid overcrowding).
        save_path: Path to save the figure. If None, figure is not saved.

    Returns:
        Matplotlib Figure object.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If inputs have incompatible shapes.
        ImportError: If matplotlib is not available.
    """
    if not VISUALIZATION_AVAILABLE:
        logger.error("Matplotlib and/or seaborn is not available. Cannot create plot.")
        raise ImportError("Matplotlib and/or seaborn is not available. Cannot create plot.")
    # Validate inputs
    if not isinstance(y_true, (list, np.ndarray, pd.Series)):
        raise TypeError("y_true must be a list, numpy array, or pandas Series")

    if not isinstance(y_pred, (list, np.ndarray, pd.Series)):
        raise TypeError("y_pred must be a list, numpy array, or pandas Series")

    if not isinstance(y_std, (list, np.ndarray, pd.Series)):
        raise TypeError("y_std must be a list, numpy array, or pandas Series")

    # Convert inputs to numpy arrays for consistency
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    y_std_array = np.array(y_std)

    # Check shapes
    if y_true_array.shape != y_pred_array.shape or y_true_array.shape != y_std_array.shape:
        raise ValueError("y_true, y_pred, and y_std must have the same shape")

    # Check confidence level
    if not 0 < confidence_level < 1:
        raise ValueError(f"confidence_level must be between 0 and 1, got {confidence_level}")

    # Calculate z-score for the confidence level
    from scipy.stats import norm
    z_score = norm.ppf((1 + confidence_level) / 2)

    # Calculate confidence intervals
    lower_bound = y_pred_array - z_score * y_std_array
    upper_bound = y_pred_array + z_score * y_std_array

    # Sample points if there are too many
    n_samples = len(y_true_array)
    if n_samples > max_points:
        indices = np.random.choice(n_samples, max_points, replace=False)
        indices = np.sort(indices)  # Sort for better visualization
        x = np.arange(len(indices))
        y_true_plot = y_true_array[indices]
        y_pred_plot = y_pred_array[indices]
        lower_bound_plot = lower_bound[indices]
        upper_bound_plot = upper_bound[indices]
    else:
        x = np.arange(n_samples)
        y_true_plot = y_true_array
        y_pred_plot = y_pred_array
        lower_bound_plot = lower_bound
        upper_bound_plot = upper_bound

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot actual values
    ax.scatter(x, y_true_plot, color='blue', alpha=0.6, label='Actual')

    # Plot predicted values with confidence intervals
    ax.plot(x, y_pred_plot, color='red', label='Predicted')
    ax.fill_between(x, lower_bound_plot, upper_bound_plot, color='red', alpha=0.2,
                   label=f'{int(confidence_level*100)}% Confidence Interval')

    # Add labels and title
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Value")
    ax.set_title(title)

    # Add legend
    ax.legend()

    # Add grid
    ax.grid(True, alpha=0.3)

    # Save figure if path is provided
    if save_path:
        try:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Prediction intervals plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving prediction intervals plot: {str(e)}")

    return fig
