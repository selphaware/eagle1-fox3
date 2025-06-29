"""
K-Means clustering implementation.

This module provides a wrapper around scikit-learn's KMeans implementation
with additional functionality for optimal k selection and evaluation.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, cast
import numpy as np
import pandas as pd
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Configure logging
logger = logging.getLogger(__name__)


class KMeansModel:
    """
    K-Means clustering model with additional functionality.
    
    This class wraps scikit-learn's KMeans implementation and provides
    additional methods for optimal k selection, evaluation, and visualization.
    
    Attributes:
        model: The underlying scikit-learn KMeans model.
        k: The number of clusters.
        random_state: Random seed for reproducibility.
        scaler: Optional StandardScaler for feature standardization.
        cluster_centers_: Cluster centers after fitting.
        labels_: Cluster labels after fitting.
        inertia_: Sum of squared distances after fitting.
    """
    
    def __init__(
        self,
        k: int = 3,
        random_state: Optional[int] = None,
        standardize: bool = True,
        n_init: int = 10,
        **kwargs: Any
    ) -> None:
        """
        Initialize the KMeansModel.
        
        Args:
            k: Number of clusters.
            random_state: Random seed for reproducibility.
            standardize: Whether to standardize features before clustering.
            **kwargs: Additional arguments to pass to sklearn.cluster.KMeans.
        
        Raises:
            ValueError: If k is less than 2.
        """
        if k < 2:
            error_msg = f"Number of clusters must be at least 2, got {k}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        self.k = k
        self.random_state = random_state
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        
        # Initialize the underlying KMeans model
        self.model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
            **kwargs
        )
        
        # These will be set after fitting
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        
        logger.info(f"Initialized KMeansModel with k={k}, standardize={standardize}, n_init={n_init}")
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> 'KMeansModel':
        """
        Fit the K-Means model to the data.
        
        Args:
            X: Input features as DataFrame or ndarray.
            
        Returns:
            Self for method chaining.
            
        Raises:
            ValueError: If X is empty or contains NaN values.
            TypeError: If X is not a DataFrame or ndarray.
        """
        # Input validation
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            error_msg = f"X must be a pandas DataFrame or numpy ndarray, got {type(X)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
            
        if isinstance(X, pd.DataFrame):
            if X.empty:
                error_msg = "X cannot be empty"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if X.isna().any().any():
                error_msg = "X contains NaN values"
                logger.error(error_msg)
                raise ValueError(error_msg)
            # Convert DataFrame to numpy array for processing
            X_array = X.values
        else:  # numpy ndarray
            if X.size == 0:
                error_msg = "X cannot be empty"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if np.isnan(X).any():
                error_msg = "X contains NaN values"
                logger.error(error_msg)
                raise ValueError(error_msg)
            X_array = X
        
        # Standardize if requested
        if self.standardize and self.scaler is not None:
            X_processed = self.scaler.fit_transform(X_array)
            logger.info("Data standardized before clustering")
        else:
            X_processed = X_array
        
        # Fit the model
        self.model.fit(X_processed)
        
        # Store results
        self.cluster_centers_ = self.model.cluster_centers_
        self.labels_ = self.model.labels_
        self.inertia_ = self.model.inertia_
        
        logger.info(f"KMeansModel fitted successfully with inertia: {self.inertia_:.4f}")
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Input features as DataFrame or ndarray.
            
        Returns:
            Array of cluster labels.
            
        Raises:
            ValueError: If model is not fitted or X contains NaN values.
            TypeError: If X is not a DataFrame or ndarray.
        """
        # Check if model is fitted
        if self.cluster_centers_ is None:
            error_msg = "Model must be fitted before prediction"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Input validation
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            error_msg = f"X must be a pandas DataFrame or numpy ndarray, got {type(X)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
            
        if isinstance(X, pd.DataFrame):
            if X.empty:
                error_msg = "X cannot be empty"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if X.isna().any().any():
                error_msg = "X contains NaN values"
                logger.error(error_msg)
                raise ValueError(error_msg)
            # Convert DataFrame to numpy array for processing
            X_array = X.values
        else:  # numpy ndarray
            if X.size == 0:
                error_msg = "X cannot be empty"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if np.isnan(X).any():
                error_msg = "X contains NaN values"
                logger.error(error_msg)
                raise ValueError(error_msg)
            X_array = X
        
        # Standardize if requested
        if self.standardize and self.scaler is not None:
            X_processed = self.scaler.transform(X_array)
        else:
            X_processed = X_array
        
        # Predict
        labels = self.model.predict(X_processed)
        logger.info(f"Predicted cluster labels for {len(X_processed)} samples")
        return labels
    
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate the clustering performance.
        
        Args:
            X: Input features as DataFrame or ndarray.
            
        Returns:
            Dictionary with evaluation metrics.
            
        Raises:
            ValueError: If model is not fitted or X contains NaN values.
            TypeError: If X is not a DataFrame or ndarray.
        """
        # Check if model is fitted
        if self.cluster_centers_ is None:
            error_msg = "Model must be fitted before evaluation"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Input validation (reuse from predict method)
        labels = self.predict(X)
        
        # Prepare data for evaluation
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        if self.standardize and self.scaler is not None:
            X_processed = self.scaler.transform(X_array)
        else:
            X_processed = X_array
        
        # Calculate metrics
        metrics = {}
        
        # Inertia (within-cluster sum of squares)
        metrics['inertia'] = self.inertia_
        
        # Silhouette score (only if we have more than one cluster and enough samples)
        if self.k > 1 and len(X_processed) > self.k:
            try:
                silhouette = silhouette_score(X_processed, labels)
                metrics['silhouette_score'] = silhouette
                logger.info(f"Silhouette score: {silhouette:.4f}")
            except Exception as e:
                logger.warning(f"Could not calculate silhouette score: {str(e)}")
                metrics['silhouette_score'] = None
        else:
            metrics['silhouette_score'] = None
            logger.warning("Silhouette score not calculated: requires k > 1 and enough samples")
        
        # Number of samples in each cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        metrics['cluster_sizes'] = dict(zip(unique_labels, counts))
        
        logger.info(f"Clustering evaluation completed with metrics: {metrics}")
        return metrics
    
    @staticmethod
    def find_optimal_k(
        X: Union[pd.DataFrame, np.ndarray],
        k_range: List[int],
        random_state: Optional[int] = None,
        standardize: bool = True,
        max_iter: int = 300,
        n_init: int = 10,
        method: str = 'elbow'
    ) -> Dict[str, Union[int, Dict[int, float]]]:
        """
        Find the optimal number of clusters using the elbow method or silhouette score.
        
        Args:
            X: Input features as DataFrame or ndarray.
            k_range: List of k values to try.
            random_state: Random seed for reproducibility.
            standardize: Whether to standardize features before clustering.
            max_iter: Maximum number of iterations for K-Means.
            n_init: Number of initializations for K-Means.
            method: Method to use for finding optimal k ('elbow' or 'silhouette').
            
        Returns:
            Dictionary with optimal k and metrics for each k.
            
        Raises:
            ValueError: If k_range is empty or contains invalid values.
            TypeError: If X is not a DataFrame or ndarray.
        """
        # Input validation
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            error_msg = f"X must be a pandas DataFrame or numpy ndarray, got {type(X)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
            
        if not k_range or min(k_range) < 2:
            error_msg = "k_range must contain values >= 2"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if method not in ['elbow', 'silhouette']:
            error_msg = f"method must be 'elbow' or 'silhouette', got {method}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Prepare data
        if isinstance(X, pd.DataFrame):
            if X.empty:
                error_msg = "X cannot be empty"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if X.isna().any().any():
                error_msg = "X contains NaN values"
                logger.error(error_msg)
                raise ValueError(error_msg)
            X_array = X.values
        else:  # numpy ndarray
            if X.size == 0:
                error_msg = "X cannot be empty"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if np.isnan(X).any():
                error_msg = "X contains NaN values"
                logger.error(error_msg)
                raise ValueError(error_msg)
            X_array = X
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_array)
            logger.info("Data standardized before optimal k selection")
        else:
            X_processed = X_array
        
        # Initialize results
        inertia_values = {}
        silhouette_values = {}
        
        # Try different k values
        for k in k_range:
            logger.info(f"Trying k={k}")
            
            # Fit KMeans
            kmeans = KMeans(
                n_clusters=k,
                random_state=random_state,
                max_iter=max_iter,
                n_init=n_init  # Explicitly set to avoid FutureWarning
            )
            labels = kmeans.fit_predict(X_processed)
            inertia_values[k] = kmeans.inertia_
            
            # Calculate silhouette score if requested and possible
            if method == 'silhouette' and k > 1 and len(X_processed) > k:
                try:
                    silhouette_values[k] = silhouette_score(X_processed, labels)
                except Exception as e:
                    logger.warning(f"Could not calculate silhouette score for k={k}: {str(e)}")
                    silhouette_values[k] = None
        
        # Determine optimal k
        if method == 'elbow':
            # Use the elbow method (simple heuristic)
            # Calculate the rate of decrease in inertia
            k_list = sorted(inertia_values.keys())
            if len(k_list) <= 1:
                optimal_k = k_list[0]
            else:
                # Calculate the rate of decrease in inertia
                rates = []
                for i in range(1, len(k_list)):
                    prev_k = k_list[i-1]
                    curr_k = k_list[i]
                    rate = (inertia_values[prev_k] - inertia_values[curr_k]) / inertia_values[prev_k]
                    rates.append(rate)
                
                # Find the elbow point (where the rate of decrease slows down)
                # Simple approach: find the k where the rate of decrease is less than half the max rate
                if rates:
                    max_rate = max(rates)
                    threshold = max_rate * 0.5
                    for i, rate in enumerate(rates):
                        if rate < threshold:
                            optimal_k = k_list[i]
                            break
                    else:
                        # If no elbow is found, use the last k
                        optimal_k = k_list[-1]
                else:
                    optimal_k = k_list[0]
            
            logger.info(f"Optimal k determined by elbow method: {optimal_k}")
            return {
                'optimal_k': optimal_k,
                'inertia_values': inertia_values
            }
        else:  # silhouette method
            # Use the silhouette method (maximize silhouette score)
            valid_scores = {k: score for k, score in silhouette_values.items() if score is not None}
            if valid_scores:
                optimal_k = max(valid_scores.items(), key=lambda x: x[1])[0]
                logger.info(f"Optimal k determined by silhouette method: {optimal_k}")
                return {
                    'optimal_k': optimal_k,
                    'silhouette_values': silhouette_values
                }
            else:
                # Fall back to elbow method if no valid silhouette scores
                logger.warning("No valid silhouette scores, falling back to elbow method")
                return KMeansModel.find_optimal_k(
                    X=X,
                    k_range=k_range,
                    random_state=random_state,
                    standardize=standardize,
                    max_iter=max_iter,
                    n_init=n_init,
                    method='elbow'
                )
    
    def plot_clusters(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_indices: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (10, 8),
        title: str = 'K-Means Clustering Results',
        show_centers: bool = True
    ) -> plt.Figure:
        """
        Plot the clustering results for visualization.
        
        Args:
            X: Input features as DataFrame or ndarray.
            feature_indices: Indices of features to plot (for high-dimensional data).
                If None, uses first two features. If 3 indices, creates a 3D plot.
            figsize: Figure size as (width, height).
            title: Plot title.
            show_centers: Whether to show cluster centers.
            
        Returns:
            Matplotlib figure object.
            
        Raises:
            ValueError: If model is not fitted or feature_indices are invalid.
            TypeError: If X is not a DataFrame or ndarray.
        """
        # Check if model is fitted
        if self.cluster_centers_ is None:
            error_msg = "Model must be fitted before plotting"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Get labels for the data
        labels = self.predict(X)
        
        # Prepare data for plotting
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns
        else:
            X_array = X
            feature_names = [f"Feature {i}" for i in range(X_array.shape[1])]
        
        # Apply standardization if used during fitting
        if self.standardize and self.scaler is not None:
            X_plot = self.scaler.transform(X_array)
        else:
            X_plot = X_array
        
        # Select features to plot
        if feature_indices is None:
            if X_plot.shape[1] >= 2:
                feature_indices = [0, 1]
            else:
                error_msg = "Data must have at least 2 features for plotting"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Validate feature indices
        if max(feature_indices) >= X_plot.shape[1] or min(feature_indices) < 0:
            error_msg = f"Feature indices {feature_indices} out of range for data with {X_plot.shape[1]} features"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # 2D or 3D plot based on number of selected features
        if len(feature_indices) == 2:
            ax = fig.add_subplot(111)
            
            # Plot data points colored by cluster
            scatter = ax.scatter(
                X_plot[:, feature_indices[0]],
                X_plot[:, feature_indices[1]],
                c=labels,
                cmap='viridis',
                alpha=0.7,
                s=50
            )
            
            # Plot cluster centers if requested
            if show_centers and self.cluster_centers_ is not None:
                centers = ax.scatter(
                    self.cluster_centers_[:, feature_indices[0]],
                    self.cluster_centers_[:, feature_indices[1]],
                    c='red',
                    marker='X',
                    s=200,
                    alpha=1.0,
                    label='Cluster Centers'
                )
                ax.legend()
            
            # Add colorbar
            plt.colorbar(scatter, label='Cluster')
            
            # Set labels and title
            ax.set_xlabel(feature_names[feature_indices[0]])
            ax.set_ylabel(feature_names[feature_indices[1]])
            
        elif len(feature_indices) == 3:
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot data points colored by cluster
            scatter = ax.scatter(
                X_plot[:, feature_indices[0]],
                X_plot[:, feature_indices[1]],
                X_plot[:, feature_indices[2]],
                c=labels,
                cmap='viridis',
                alpha=0.7,
                s=50
            )
            
            # Plot cluster centers if requested
            if show_centers and self.cluster_centers_ is not None:
                centers = ax.scatter(
                    self.cluster_centers_[:, feature_indices[0]],
                    self.cluster_centers_[:, feature_indices[1]],
                    self.cluster_centers_[:, feature_indices[2]],
                    c='red',
                    marker='X',
                    s=200,
                    alpha=1.0,
                    label='Cluster Centers'
                )
                ax.legend()
            
            # Add colorbar
            plt.colorbar(scatter, label='Cluster')
            
            # Set labels
            ax.set_xlabel(feature_names[feature_indices[0]])
            ax.set_ylabel(feature_names[feature_indices[1]])
            ax.set_zlabel(feature_names[feature_indices[2]])
        else:
            plt.close(fig)
            error_msg = f"Feature indices must contain 2 or 3 elements, got {len(feature_indices)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Set title
        plt.title(title)
        plt.tight_layout()
        
        logger.info("Cluster plot created successfully")
        return fig
    
    @staticmethod
    def plot_elbow_method(
        X: Union[pd.DataFrame, np.ndarray],
        k_range: List[int],
        random_state: Optional[int] = None,
        standardize: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        title: str = 'Elbow Method for Optimal k'
    ) -> plt.Figure:
        """
        Plot the elbow method curve to help determine optimal k.
        
        Args:
            X: Input features as DataFrame or ndarray.
            k_range: List of k values to try.
            random_state: Random seed for reproducibility.
            standardize: Whether to standardize features before clustering.
            figsize: Figure size as (width, height).
            title: Plot title.
            
        Returns:
            Matplotlib figure object.
            
        Raises:
            ValueError: If k_range is empty or contains invalid values.
            TypeError: If X is not a DataFrame or ndarray.
        """
        # Get inertia values for different k
        result = KMeansModel.find_optimal_k(
            X=X,
            k_range=k_range,
            random_state=random_state,
            standardize=standardize,
            method='elbow'
        )
        
        inertia_values = result['inertia_values']
        optimal_k = result['optimal_k']
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot elbow curve
        k_values = sorted(inertia_values.keys())
        inertia_list = [inertia_values[k] for k in k_values]
        
        ax.plot(k_values, inertia_list, 'bo-')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
        ax.set_title(title)
        
        # Highlight optimal k
        ax.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
        ax.legend()
        
        plt.tight_layout()
        logger.info("Elbow method plot created successfully")
        return fig
    
    @staticmethod
    def plot_silhouette_method(
        X: Union[pd.DataFrame, np.ndarray],
        k_range: List[int],
        random_state: Optional[int] = None,
        standardize: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        title: str = 'Silhouette Method for Optimal k'
    ) -> plt.Figure:
        """
        Plot the silhouette scores to help determine optimal k.
        
        Args:
            X: Input features as DataFrame or ndarray.
            k_range: List of k values to try.
            random_state: Random seed for reproducibility.
            standardize: Whether to standardize features before clustering.
            figsize: Figure size as (width, height).
            title: Plot title.
            
        Returns:
            Matplotlib figure object.
            
        Raises:
            ValueError: If k_range is empty or contains invalid values.
            TypeError: If X is not a DataFrame or ndarray.
        """
        # Get silhouette values for different k
        result = KMeansModel.find_optimal_k(
            X=X,
            k_range=k_range,
            random_state=random_state,
            standardize=standardize,
            method='silhouette'
        )
        
        silhouette_values = result.get('silhouette_values', {})
        optimal_k = result['optimal_k']
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot silhouette curve
        k_values = sorted(silhouette_values.keys())
        silhouette_list = [silhouette_values[k] for k in k_values if silhouette_values[k] is not None]
        valid_k_values = [k for k in k_values if silhouette_values[k] is not None]
        
        if valid_k_values:
            ax.plot(valid_k_values, silhouette_list, 'bo-')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Silhouette Score')
            ax.set_title(title)
            
            # Highlight optimal k
            ax.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No valid silhouette scores available',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
        
        plt.tight_layout()
        logger.info("Silhouette method plot created successfully")
        return fig
