"""
Metrics for unsupervised learning algorithms.

This module provides functions to evaluate unsupervised learning models,
including clustering metrics and dimensionality reduction metrics.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances
)

# Configure logging
logger = logging.getLogger(__name__)


def clustering_metrics(
    X: Union[pd.DataFrame, np.ndarray],
    labels: np.ndarray,
    metric: str = 'euclidean'
) -> Dict[str, float]:
    """
    Calculate metrics for clustering results.
    
    Args:
        X: Input features as DataFrame or ndarray.
        labels: Cluster labels for each sample.
        metric: Distance metric to use for silhouette score.
            Options include 'euclidean', 'manhattan', 'cosine', etc.
    
    Returns:
        Dictionary with clustering metrics.
        
    Raises:
        ValueError: If X is empty, contains NaN values, or if labels are invalid.
        TypeError: If X is not a DataFrame or ndarray, or if labels is not an ndarray.
    """
    # Input validation
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        error_msg = f"X must be a pandas DataFrame or numpy ndarray, got {type(X)}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    
    if not isinstance(labels, np.ndarray):
        error_msg = f"labels must be a numpy ndarray, got {type(labels)}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    
    # Convert DataFrame to numpy array
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
    else:
        if X.size == 0:
            error_msg = "X cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if np.isnan(X).any():
            error_msg = "X contains NaN values"
            logger.error(error_msg)
            raise ValueError(error_msg)
        X_array = X
    
    # Validate labels
    if labels.size == 0:
        error_msg = "labels cannot be empty"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if labels.size != X_array.shape[0]:
        error_msg = f"Number of labels ({labels.size}) does not match number of samples ({X_array.shape[0]})"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check if we have at least 2 clusters and more than 1 sample per cluster
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        error_msg = f"Number of clusters must be at least 2, got {n_clusters}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check if any cluster has only 1 sample (silhouette score will fail)
    cluster_sizes = np.bincount(labels.astype(int))
    if np.any(cluster_sizes[cluster_sizes > 0] < 2):
        error_msg = "Each cluster must have at least 2 samples for silhouette score"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Calculate metrics
    metrics = {}
    
    # Silhouette score (higher is better, range [-1, 1])
    # Measures how similar an object is to its own cluster compared to other clusters
    try:
        metrics['silhouette_score'] = silhouette_score(X_array, labels, metric=metric)
    except Exception as e:
        logger.warning(f"Failed to calculate silhouette score: {str(e)}")
        metrics['silhouette_score'] = np.nan
    
    # Calinski-Harabasz Index (higher is better)
    # Ratio of between-cluster dispersion to within-cluster dispersion
    try:
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_array, labels)
    except Exception as e:
        logger.warning(f"Failed to calculate Calinski-Harabasz score: {str(e)}")
        metrics['calinski_harabasz_score'] = np.nan
    
    # Davies-Bouldin Index (lower is better)
    # Average similarity between clusters, where similarity is the ratio of within-cluster distances to between-cluster distances
    try:
        metrics['davies_bouldin_score'] = davies_bouldin_score(X_array, labels)
    except Exception as e:
        logger.warning(f"Failed to calculate Davies-Bouldin score: {str(e)}")
        metrics['davies_bouldin_score'] = np.nan
    
    # Inertia (sum of squared distances to nearest centroid, lower is better)
    # Calculate manually since we don't have direct access to the centroids
    try:
        # Calculate cluster centroids
        centroids = np.zeros((n_clusters, X_array.shape[1]))
        for i, label in enumerate(unique_labels):
            cluster_points = X_array[labels == label]
            centroids[i] = np.mean(cluster_points, axis=0)
        
        # Calculate inertia
        inertia = 0.0
        for i, label in enumerate(labels):
            centroid_idx = np.where(unique_labels == label)[0][0]
            inertia += np.sum((X_array[i] - centroids[centroid_idx]) ** 2)
        
        metrics['inertia'] = inertia
    except Exception as e:
        logger.warning(f"Failed to calculate inertia: {str(e)}")
        metrics['inertia'] = np.nan
    
    # Cluster sizes
    cluster_counts = {}
    for label in unique_labels:
        cluster_counts[f'cluster_{label}_size'] = np.sum(labels == label)
    metrics['cluster_sizes'] = cluster_counts
    
    logger.info(f"Calculated clustering metrics: silhouette={metrics['silhouette_score']:.4f}, "
                f"calinski_harabasz={metrics['calinski_harabasz_score']:.4f}, "
                f"davies_bouldin={metrics['davies_bouldin_score']:.4f}")
    
    return metrics


def explained_variance_metrics(
    X: Union[pd.DataFrame, np.ndarray],
    n_components: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calculate explained variance metrics for dimensionality reduction.
    
    Args:
        X: Input features as DataFrame or ndarray.
        n_components: Number of components to consider. If None, all components are used.
    
    Returns:
        Dictionary with explained variance metrics.
        
    Raises:
        ValueError: If X is empty or contains NaN values.
        TypeError: If X is not a DataFrame or ndarray.
    """
    # Input validation
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        error_msg = f"X must be a pandas DataFrame or numpy ndarray, got {type(X)}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    
    # Convert DataFrame to numpy array
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
    else:
        if X.size == 0:
            error_msg = "X cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if np.isnan(X).any():
            error_msg = "X contains NaN values"
            logger.error(error_msg)
            raise ValueError(error_msg)
        X_array = X
    
    # Check if we have enough samples and features
    n_samples, n_features = X_array.shape
    if n_samples < 2 or n_features < 1:
        error_msg = f"Need at least 2 samples and 1 feature, got {n_samples} samples and {n_features} features"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Calculate covariance matrix
    cov_matrix = np.cov(X_array, rowvar=False)
    
    # Handle single feature case specially
    if n_features == 1:
        # For 1D data, just calculate variance directly
        variance = np.var(X_array, axis=0)[0]
        eigenvalues = np.array([variance])
        eigenvectors = np.array([[1.0]])
    else:
        # Calculate eigenvalues and eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        except np.linalg.LinAlgError as e:
            error_msg = f"Failed to compute eigenvalues/eigenvectors: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    # Limit to n_components if specified
    if n_components is not None:
        if n_components > len(eigenvalues):
            logger.warning(f"n_components ({n_components}) is greater than the number of features ({len(eigenvalues)}). Using all features.")
            n_components = len(eigenvalues)
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]
    
    # Calculate metrics
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    
    # Prepare results
    metrics = {
        'eigenvalues': eigenvalues.tolist(),
        'explained_variance_ratio': explained_variance_ratio.tolist(),
        'cumulative_explained_variance': cumulative_explained_variance.tolist(),
        'total_variance': float(total_variance),
        'n_components': len(eigenvalues)
    }
    
    # Calculate number of components needed for different variance thresholds
    thresholds = [0.7, 0.8, 0.9, 0.95, 0.99]
    components_for_threshold = {}
    for threshold in thresholds:
        n_needed = np.argmax(cumulative_explained_variance >= threshold) + 1
        components_for_threshold[f'n_components_for_{int(threshold*100)}pct'] = int(n_needed)
    
    metrics['components_for_threshold'] = components_for_threshold
    
    logger.info(f"Calculated explained variance metrics for {len(eigenvalues)} components")
    logger.info(f"Total variance: {total_variance:.4f}")
    logger.info(f"Components needed for 95% variance: {components_for_threshold['n_components_for_95pct']}")
    
    return metrics


def cluster_separation_metrics(
    X: Union[pd.DataFrame, np.ndarray],
    labels: np.ndarray,
    metric: str = 'euclidean'
) -> Dict[str, float]:
    """
    Calculate metrics for cluster separation and compactness.
    
    Args:
        X: Input features as DataFrame or ndarray.
        labels: Cluster labels for each sample.
        metric: Distance metric to use.
            Options include 'euclidean', 'manhattan', 'cosine', etc.
    
    Returns:
        Dictionary with cluster separation metrics.
        
    Raises:
        ValueError: If X is empty, contains NaN values, or if labels are invalid.
        TypeError: If X is not a DataFrame or ndarray, or if labels is not an ndarray.
    """
    # Input validation (similar to clustering_metrics)
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        error_msg = f"X must be a pandas DataFrame or numpy ndarray, got {type(X)}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    
    if not isinstance(labels, np.ndarray):
        error_msg = f"labels must be a numpy ndarray, got {type(labels)}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    
    # Convert DataFrame to numpy array
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
    else:
        if X.size == 0:
            error_msg = "X cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if np.isnan(X).any():
            error_msg = "X contains NaN values"
            logger.error(error_msg)
            raise ValueError(error_msg)
        X_array = X
    
    # Validate labels
    if labels.size == 0:
        error_msg = "labels cannot be empty"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if labels.size != X_array.shape[0]:
        error_msg = f"Number of labels ({labels.size}) does not match number of samples ({X_array.shape[0]})"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check if we have at least 2 clusters
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        error_msg = f"Number of clusters must be at least 2, got {n_clusters}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Calculate cluster centroids
        centroids = np.zeros((n_clusters, X_array.shape[1]))
        for i, label in enumerate(unique_labels):
            cluster_points = X_array[labels == label]
            centroids[i] = np.mean(cluster_points, axis=0)
        
        # Calculate pairwise distances between all points
        distances = pairwise_distances(X_array, metric=metric)
        
        # Calculate metrics
        metrics = {}
        
        # Within-cluster distances (average distance between points in the same cluster)
        within_cluster_distances = []
        for i, label in enumerate(unique_labels):
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) > 1:  # Need at least 2 points for distances
                cluster_distances = distances[np.ix_(cluster_indices, cluster_indices)]
                # Only consider upper triangle to avoid counting distances twice
                upper_triangle = np.triu_indices(len(cluster_indices), k=1)
                if upper_triangle[0].size > 0:
                    within_cluster_distances.append(np.mean(cluster_distances[upper_triangle]))
        
        metrics['avg_within_cluster_distance'] = np.mean(within_cluster_distances) if within_cluster_distances else np.nan
        
        # Between-cluster distances (average distance between centroids)
        centroid_distances = pairwise_distances(centroids, metric=metric)
        upper_triangle = np.triu_indices(n_clusters, k=1)
        metrics['avg_between_cluster_distance'] = np.mean(centroid_distances[upper_triangle]) if upper_triangle[0].size > 0 else np.nan
        
        # Separation index (ratio of between to within distances, higher is better)
        if metrics['avg_within_cluster_distance'] > 0:
            metrics['separation_index'] = metrics['avg_between_cluster_distance'] / metrics['avg_within_cluster_distance']
        else:
            metrics['separation_index'] = np.nan
        
        # Calculate minimum between-cluster distance
        metrics['min_between_cluster_distance'] = np.min(centroid_distances[upper_triangle]) if upper_triangle[0].size > 0 else np.nan
        
        # Calculate maximum within-cluster distance
        max_within = []
        for i, label in enumerate(unique_labels):
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) > 1:
                cluster_distances = distances[np.ix_(cluster_indices, cluster_indices)]
                upper_triangle = np.triu_indices(len(cluster_indices), k=1)
                if upper_triangle[0].size > 0:
                    max_within.append(np.max(cluster_distances[upper_triangle]))
        
        metrics['max_within_cluster_distance'] = np.max(max_within) if max_within else np.nan
        
        # Dunn index (min between-cluster / max within-cluster, higher is better)
        if metrics['max_within_cluster_distance'] > 0:
            metrics['dunn_index'] = metrics['min_between_cluster_distance'] / metrics['max_within_cluster_distance']
        else:
            metrics['dunn_index'] = np.nan
        
        logger.info(f"Calculated cluster separation metrics: "
                    f"separation_index={metrics['separation_index']:.4f}, "
                    f"dunn_index={metrics['dunn_index']:.4f}")
        
        return metrics
    
    except Exception as e:
        error_msg = f"Failed to calculate cluster separation metrics: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
