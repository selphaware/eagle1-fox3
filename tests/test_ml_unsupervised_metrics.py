"""
Tests for unsupervised learning metrics.

This module contains tests for the clustering and dimensionality reduction
metrics implemented in the ml.unsupervised.metrics module.
"""

from typing import Dict, List, Tuple, Any, Optional, Union, cast
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs, make_classification
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ml.unsupervised.metrics import (
    clustering_metrics,
    explained_variance_metrics,
    cluster_separation_metrics
)

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])


@pytest.fixture
def sample_clustered_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample data with clear clusters.
    
    Returns:
        Tuple containing features array and cluster labels.
    """
    # Generate sample data with 3 clusters
    X, y = make_blobs(
        n_samples=300,
        n_features=5,
        centers=3,
        cluster_std=1.0,
        random_state=42
    )
    return X, y


@pytest.fixture
def sample_high_dim_data() -> np.ndarray:
    """
    Generate sample high-dimensional data for PCA testing.
    
    Returns:
        Array with correlated features suitable for dimensionality reduction.
    """
    # Generate data with some correlation between features
    X, _ = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=5,
        n_redundant=10,
        random_state=42
    )
    return X


class TestClusteringMetrics:
    """Tests for clustering metrics functions."""
    
    def test_clustering_metrics_basic(self, sample_clustered_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test basic functionality of clustering_metrics.
        
        Args:
            sample_clustered_data: Sample data with cluster labels.
        """
        X, labels = sample_clustered_data
        
        # Calculate metrics
        metrics = clustering_metrics(X, labels)
        
        # Check that all expected metrics are present
        assert isinstance(metrics, dict)
        assert 'silhouette_score' in metrics
        assert 'calinski_harabasz_score' in metrics
        assert 'davies_bouldin_score' in metrics
        assert 'inertia' in metrics
        assert 'cluster_sizes' in metrics
        
        # Check that metrics have reasonable values
        assert -1.0 <= metrics['silhouette_score'] <= 1.0
        assert metrics['calinski_harabasz_score'] > 0
        assert metrics['davies_bouldin_score'] >= 0
        assert metrics['inertia'] >= 0
        
        # Check cluster sizes
        assert len(metrics['cluster_sizes']) == len(np.unique(labels))
        assert sum(metrics['cluster_sizes'].values()) == len(labels)
    
    def test_clustering_metrics_with_dataframe(self, sample_clustered_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test clustering_metrics with pandas DataFrame input.
        
        Args:
            sample_clustered_data: Sample data with cluster labels.
        """
        X, labels = sample_clustered_data
        
        # Convert to DataFrame
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Calculate metrics
        metrics = clustering_metrics(df, labels)
        
        # Check that all expected metrics are present
        assert isinstance(metrics, dict)
        assert 'silhouette_score' in metrics
        assert 'calinski_harabasz_score' in metrics
        assert 'davies_bouldin_score' in metrics
    
    def test_clustering_metrics_different_distance_metrics(self, sample_clustered_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test clustering_metrics with different distance metrics.
        
        Args:
            sample_clustered_data: Sample data with cluster labels.
        """
        X, labels = sample_clustered_data
        
        # Test with different distance metrics
        metrics_euclidean = clustering_metrics(X, labels, metric='euclidean')
        metrics_manhattan = clustering_metrics(X, labels, metric='manhattan')
        metrics_cosine = clustering_metrics(X, labels, metric='cosine')
        
        # All should return valid results
        assert -1.0 <= metrics_euclidean['silhouette_score'] <= 1.0
        assert -1.0 <= metrics_manhattan['silhouette_score'] <= 1.0
        assert -1.0 <= metrics_cosine['silhouette_score'] <= 1.0
        
        # Results should be different for different metrics
        assert metrics_euclidean['silhouette_score'] != metrics_manhattan['silhouette_score']
        assert metrics_euclidean['silhouette_score'] != metrics_cosine['silhouette_score']
    
    def test_clustering_metrics_input_validation(self) -> None:
        """Test input validation for clustering_metrics."""
        # Invalid X type
        with pytest.raises(TypeError):
            clustering_metrics("not_an_array", np.array([0, 1, 2]))
        
        # Invalid labels type
        with pytest.raises(TypeError):
            clustering_metrics(np.array([[1, 2], [3, 4]]), "not_an_array")
        
        # Empty X
        with pytest.raises(ValueError):
            clustering_metrics(np.array([]), np.array([]))
        
        # X with NaN
        X_with_nan = np.array([[1, 2], [3, np.nan]])
        with pytest.raises(ValueError):
            clustering_metrics(X_with_nan, np.array([0, 1]))
        
        # Mismatched dimensions
        with pytest.raises(ValueError):
            clustering_metrics(np.array([[1, 2], [3, 4]]), np.array([0]))
        
        # Only one cluster
        with pytest.raises(ValueError):
            clustering_metrics(np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 0, 0]))
        
        # Cluster with only one sample
        with pytest.raises(ValueError):
            clustering_metrics(np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 0, 1]))
    
    def test_clustering_metrics_edge_cases(self) -> None:
        """Test edge cases for clustering_metrics."""
        # Create data with exactly 2 samples per cluster
        X = np.array([
            [0, 0], [0, 0.1],  # Cluster 0
            [10, 10], [10, 10.1]  # Cluster 1
        ])
        labels = np.array([0, 0, 1, 1])
        
        metrics = clustering_metrics(X, labels)
        
        # Should work with minimum samples per cluster
        assert -1.0 <= metrics['silhouette_score'] <= 1.0
        assert metrics['calinski_harabasz_score'] > 0
        
        # Very high silhouette score expected for well-separated clusters
        assert metrics['silhouette_score'] > 0.9


class TestExplainedVarianceMetrics:
    """Tests for explained variance metrics functions."""
    
    def test_explained_variance_metrics_basic(self, sample_high_dim_data: np.ndarray) -> None:
        """
        Test basic functionality of explained_variance_metrics.
        
        Args:
            sample_high_dim_data: Sample high-dimensional data.
        """
        X = sample_high_dim_data
        
        # Calculate metrics
        metrics = explained_variance_metrics(X)
        
        # Check that all expected metrics are present
        assert isinstance(metrics, dict)
        assert 'eigenvalues' in metrics
        assert 'explained_variance_ratio' in metrics
        assert 'cumulative_explained_variance' in metrics
        assert 'total_variance' in metrics
        assert 'n_components' in metrics
        assert 'components_for_threshold' in metrics
        
        # Check that metrics have reasonable values
        assert len(metrics['eigenvalues']) == X.shape[1]
        assert len(metrics['explained_variance_ratio']) == X.shape[1]
        assert len(metrics['cumulative_explained_variance']) == X.shape[1]
        assert metrics['total_variance'] > 0
        assert metrics['n_components'] == X.shape[1]
        
        # Check that cumulative variance is monotonically increasing and ends at or very close to 1.0
        # Use numpy's diff to check if all differences are non-negative (with small tolerance for numerical precision)
        diffs = np.diff(metrics['cumulative_explained_variance'])
        assert all(diff >= -1e-10 for diff in diffs), "Cumulative variance should be non-decreasing"
        assert np.isclose(metrics['cumulative_explained_variance'][-1], 1.0, atol=1e-10)
        
        # Check components for threshold
        assert 'n_components_for_95pct' in metrics['components_for_threshold']
        assert 0 < metrics['components_for_threshold']['n_components_for_95pct'] <= X.shape[1]
    
    def test_explained_variance_metrics_with_dataframe(self, sample_high_dim_data: np.ndarray) -> None:
        """
        Test explained_variance_metrics with pandas DataFrame input.
        
        Args:
            sample_high_dim_data: Sample high-dimensional data.
        """
        X = sample_high_dim_data
        
        # Convert to DataFrame
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Calculate metrics
        metrics = explained_variance_metrics(df)
        
        # Check that all expected metrics are present
        assert isinstance(metrics, dict)
        assert 'eigenvalues' in metrics
        assert 'explained_variance_ratio' in metrics
        assert len(metrics['eigenvalues']) == X.shape[1]
    
    def test_explained_variance_metrics_with_n_components(self, sample_high_dim_data: np.ndarray) -> None:
        """
        Test explained_variance_metrics with specified n_components.
        
        Args:
            sample_high_dim_data: Sample high-dimensional data.
        """
        X = sample_high_dim_data
        n_components = 5
        
        # Calculate metrics with limited components
        metrics = explained_variance_metrics(X, n_components=n_components)
        
        # Check that metrics have the correct dimensions
        assert len(metrics['eigenvalues']) == n_components
        assert len(metrics['explained_variance_ratio']) == n_components
        assert len(metrics['cumulative_explained_variance']) == n_components
        assert metrics['n_components'] == n_components
        
        # The cumulative variance might be less than 1.0 since we're not using all components
        # But it could also be 1.0 if the first n_components explain all variance
        assert metrics['cumulative_explained_variance'][-1] <= 1.0
    
    def test_explained_variance_metrics_input_validation(self) -> None:
        """Test input validation for explained_variance_metrics."""
        # Invalid X type
        with pytest.raises(TypeError):
            explained_variance_metrics("not_an_array")
        
        # Empty X
        with pytest.raises(ValueError):
            explained_variance_metrics(np.array([]))
        
        # X with NaN
        X_with_nan = np.array([[1, 2], [3, np.nan]])
        with pytest.raises(ValueError):
            explained_variance_metrics(X_with_nan)
    
    def test_explained_variance_metrics_edge_cases(self) -> None:
        """Test edge cases for explained_variance_metrics."""
        # Single feature
        X_single = np.array([[1], [2], [3], [4]])
        metrics_single = explained_variance_metrics(X_single)
        assert len(metrics_single['eigenvalues']) == 1
        assert np.isclose(metrics_single['explained_variance_ratio'][0], 1.0)
        
        # Two samples, two features
        X_min = np.array([[1, 2], [3, 4]])
        metrics_min = explained_variance_metrics(X_min)
        assert len(metrics_min['eigenvalues']) == 2
        
        # Minimum samples needed
        X_min_samples = np.array([[1, 2], [3, 4]])
        metrics_min_samples = explained_variance_metrics(X_min_samples)
        assert len(metrics_min_samples['eigenvalues']) > 0
        
        # Data with zero variance in one dimension
        X_zero_var = np.array([[1, 5], [2, 5], [3, 5], [4, 5]])
        metrics_zero_var = explained_variance_metrics(X_zero_var)
        # One eigenvalue should be very small compared to the other
        eigenvalues = metrics_zero_var['eigenvalues']
        assert eigenvalues[0] > 100 * eigenvalues[1]  # First eigenvalue much larger than second
        
        # n_components too large
        metrics_large_n = explained_variance_metrics(X_min, n_components=10)
        assert metrics_large_n['n_components'] == 2  # Should be capped at X.shape[1]


class TestClusterSeparationMetrics:
    """Tests for cluster separation metrics functions."""
    
    def test_cluster_separation_metrics_basic(self, sample_clustered_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test basic functionality of cluster_separation_metrics.
        
        Args:
            sample_clustered_data: Sample data with cluster labels.
        """
        X, labels = sample_clustered_data
        
        # Calculate metrics
        metrics = cluster_separation_metrics(X, labels)
        
        # Check that all expected metrics are present
        assert isinstance(metrics, dict)
        assert 'avg_within_cluster_distance' in metrics
        assert 'avg_between_cluster_distance' in metrics
        assert 'separation_index' in metrics
        assert 'min_between_cluster_distance' in metrics
        assert 'max_within_cluster_distance' in metrics
        assert 'dunn_index' in metrics
        
        # Check that metrics have reasonable values
        assert metrics['avg_within_cluster_distance'] > 0
        assert metrics['avg_between_cluster_distance'] > 0
        assert metrics['separation_index'] > 0
        assert metrics['min_between_cluster_distance'] > 0
        assert metrics['max_within_cluster_distance'] > 0
        assert metrics['dunn_index'] > 0
        
        # For well-separated clusters, separation index should be high
        assert metrics['separation_index'] > 1.0
    
    def test_cluster_separation_metrics_with_dataframe(self, sample_clustered_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test cluster_separation_metrics with pandas DataFrame input.
        
        Args:
            sample_clustered_data: Sample data with cluster labels.
        """
        X, labels = sample_clustered_data
        
        # Convert to DataFrame
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Calculate metrics
        metrics = cluster_separation_metrics(df, labels)
        
        # Check that all expected metrics are present
        assert isinstance(metrics, dict)
        assert 'avg_within_cluster_distance' in metrics
        assert 'avg_between_cluster_distance' in metrics
        assert 'separation_index' in metrics
    
    def test_cluster_separation_metrics_different_distance_metrics(self, sample_clustered_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test cluster_separation_metrics with different distance metrics.
        
        Args:
            sample_clustered_data: Sample data with cluster labels.
        """
        X, labels = sample_clustered_data
        
        # Test with different distance metrics
        metrics_euclidean = cluster_separation_metrics(X, labels, metric='euclidean')
        metrics_manhattan = cluster_separation_metrics(X, labels, metric='manhattan')
        metrics_cosine = cluster_separation_metrics(X, labels, metric='cosine')
        
        # All should return valid results
        assert metrics_euclidean['separation_index'] > 0
        assert metrics_manhattan['separation_index'] > 0
        assert metrics_cosine['separation_index'] > 0
        
        # Results should be different for different metrics
        assert metrics_euclidean['separation_index'] != metrics_manhattan['separation_index']
        assert metrics_euclidean['separation_index'] != metrics_cosine['separation_index']
    
    def test_cluster_separation_metrics_input_validation(self) -> None:
        """Test input validation for cluster_separation_metrics."""
        # Invalid X type
        with pytest.raises(TypeError):
            cluster_separation_metrics("not_an_array", np.array([0, 1, 2]))
        
        # Invalid labels type
        with pytest.raises(TypeError):
            cluster_separation_metrics(np.array([[1, 2], [3, 4]]), "not_an_array")
        
        # Empty X
        with pytest.raises(ValueError):
            cluster_separation_metrics(np.array([]), np.array([]))
        
        # X with NaN
        X_with_nan = np.array([[1, 2], [3, np.nan]])
        with pytest.raises(ValueError):
            cluster_separation_metrics(X_with_nan, np.array([0, 1]))
        
        # Mismatched dimensions
        with pytest.raises(ValueError):
            cluster_separation_metrics(np.array([[1, 2], [3, 4]]), np.array([0]))
        
        # Only one cluster
        with pytest.raises(ValueError):
            cluster_separation_metrics(np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 0, 0]))
    
    def test_cluster_separation_metrics_edge_cases(self) -> None:
        """Test edge cases for cluster_separation_metrics."""
        # Create data with exactly 2 samples per cluster
        X = np.array([
            [0, 0], [0, 0.1],  # Cluster 0
            [10, 10], [10, 10.1]  # Cluster 1
        ])
        labels = np.array([0, 0, 1, 1])
        
        metrics = cluster_separation_metrics(X, labels)
        
        # Should work with minimum samples per cluster
        assert metrics['avg_within_cluster_distance'] > 0
        assert metrics['avg_between_cluster_distance'] > 0
        
        # Very high separation index expected for well-separated clusters
        assert metrics['separation_index'] > 10.0
        
        # Create data with clusters that have only one point each
        X_single = np.array([
            [0, 0],  # Cluster 0
            [10, 10]  # Cluster 1
        ])
        labels_single = np.array([0, 1])
        
        # Should handle this case gracefully
        metrics_single = cluster_separation_metrics(X_single, labels_single)
        assert np.isnan(metrics_single['avg_within_cluster_distance'])
        assert metrics_single['avg_between_cluster_distance'] > 0
        assert np.isnan(metrics_single['separation_index'])


def test_metrics_integration() -> None:
    """Test integration of all metrics functions together."""
    # Generate data
    X, true_labels = make_blobs(
        n_samples=200,
        n_features=10,
        centers=4,
        cluster_std=1.0,
        random_state=42
    )
    
    # Run KMeans with explicit n_init to avoid FutureWarning
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(X)
    
    # Calculate all metrics
    cluster_metrics = clustering_metrics(X, pred_labels)
    separation_metrics = cluster_separation_metrics(X, pred_labels)
    variance_metrics = explained_variance_metrics(X)
    
    # Check that all metrics are calculated successfully
    assert cluster_metrics['silhouette_score'] > 0.5  # Well-separated clusters
    assert separation_metrics['separation_index'] > 1.0
    assert variance_metrics['components_for_threshold']['n_components_for_95pct'] <= X.shape[1]
    
    # Check that the metrics are consistent with each other
    # High silhouette score should correspond to high separation index
    assert (cluster_metrics['silhouette_score'] > 0.7) == (separation_metrics['separation_index'] > 5.0)
