"""
Tests for K-Means clustering implementation.

This module contains tests for the KMeansModel class, including initialization,
fitting, prediction, evaluation, and visualization methods.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union, cast
from typing import TYPE_CHECKING

from ml.unsupervised.kmeans import KMeansModel

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def sample_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample data with 3 distinct clusters.
    
    Returns:
        Tuple containing features array and true labels.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate 3 clusters
    n_samples = 300
    n_samples_per_cluster = n_samples // 3
    
    # Cluster 1: centered at (0, 0)
    cluster1 = np.random.randn(n_samples_per_cluster, 2) * 0.5
    
    # Cluster 2: centered at (5, 5)
    cluster2 = np.random.randn(n_samples_per_cluster, 2) * 0.5 + np.array([5, 5])
    
    # Cluster 3: centered at (0, 5)
    cluster3 = np.random.randn(n_samples_per_cluster, 2) * 0.5 + np.array([0, 5])
    
    # Combine clusters
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # True labels (for reference)
    y_true = np.hstack([
        np.zeros(n_samples_per_cluster),
        np.ones(n_samples_per_cluster),
        np.ones(n_samples_per_cluster) * 2
    ]).astype(int)
    
    return X, y_true


@pytest.fixture
def sample_dataframe(sample_data: Tuple[np.ndarray, np.ndarray]) -> pd.DataFrame:
    """
    Create sample data as pandas DataFrame.
    
    Args:
        sample_data: Sample data from sample_data fixture.
        
    Returns:
        DataFrame with sample data.
    """
    X, _ = sample_data
    return pd.DataFrame(X, columns=['feature1', 'feature2'])


class TestKMeansModel:
    """Tests for the KMeansModel class."""
    
    def test_init(self) -> None:
        """Test initialization of KMeansModel."""
        # Test with default parameters
        model = KMeansModel()
        assert model.k == 3
        assert model.standardize is True
        assert model.scaler is not None
        assert model.cluster_centers_ is None
        assert model.labels_ is None
        assert model.inertia_ is None
        
        # Test with custom parameters
        model = KMeansModel(k=5, random_state=42, standardize=False)
        assert model.k == 5
        assert model.random_state == 42
        assert model.standardize is False
        assert model.scaler is None
        
        # Test with invalid k
        with pytest.raises(ValueError, match="Number of clusters must be at least 2"):
            KMeansModel(k=1)
    
    def test_fit_ndarray(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test fitting KMeansModel with numpy ndarray.
        
        Args:
            sample_data: Sample data from sample_data fixture.
        """
        X, _ = sample_data
        
        # Create and fit model
        model = KMeansModel(k=3, random_state=42, n_init=10)
        fitted_model = model.fit(X)
        
        # Check that fit returns self
        assert fitted_model is model
        
        # Check that model attributes are set
        assert model.cluster_centers_ is not None
        assert model.labels_ is not None
        assert model.inertia_ is not None
        assert model.cluster_centers_.shape == (3, 2)
        assert len(model.labels_) == X.shape[0]
    
    def test_fit_dataframe(self, sample_dataframe: pd.DataFrame) -> None:
        """
        Test fitting KMeansModel with pandas DataFrame.
        
        Args:
            sample_dataframe: Sample data as DataFrame.
        """
        # Create and fit model
        model = KMeansModel(k=3, random_state=42, n_init=10)
        model.fit(sample_dataframe)
        
        # Check that model attributes are set
        assert model.cluster_centers_ is not None
        assert model.labels_ is not None
        assert model.inertia_ is not None
        assert model.cluster_centers_.shape == (3, 2)
        assert len(model.labels_) == len(sample_dataframe)
    
    def test_fit_input_validation(self) -> None:
        """Test input validation in fit method."""
        model = KMeansModel()
        
        # Test with invalid input type
        with pytest.raises(TypeError, match="X must be a pandas DataFrame or numpy ndarray"):
            model.fit([1, 2, 3])  # type: ignore
        
        # Test with empty DataFrame
        with pytest.raises(ValueError, match="X cannot be empty"):
            model.fit(pd.DataFrame())
        
        # Test with empty ndarray
        with pytest.raises(ValueError, match="X cannot be empty"):
            model.fit(np.array([]))
        
        # Test with NaN values in DataFrame
        df_with_nan = pd.DataFrame({'a': [1, 2, np.nan], 'b': [4, 5, 6]})
        with pytest.raises(ValueError, match="X contains NaN values"):
            model.fit(df_with_nan)
        
        # Test with NaN values in ndarray
        array_with_nan = np.array([[1, 2], [3, 4], [np.nan, 6]])
        with pytest.raises(ValueError, match="X contains NaN values"):
            model.fit(array_with_nan)
    
    def test_predict(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test predict method.
        
        Args:
            sample_data: Sample data from sample_data fixture.
        """
        X, _ = sample_data
        
        # Create and fit model
        model = KMeansModel(k=3, random_state=42, n_init=10)
        model.fit(X)
        
        # Predict on training data
        labels = model.predict(X)
        assert labels.shape == (X.shape[0],)
        assert np.all(labels == model.labels_)
        
        # Predict on new data
        new_data = np.array([[0, 0], [5, 5], [0, 5]])
        new_labels = model.predict(new_data)
        assert new_labels.shape == (3,)
    
    def test_predict_input_validation(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test input validation in predict method.
        
        Args:
            sample_data: Sample data from sample_data fixture.
        """
        X, _ = sample_data
        
        # Create and fit model
        model = KMeansModel(k=3, random_state=42, n_init=10)
        model.fit(X)
        
        # Test with unfitted model
        unfitted_model = KMeansModel()
        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            unfitted_model.predict(X)
        
        # Test with invalid input type
        with pytest.raises(TypeError, match="X must be a pandas DataFrame or numpy ndarray"):
            model.predict([1, 2, 3])  # type: ignore
        
        # Test with empty DataFrame
        with pytest.raises(ValueError, match="X cannot be empty"):
            model.predict(pd.DataFrame())
        
        # Test with NaN values
        df_with_nan = pd.DataFrame({'a': [1, 2, np.nan], 'b': [4, 5, 6]})
        with pytest.raises(ValueError, match="X contains NaN values"):
            model.predict(df_with_nan)
    
    def test_evaluate(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test evaluate method.
        
        Args:
            sample_data: Sample data from sample_data fixture.
        """
        X, _ = sample_data
        
        # Create and fit model
        model = KMeansModel(k=3, random_state=42, n_init=10)
        model.fit(X)
        
        # Evaluate on training data
        metrics = model.evaluate(X)
        
        # Check metrics
        assert 'inertia' in metrics
        assert 'silhouette_score' in metrics
        assert 'cluster_sizes' in metrics
        assert metrics['inertia'] > 0
        assert -1 <= metrics['silhouette_score'] <= 1
        assert len(metrics['cluster_sizes']) == 3
        assert sum(metrics['cluster_sizes'].values()) == X.shape[0]
    
    def test_evaluate_input_validation(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test input validation in evaluate method.
        
        Args:
            sample_data: Sample data from sample_data fixture.
        """
        X, _ = sample_data
        
        # Create and fit model
        model = KMeansModel(k=3, random_state=42, n_init=10)
        model.fit(X)
        
        # Test with unfitted model
        unfitted_model = KMeansModel()
        with pytest.raises(ValueError, match="Model must be fitted before evaluation"):
            unfitted_model.evaluate(X)
    
    def test_find_optimal_k(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test find_optimal_k method.
        
        Args:
            sample_data: Sample data from sample_data fixture.
        """
        X, _ = sample_data
        
        # Test elbow method
        result_elbow = KMeansModel.find_optimal_k(
            X=X,
            k_range=[2, 3, 4, 5, 6],
            random_state=42,
            method='elbow'
        )
        
        assert 'optimal_k' in result_elbow
        assert 'inertia_values' in result_elbow
        assert result_elbow['optimal_k'] in [2, 3, 4, 5, 6]
        assert len(result_elbow['inertia_values']) == 5
        
        # Test silhouette method
        result_silhouette = KMeansModel.find_optimal_k(
            X=X,
            k_range=[2, 3, 4, 5, 6],
            random_state=42,
            method='silhouette'
        )
        
        assert 'optimal_k' in result_silhouette
        assert result_silhouette['optimal_k'] in [2, 3, 4, 5, 6]
    
    def test_find_optimal_k_input_validation(self) -> None:
        """Test input validation in find_optimal_k method."""
        # Test with invalid input type
        with pytest.raises(TypeError, match="X must be a pandas DataFrame or numpy ndarray"):
            KMeansModel.find_optimal_k([1, 2, 3], [2, 3, 4])  # type: ignore
        
        # Test with empty k_range
        X = np.random.rand(10, 2)
        with pytest.raises(ValueError, match="k_range must contain values >= 2"):
            KMeansModel.find_optimal_k(X, [])
        
        # Test with invalid k values
        with pytest.raises(ValueError, match="k_range must contain values >= 2"):
            KMeansModel.find_optimal_k(X, [1, 2, 3])
        
        # Test with invalid method
        with pytest.raises(ValueError, match="method must be 'elbow' or 'silhouette'"):
            KMeansModel.find_optimal_k(X, [2, 3, 4], method='invalid')  # type: ignore
    
    def test_plot_clusters(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test plot_clusters method.
        
        Args:
            sample_data: Sample data from sample_data fixture.
        """
        X, _ = sample_data
        
        # Create and fit model
        model = KMeansModel(k=3, random_state=42, n_init=10)
        model.fit(X)
        
        # Test 2D plot
        fig = model.plot_clusters(X)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with feature indices
        fig = model.plot_clusters(X, feature_indices=[0, 1])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with custom parameters
        fig = model.plot_clusters(
            X,
            figsize=(8, 6),
            title='Custom Title',
            show_centers=False
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_clusters_input_validation(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test input validation in plot_clusters method.
        
        Args:
            sample_data: Sample data from sample_data fixture.
        """
        X, _ = sample_data
        
        # Create and fit model
        model = KMeansModel(k=3, random_state=42, n_init=10)
        model.fit(X)
        
        # Test with unfitted model
        unfitted_model = KMeansModel()
        with pytest.raises(ValueError, match="Model must be fitted before plotting"):
            unfitted_model.plot_clusters(X)
        
        # Test with invalid feature indices
        with pytest.raises(ValueError, match="Feature indices .* out of range"):
            model.plot_clusters(X, feature_indices=[10, 11])
        
        # Test with wrong number of feature indices
        with pytest.raises(ValueError, match="Feature indices must contain 2 or 3 elements"):
            model.plot_clusters(X, feature_indices=[0])
    
    def test_plot_elbow_method(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test plot_elbow_method.
        
        Args:
            sample_data: Sample data from sample_data fixture.
        """
        X, _ = sample_data
        
        # Test basic functionality
        fig = KMeansModel.plot_elbow_method(
            X=X,
            k_range=[2, 3, 4, 5, 6],
            random_state=42
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_silhouette_method(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test plot_silhouette_method.
        
        Args:
            sample_data: Sample data from sample_data fixture.
        """
        X, _ = sample_data
        
        # Test basic functionality
        fig = KMeansModel.plot_silhouette_method(
            X=X,
            k_range=[2, 3, 4, 5, 6],
            random_state=42
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_logging(self, sample_data: Tuple[np.ndarray, np.ndarray], caplog: "LogCaptureFixture") -> None:
        """
        Test that appropriate logging messages are generated.
        
        Args:
            sample_data: Sample data from sample_data fixture.
            caplog: Fixture to capture log messages.
        """
        X, _ = sample_data
        
        # Set log level to INFO
        caplog.set_level("INFO")
        
        # Create and fit model
        model = KMeansModel(k=3, random_state=42, n_init=10)
        model.fit(X)
        
        # Check initialization log
        assert "Initialized KMeansModel with k=3" in caplog.text
        
        # Check fit log
        assert "Data standardized before clustering" in caplog.text
        assert "KMeansModel fitted successfully with inertia" in caplog.text
        
        # Check predict log
        model.predict(X)
        assert f"Predicted cluster labels for {X.shape[0]} samples" in caplog.text
        
        # Check evaluate log
        model.evaluate(X)
        assert "Silhouette score:" in caplog.text
        assert "Clustering evaluation completed with metrics" in caplog.text
        
        # Check find_optimal_k log
        caplog.clear()
        KMeansModel.find_optimal_k(X, [2, 3, 4])
        assert "Data standardized before optimal k selection" in caplog.text
        assert "Trying k=" in caplog.text
        assert "Optimal k determined by" in caplog.text
