"""
Tests for the PCA implementation in the unsupervised learning module.

This module contains tests for the PCAModel class, including initialization,
fit, transform, evaluation, and visualization methods.
"""

from typing import Tuple, List, Dict, Any, Optional, cast
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import logging
from _pytest.logging import LogCaptureFixture

from ml.unsupervised.pca import PCAModel


@pytest.fixture
def sample_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for testing PCA.
    
    Returns:
        Tuple containing features array and target labels.
    """
    # Generate synthetic data with 3 clusters in 5 dimensions
    X, y = make_blobs(
        n_samples=100,
        n_features=5,
        centers=3,
        cluster_std=1.0,
        random_state=42
    )
    
    # Add some correlation between features
    correlation_matrix = np.array([
        [1.0, 0.8, 0.2, 0.1, 0.05],
        [0.8, 1.0, 0.3, 0.2, 0.1],
        [0.2, 0.3, 1.0, 0.7, 0.4],
        [0.1, 0.2, 0.7, 1.0, 0.6],
        [0.05, 0.1, 0.4, 0.6, 1.0]
    ])
    
    # Apply correlation to make the data more suitable for PCA
    X = np.dot(X, correlation_matrix)
    
    return X, y


@pytest.fixture
def sample_dataframe(sample_data: Tuple[np.ndarray, np.ndarray]) -> pd.DataFrame:
    """
    Convert sample data to pandas DataFrame.
    
    Args:
        sample_data: Sample data from the sample_data fixture.
        
    Returns:
        Sample data as DataFrame.
    """
    X, _ = sample_data
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])


class TestPCAModel:
    """Tests for the PCAModel class."""
    
    def test_init(self) -> None:
        """Test PCAModel initialization with different parameters."""
        # Test with default parameters
        model = PCAModel()
        assert model.n_components is None
        assert model.standardize is True
        assert model.scaler is not None
        assert model.components_ is None
        assert model.explained_variance_ is None
        assert model.explained_variance_ratio_ is None
        
        # Test with specific n_components
        model = PCAModel(n_components=2)
        assert model.n_components == 2
        
        # Test with float n_components (variance ratio)
        model = PCAModel(n_components=0.95)
        assert model.n_components == 0.95
        
        # Test with standardize=False
        model = PCAModel(standardize=False)
        assert model.standardize is False
        assert model.scaler is None
    
    def test_fit_ndarray(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test fitting PCAModel with numpy ndarray.
        
        Args:
            sample_data: Sample data from the sample_data fixture.
        """
        X, _ = sample_data
        
        # Create and fit model
        model = PCAModel(n_components=3, random_state=42)
        fitted_model = model.fit(X)
        
        # Check that fit returns self
        assert fitted_model is model
        
        # Check that model attributes are set
        assert model.components_ is not None
        assert model.explained_variance_ is not None
        assert model.explained_variance_ratio_ is not None
        assert model.singular_values_ is not None
        assert model.mean_ is not None
        assert model.n_samples_ == X.shape[0]
        assert model.n_features_ == X.shape[1]
        assert model.n_components_selected_ == 3
        
        # Check shapes
        assert model.components_.shape == (3, X.shape[1])
        assert len(model.explained_variance_) == 3
        assert len(model.explained_variance_ratio_) == 3
    
    def test_fit_dataframe(self, sample_dataframe: pd.DataFrame) -> None:
        """
        Test fitting PCAModel with pandas DataFrame.
        
        Args:
            sample_dataframe: Sample data as DataFrame.
        """
        # Create and fit model
        model = PCAModel(n_components=3, random_state=42)
        model.fit(sample_dataframe)
        
        # Check that model attributes are set
        assert model.components_ is not None
        assert model.explained_variance_ is not None
        assert model.explained_variance_ratio_ is not None
        assert model.feature_names_ == sample_dataframe.columns.tolist()
        
        # Check shapes
        assert model.components_.shape == (3, sample_dataframe.shape[1])
        assert len(model.explained_variance_) == 3
        assert len(model.explained_variance_ratio_) == 3
    
    def test_fit_input_validation(self) -> None:
        """Test input validation in fit method."""
        model = PCAModel()
        
        # Test with invalid input type
        with pytest.raises(TypeError):
            model.fit("not_a_dataframe_or_ndarray")  # type: ignore
        
        # Test with empty DataFrame
        with pytest.raises(ValueError):
            model.fit(pd.DataFrame())
        
        # Test with empty ndarray
        with pytest.raises(ValueError):
            model.fit(np.array([]))
        
        # Test with NaN values in DataFrame
        df_with_nan = pd.DataFrame({'a': [1, 2, np.nan], 'b': [4, 5, 6]})
        with pytest.raises(ValueError):
            model.fit(df_with_nan)
        
        # Test with NaN values in ndarray
        array_with_nan = np.array([[1, 2], [3, 4], [np.nan, 6]])
        with pytest.raises(ValueError):
            model.fit(array_with_nan)
    
    def test_transform(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test transform method.
        
        Args:
            sample_data: Sample data from the sample_data fixture.
        """
        X, _ = sample_data
        
        # Create and fit model
        model = PCAModel(n_components=3, random_state=42)
        model.fit(X)
        
        # Transform data
        transformed = model.transform(X)
        
        # Check shape of transformed data
        assert transformed.shape == (X.shape[0], 3)
        
        # Test with DataFrame
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        transformed_df = model.transform(df)
        
        # Check shape of transformed data
        assert transformed_df.shape == (df.shape[0], 3)
        
        # Check that the results are the same
        np.testing.assert_allclose(transformed, transformed_df)
    
    def test_transform_input_validation(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test input validation in transform method.
        
        Args:
            sample_data: Sample data from the sample_data fixture.
        """
        X, _ = sample_data
        
        # Create and fit model
        model = PCAModel(n_components=3, random_state=42)
        model.fit(X)
        
        # Test with unfitted model
        unfitted_model = PCAModel()
        with pytest.raises(ValueError):
            unfitted_model.transform(X)
        
        # Test with invalid input type
        with pytest.raises(TypeError):
            model.transform("not_a_dataframe_or_ndarray")  # type: ignore
        
        # Test with empty DataFrame
        with pytest.raises(ValueError):
            model.transform(pd.DataFrame())
        
        # Test with empty ndarray
        with pytest.raises(ValueError):
            model.transform(np.array([]))
        
        # Test with NaN values
        array_with_nan = np.array([[1, 2], [3, 4], [np.nan, 6]])
        with pytest.raises(ValueError):
            model.transform(array_with_nan)
    
    def test_fit_transform(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test fit_transform method.
        
        Args:
            sample_data: Sample data from the sample_data fixture.
        """
        X, _ = sample_data
        
        # Create model
        model = PCAModel(n_components=3, random_state=42)
        
        # Fit and transform in one step
        transformed = model.fit_transform(X)
        
        # Check that model is fitted
        assert model.components_ is not None
        
        # Check shape of transformed data
        assert transformed.shape == (X.shape[0], 3)
        
        # Compare with separate fit and transform
        model2 = PCAModel(n_components=3, random_state=42)
        model2.fit(X)
        transformed2 = model2.transform(X)
        
        # Results should be the same
        np.testing.assert_allclose(transformed, transformed2)
    
    def test_inverse_transform(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test inverse_transform method.
        
        Args:
            sample_data: Sample data from the sample_data fixture.
        """
        X, _ = sample_data
        
        # Create and fit model
        model = PCAModel(n_components=3, random_state=42)
        model.fit(X)
        
        # Transform data
        transformed = model.transform(X)
        
        # Inverse transform
        reconstructed = model.inverse_transform(transformed)
        
        # Check shape of reconstructed data
        assert reconstructed.shape == X.shape
        
        # Check that the reconstruction is reasonable
        # (not exact due to dimensionality reduction)
        reconstruction_error = np.mean((X - reconstructed) ** 2)
        assert reconstruction_error < 1.0
    
    def test_evaluate(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test evaluate method.
        
        Args:
            sample_data: Sample data from the sample_data fixture.
        """
        X, _ = sample_data
        
        # Create and fit model
        model = PCAModel(n_components=3, random_state=42)
        model.fit(X)
        
        # Evaluate model
        metrics = model.evaluate()
        
        # Check that metrics are returned
        assert isinstance(metrics, dict)
        assert 'n_components' in metrics
        assert 'explained_variance' in metrics
        assert 'explained_variance_ratio' in metrics
        assert 'cumulative_explained_variance_ratio' in metrics
        assert 'total_explained_variance' in metrics
        
        # Check values
        assert metrics['n_components'] == 3
        assert len(metrics['explained_variance']) == 3
        assert len(metrics['explained_variance_ratio']) == 3
        assert len(metrics['cumulative_explained_variance_ratio']) == 3
        assert 0 <= metrics['total_explained_variance'] <= 1
    
    def test_evaluate_input_validation(self) -> None:
        """Test input validation in evaluate method."""
        # Test with unfitted model
        model = PCAModel()
        with pytest.raises(ValueError):
            model.evaluate()
    
    def test_find_optimal_n_components(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test find_optimal_n_components static method.
        
        Args:
            sample_data: Sample data from the sample_data fixture.
        """
        X, _ = sample_data
        
        # Find optimal n_components
        result = PCAModel.find_optimal_n_components(X, variance_threshold=0.95)
        
        # Check that results are returned
        assert isinstance(result, dict)
        assert 'optimal_n_components' in result
        assert 'variance_threshold' in result
        assert 'explained_variance_ratio' in result
        assert 'cumulative_variance_ratio' in result
        assert 'total_components' in result
        
        # Check values
        assert isinstance(result['optimal_n_components'], int)
        assert result['optimal_n_components'] > 0
        assert result['optimal_n_components'] <= X.shape[1]
        assert result['variance_threshold'] == 0.95
        assert len(result['explained_variance_ratio']) == X.shape[1]
        assert len(result['cumulative_variance_ratio']) == X.shape[1]
        assert result['total_components'] == X.shape[1]
        
        # Check with different threshold
        result_low = PCAModel.find_optimal_n_components(X, variance_threshold=0.7)
        result_high = PCAModel.find_optimal_n_components(X, variance_threshold=0.99)
        
        # Lower threshold should require fewer components
        assert result_low['optimal_n_components'] <= result['optimal_n_components']
        # Higher threshold should require more components
        assert result_high['optimal_n_components'] >= result['optimal_n_components']
    
    def test_find_optimal_n_components_input_validation(self) -> None:
        """Test input validation in find_optimal_n_components static method."""
        # Test with invalid input type
        with pytest.raises(TypeError):
            PCAModel.find_optimal_n_components("not_a_dataframe_or_ndarray")  # type: ignore
        
        # Test with invalid variance threshold
        with pytest.raises(ValueError):
            PCAModel.find_optimal_n_components(np.random.rand(10, 5), variance_threshold=0)
        
        with pytest.raises(ValueError):
            PCAModel.find_optimal_n_components(np.random.rand(10, 5), variance_threshold=1.1)
    
    def test_plot_explained_variance(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test plot_explained_variance method.
        
        Args:
            sample_data: Sample data from the sample_data fixture.
        """
        X, _ = sample_data
        
        # Create and fit model
        model = PCAModel(random_state=42)
        model.fit(X)
        
        # Plot explained variance
        fig = model.plot_explained_variance()
        
        # Check that figure is returned
        assert isinstance(fig, plt.Figure)
        
        # Test with cumulative=False
        fig = model.plot_explained_variance(cumulative=False)
        assert isinstance(fig, plt.Figure)
    
    def test_plot_components(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test plot_components method.
        
        Args:
            sample_data: Sample data from the sample_data fixture.
        """
        X, _ = sample_data
        
        # Create and fit model
        model = PCAModel(random_state=42)
        model.fit(X)
        
        # Plot components
        fig = model.plot_components()
        
        # Check that figure is returned
        assert isinstance(fig, plt.Figure)
        
        # Test with different component indices
        fig = model.plot_components(component_indices=[0, 1, 2])
        assert isinstance(fig, plt.Figure)
        
        # Test with single component
        fig = model.plot_components(component_indices=[0])
        assert isinstance(fig, plt.Figure)
    
    def test_plot_transformed_data(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test plot_transformed_data method.
        
        Args:
            sample_data: Sample data from the sample_data fixture.
        """
        X, y = sample_data
        
        # Create and fit model
        model = PCAModel(n_components=3, random_state=42)
        model.fit(X)
        
        # Plot transformed data
        fig = model.plot_transformed_data(X)
        
        # Check that figure is returned
        assert isinstance(fig, plt.Figure)
        
        # Test with labels
        fig = model.plot_transformed_data(X, labels=y)
        assert isinstance(fig, plt.Figure)
        
        # Test with different components
        fig = model.plot_transformed_data(X, component_x=1, component_y=2)
        assert isinstance(fig, plt.Figure)
    
    def test_plot_input_validation(self, sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Test input validation in plotting methods.
        
        Args:
            sample_data: Sample data from the sample_data fixture.
        """
        X, _ = sample_data
        
        # Create and fit model
        model = PCAModel(n_components=3, random_state=42)
        model.fit(X)
        
        # Test with unfitted model
        unfitted_model = PCAModel()
        with pytest.raises(ValueError):
            unfitted_model.plot_explained_variance()
        
        with pytest.raises(ValueError):
            unfitted_model.plot_components()
        
        with pytest.raises(ValueError):
            unfitted_model.plot_transformed_data(X)
        
        # Test with invalid component indices
        with pytest.raises(ValueError):
            model.plot_components(component_indices=[10])  # Out of range
        
        with pytest.raises(ValueError):
            model.plot_transformed_data(X, component_x=10)  # Out of range
    
    def test_logging(self, sample_data: Tuple[np.ndarray, np.ndarray], caplog: LogCaptureFixture) -> None:
        """
        Test that logging works correctly.
        
        Args:
            sample_data: Sample data from the sample_data fixture.
            caplog: Fixture to capture log messages.
        """
        X, _ = sample_data
        
        # Set log level to INFO
        caplog.set_level("INFO")
        
        # Create and fit model
        model = PCAModel(n_components=3, random_state=42)
        model.fit(X)
        
        # Check initialization log
        assert "Initialized PCAModel" in caplog.text
        
        # Check fit log
        assert "PCAModel fitted successfully" in caplog.text
        assert "Total variance explained" in caplog.text
        
        # Check transform log
        caplog.clear()
        model.transform(X)
        assert "Transformed" in caplog.text
        
        # Check evaluate log
        caplog.clear()
        model.evaluate()
        assert "PCA evaluation completed" in caplog.text
        
        # Check find_optimal_n_components log
        caplog.clear()
        PCAModel.find_optimal_n_components(X)
        assert "Optimal n_components determined" in caplog.text
        
        # Check plotting logs
        caplog.clear()
        model.plot_explained_variance()
        assert "Explained variance plot created" in caplog.text
        
        caplog.clear()
        model.plot_components()
        assert "Component contribution plot created" in caplog.text
        
        caplog.clear()
        model.plot_transformed_data(X)
        assert "Transformed data plot created" in caplog.text
