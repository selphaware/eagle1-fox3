"""
Principal Component Analysis (PCA) implementation.

This module provides a wrapper around scikit-learn's PCA implementation
with additional functionality for component selection and visualization.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, cast
import numpy as np
import pandas as pd
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Configure logging
logger = logging.getLogger(__name__)


class PCAModel:
    """
    Principal Component Analysis model with additional functionality.
    
    This class wraps scikit-learn's PCA implementation and provides
    additional methods for optimal component selection, evaluation, and visualization.
    
    Attributes:
        model: The underlying scikit-learn PCA model.
        n_components: The number of principal components.
        random_state: Random seed for reproducibility.
        scaler: Optional StandardScaler for feature standardization.
        components_: Principal components after fitting.
        explained_variance_: Explained variance of each component after fitting.
        explained_variance_ratio_: Explained variance ratio of each component after fitting.
        singular_values_: Singular values of the data matrix after fitting.
    """
    
    def __init__(
        self,
        n_components: Optional[Union[int, float, str]] = None,
        random_state: Optional[int] = None,
        standardize: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize the PCAModel.
        
        Args:
            n_components: Number of components to keep. If None, keep all components.
                If int, keep n_components components.
                If float between 0 and 1, keep components that explain that fraction of variance.
                If 'mle', use Minka's MLE to determine the number of components.
            random_state: Random seed for reproducibility.
            standardize: Whether to standardize features before PCA.
            **kwargs: Additional arguments to pass to sklearn.decomposition.PCA.
        
        Raises:
            ValueError: If n_components is an int less than 1.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        
        # Initialize the underlying PCA model
        self.model = PCA(
            n_components=n_components,
            random_state=random_state,
            **kwargs
        )
        
        # These will be set after fitting
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_samples_ = None
        self.n_features_ = None
        self.n_components_selected_ = None
        
        logger.info(f"Initialized PCAModel with n_components={n_components}, standardize={standardize}")
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> 'PCAModel':
        """
        Fit the PCA model to the data.
        
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
            self.feature_names_ = X.columns.tolist()
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
            self.feature_names_ = [f"Feature {i}" for i in range(X_array.shape[1])]
        
        # Standardize if requested
        if self.standardize and self.scaler is not None:
            X_processed = self.scaler.fit_transform(X_array)
            logger.info("Data standardized before PCA")
        else:
            X_processed = X_array
        
        # Fit the model
        self.model.fit(X_processed)
        
        # Store results
        self.components_ = self.model.components_
        self.explained_variance_ = self.model.explained_variance_
        self.explained_variance_ratio_ = self.model.explained_variance_ratio_
        self.singular_values_ = self.model.singular_values_
        self.mean_ = self.model.mean_
        self.n_samples_ = X_processed.shape[0]
        self.n_features_ = X_processed.shape[1]
        self.n_components_selected_ = self.model.n_components_
        
        # Log results
        total_variance_explained = np.sum(self.explained_variance_ratio_)
        logger.info(f"PCAModel fitted successfully with {self.n_components_selected_} components")
        logger.info(f"Total variance explained: {total_variance_explained:.4f}")
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform data to the principal component space.
        
        Args:
            X: Input features as DataFrame or ndarray.
            
        Returns:
            Transformed data in the principal component space.
            
        Raises:
            ValueError: If model is not fitted or X contains NaN values.
            TypeError: If X is not a DataFrame or ndarray.
        """
        # Check if model is fitted
        if self.components_ is None:
            error_msg = "Model must be fitted before transformation"
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
        
        # Transform
        transformed = self.model.transform(X_processed)
        logger.info(f"Transformed {X_processed.shape[0]} samples to {transformed.shape[1]} principal components")
        return transformed
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Fit the model and transform the data to the principal component space.
        
        Args:
            X: Input features as DataFrame or ndarray.
            
        Returns:
            Transformed data in the principal component space.
            
        Raises:
            ValueError: If X is empty or contains NaN values.
            TypeError: If X is not a DataFrame or ndarray.
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data back to the original feature space.
        
        Args:
            X: Data in the principal component space.
            
        Returns:
            Data in the original feature space.
            
        Raises:
            ValueError: If model is not fitted.
            TypeError: If X is not a numpy ndarray.
        """
        # Check if model is fitted
        if self.components_ is None:
            error_msg = "Model must be fitted before inverse transformation"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Input validation
        if not isinstance(X, np.ndarray):
            error_msg = f"X must be a numpy ndarray, got {type(X)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        # Inverse transform
        inverse_transformed = self.model.inverse_transform(X)
        
        # Inverse standardize if needed
        if self.standardize and self.scaler is not None:
            inverse_transformed = self.scaler.inverse_transform(inverse_transformed)
        
        logger.info(f"Inverse transformed {X.shape[0]} samples back to original {inverse_transformed.shape[1]} features")
        return inverse_transformed
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the PCA model performance.
        
        Returns:
            Dictionary with evaluation metrics.
            
        Raises:
            ValueError: If model is not fitted.
        """
        # Check if model is fitted
        if self.components_ is None:
            error_msg = "Model must be fitted before evaluation"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Calculate metrics
        metrics = {}
        
        # Number of components
        metrics['n_components'] = self.n_components_selected_
        
        # Explained variance for each component
        metrics['explained_variance'] = self.explained_variance_.tolist()
        
        # Explained variance ratio for each component
        metrics['explained_variance_ratio'] = self.explained_variance_ratio_.tolist()
        
        # Cumulative explained variance ratio
        metrics['cumulative_explained_variance_ratio'] = np.cumsum(self.explained_variance_ratio_).tolist()
        
        # Total explained variance
        metrics['total_explained_variance'] = np.sum(self.explained_variance_ratio_)
        
        logger.info(f"PCA evaluation completed with total explained variance: {metrics['total_explained_variance']:.4f}")
        return metrics
    
    @staticmethod
    def find_optimal_n_components(
        X: Union[pd.DataFrame, np.ndarray],
        variance_threshold: float = 0.95,
        random_state: Optional[int] = None,
        standardize: bool = True
    ) -> Dict[str, Any]:
        """
        Find the optimal number of principal components.
        
        Args:
            X: Input features as DataFrame or ndarray.
            variance_threshold: Minimum fraction of variance to retain.
            random_state: Random seed for reproducibility.
            standardize: Whether to standardize features before PCA.
            
        Returns:
            Dictionary with optimal number of components and evaluation metrics.
            
        Raises:
            ValueError: If X is empty or contains NaN values, or if variance_threshold is not between 0 and 1.
            TypeError: If X is not a DataFrame or ndarray.
        """
        # Input validation
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            error_msg = f"X must be a pandas DataFrame or numpy ndarray, got {type(X)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
            
        if not 0 < variance_threshold <= 1:
            error_msg = f"variance_threshold must be between 0 and 1, got {variance_threshold}"
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
            logger.info("Data standardized before optimal n_components selection")
        else:
            X_processed = X_array
        
        # Fit PCA with all components
        pca = PCA(random_state=random_state)
        pca.fit(X_processed)
        
        # Calculate cumulative explained variance ratio
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        
        # Find optimal number of components
        optimal_n_components = int(np.argmax(cumulative_variance_ratio >= variance_threshold) + 1)
        
        # Prepare results
        result = {
            'optimal_n_components': optimal_n_components,
            'variance_threshold': variance_threshold,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': cumulative_variance_ratio.tolist(),
            'total_components': len(pca.explained_variance_ratio_)
        }
        
        logger.info(f"Optimal n_components determined: {optimal_n_components} (threshold: {variance_threshold})")
        logger.info(f"Explained variance with {optimal_n_components} components: {cumulative_variance_ratio[optimal_n_components-1]:.4f}")
        
        return result
    
    def plot_explained_variance(
        self,
        figsize: Tuple[int, int] = (10, 6),
        title: str = 'Explained Variance by Principal Component',
        cumulative: bool = True
    ) -> plt.Figure:
        """
        Plot the explained variance by principal component.
        
        Args:
            figsize: Figure size as (width, height).
            title: Plot title.
            cumulative: Whether to include cumulative explained variance.
            
        Returns:
            Matplotlib figure object.
            
        Raises:
            ValueError: If model is not fitted.
        """
        # Check if model is fitted
        if self.components_ is None:
            error_msg = "Model must be fitted before plotting"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Component indices (1-based for better readability)
        components = np.arange(1, len(self.explained_variance_ratio_) + 1)
        
        # Plot individual explained variance
        ax.bar(
            components,
            self.explained_variance_ratio_,
            alpha=0.7,
            label='Individual Explained Variance'
        )
        
        # Plot cumulative explained variance if requested
        if cumulative:
            ax.step(
                components,
                np.cumsum(self.explained_variance_ratio_),
                where='mid',
                color='red',
                label='Cumulative Explained Variance'
            )
            # Add horizontal line at 0.95 for reference
            ax.axhline(y=0.95, color='k', linestyle='--', alpha=0.7, label='95% Explained Variance')
        
        # Set labels and title
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title(title)
        ax.set_xticks(components)
        ax.legend()
        
        plt.tight_layout()
        logger.info("Explained variance plot created successfully")
        return fig
    
    def plot_components(
        self,
        component_indices: List[int] = [0, 1],
        figsize: Tuple[int, int] = (12, 10),
        title: str = 'Feature Contributions to Principal Components'
    ) -> plt.Figure:
        """
        Plot the feature contributions to selected principal components.
        
        Args:
            component_indices: Indices of components to plot (0-based).
            figsize: Figure size as (width, height).
            title: Plot title.
            
        Returns:
            Matplotlib figure object.
            
        Raises:
            ValueError: If model is not fitted or component_indices are invalid.
        """
        # Check if model is fitted
        if self.components_ is None:
            error_msg = "Model must be fitted before plotting"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate component indices
        if max(component_indices) >= len(self.components_):
            error_msg = f"Component indices {component_indices} out of range for model with {len(self.components_)} components"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create figure
        fig, axes = plt.subplots(len(component_indices), 1, figsize=figsize)
        if len(component_indices) == 1:
            axes = [axes]
        
        # Plot each selected component
        for i, comp_idx in enumerate(component_indices):
            # Get component and feature names
            component = self.components_[comp_idx]
            features = getattr(self, 'feature_names_', [f"Feature {j}" for j in range(len(component))])
            
            # Sort by absolute contribution
            sorted_indices = np.argsort(np.abs(component))[::-1]
            sorted_features = [features[j] for j in sorted_indices]
            sorted_contributions = component[sorted_indices]
            
            # Plot
            axes[i].barh(sorted_features, sorted_contributions, color='skyblue')
            axes[i].set_xlabel('Contribution')
            axes[i].set_title(f'PC{comp_idx+1} ({self.explained_variance_ratio_[comp_idx]:.2%} variance)')
            axes[i].axvline(x=0, color='k', linestyle='--', alpha=0.3)
            
            # Add grid lines
            axes[i].grid(axis='x', linestyle='--', alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        
        logger.info(f"Component contribution plot created for components {[i+1 for i in component_indices]}")
        return fig
    
    def plot_transformed_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        component_x: int = 0,
        component_y: int = 1,
        labels: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 8),
        title: str = 'Data in Principal Component Space'
    ) -> plt.Figure:
        """
        Plot the data transformed to the principal component space.
        
        Args:
            X: Input features as DataFrame or ndarray.
            component_x: Index of component to plot on x-axis (0-based).
            component_y: Index of component to plot on y-axis (0-based).
            labels: Optional array of labels for coloring points.
            figsize: Figure size as (width, height).
            title: Plot title.
            
        Returns:
            Matplotlib figure object.
            
        Raises:
            ValueError: If model is not fitted or component indices are invalid.
            TypeError: If X is not a DataFrame or ndarray.
        """
        # Transform the data
        transformed_data = self.transform(X)
        
        # Validate component indices
        if max(component_x, component_y) >= transformed_data.shape[1]:
            error_msg = f"Component indices {[component_x, component_y]} out of range for transformed data with {transformed_data.shape[1]} components"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Close all existing figures to prevent 'More than 20 figures' warning
        plt.close('all')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot transformed data
        if labels is not None:
            scatter = ax.scatter(
                transformed_data[:, component_x],
                transformed_data[:, component_y],
                c=labels,
                cmap='viridis',
                alpha=0.7,
                s=50
            )
            plt.colorbar(scatter, label='Class')
        else:
            ax.scatter(
                transformed_data[:, component_x],
                transformed_data[:, component_y],
                alpha=0.7,
                s=50
            )
        
        # Set labels and title
        var_x = self.explained_variance_ratio_[component_x] * 100
        var_y = self.explained_variance_ratio_[component_y] * 100
        ax.set_xlabel(f'PC{component_x+1} ({var_x:.2f}% variance)')
        ax.set_ylabel(f'PC{component_y+1} ({var_y:.2f}% variance)')
        ax.set_title(title)
        
        # Add grid
        ax.grid(linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        logger.info(f"Transformed data plot created for PC{component_x+1} vs PC{component_y+1}")
        
        # Close any other figures to prevent memory leaks and avoid the 'More than 20 figures' warning
        for i in plt.get_fignums():
            if plt.figure(i) != fig:
                plt.close(i)
                
        return fig
