"""
TensorFlow Deep Neural Network Regressor implementation.

This module provides a wrapper for TensorFlow's DNN Regressor
with additional functionality for training, prediction, evaluation,
and model architecture management.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from ml.metrics.regression import regression_metrics

logger = logging.getLogger(__name__)


class TensorFlowDNNRegressor:
    """
    TensorFlow Deep Neural Network Regressor.
    
    This class provides a consistent interface for training, prediction,
    and evaluation of deep neural network regression models using TensorFlow.
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [64, 32],
        activation: str = 'relu',
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        validation_split: float = 0.2,
        random_state: Optional[int] = None,
        use_batch_norm: bool = True,
        model_dir: Optional[str] = None
    ) -> None:
        """
        Initialize the TensorFlow DNN Regressor.
        
        Args:
            hidden_layers: List of integers, each representing the number of neurons
                in a hidden layer.
            activation: Activation function to use in hidden layers.
            dropout_rate: Dropout rate for regularization.
            learning_rate: Learning rate for the Adam optimizer.
            batch_size: Number of samples per gradient update.
            epochs: Maximum number of epochs to train for.
            early_stopping_patience: Number of epochs with no improvement after
                which training will be stopped.
            validation_split: Fraction of the training data to be used as validation data.
            random_state: Random seed for reproducibility.
            use_batch_norm: Whether to use batch normalization after each hidden layer.
            model_dir: Directory to save model checkpoints. If None, no checkpoints are saved.
        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.use_batch_norm = use_batch_norm
        self.model_dir = model_dir
        self.feature_names: List[str] = []
        self.target_name: str = "target"
        self.n_features: int = 0
        self.model: Optional[Sequential] = None
        self.history: Optional[tf.keras.callbacks.History] = None
        self.is_fitted: bool = False
        
        # Set random seed for reproducibility
        if random_state is not None:
            tf.random.set_seed(random_state)
            np.random.seed(random_state)
        
        logger.info(
            f"Initialized TensorFlowDNNRegressor with hidden_layers={hidden_layers}, "
            f"activation={activation}, learning_rate={learning_rate}, "
            f"batch_size={batch_size}, epochs={epochs}"
        )
    
    def _build_model(self, input_dim: int) -> Sequential:
        """
        Build the TensorFlow Keras model architecture.
        
        Args:
            input_dim: Number of input features.
            
        Returns:
            Compiled Keras Sequential model.
        """
        model = Sequential()
        
        # Input layer
        model.add(Dense(
            self.hidden_layers[0],
            input_dim=input_dim,
            activation=self.activation,
            kernel_initializer='he_normal'
        ))
        
        if self.use_batch_norm:
            model.add(BatchNormalization())
        
        if self.dropout_rate > 0:
            model.add(Dropout(self.dropout_rate))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(Dense(
                units,
                activation=self.activation,
                kernel_initializer='he_normal'
            ))
            
            if self.use_batch_norm:
                model.add(BatchNormalization())
            
            if self.dropout_rate > 0:
                model.add(Dropout(self.dropout_rate))
        
        # Output layer (single neuron for regression)
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            loss='mean_squared_error',
            optimizer=optimizer,
            metrics=['mae', 'mse']
        )
        
        logger.info(f"Built TensorFlow DNN model with {len(self.hidden_layers)} hidden layers")
        return model
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None,
        target_name: str = "target",
        verbose: int = 1
    ) -> "TensorFlowDNNRegressor":
        """
        Fit the TensorFlow DNN Regressor model.
        
        Args:
            X: Features as DataFrame or numpy array.
            y: Target variable as Series or numpy array.
            feature_names: List of feature names when X is a numpy array.
                If X is a DataFrame, feature_names are extracted from it.
            target_name: Name of the target variable.
            verbose: Verbosity mode (0, 1, or 2).
            
        Returns:
            Self for method chaining.
            
        Raises:
            ValueError: If input validation fails.
            TypeError: If input types are not supported.
        """
        # Validate inputs
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            self.feature_names = X.columns.tolist()
        elif isinstance(X, np.ndarray):
            X_array = X
            if feature_names is not None:
                if len(feature_names) != X.shape[1]:
                    raise ValueError(
                        f"Length of feature_names ({len(feature_names)}) does not match "
                        f"number of features in X ({X.shape[1]})"
                    )
                self.feature_names = feature_names
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            raise TypeError(
                f"X must be a pandas DataFrame or numpy array, got {type(X)}"
            )
        
        if isinstance(y, pd.Series):
            y_array = y.values
            self.target_name = y.name if y.name is not None else target_name
        elif isinstance(y, np.ndarray):
            y_array = y
            self.target_name = target_name
        else:
            raise TypeError(
                f"y must be a pandas Series or numpy array, got {type(y)}"
            )
        
        # Reshape y if needed
        if len(y_array.shape) == 1:
            y_array = y_array.reshape(-1, 1)
        
        # Store number of features
        self.n_features = X_array.shape[1]
        
        # Build model
        self.model = self._build_model(self.n_features)
        
        # Set up callbacks
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=verbose
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint if model_dir is provided
        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)
            checkpoint_path = os.path.join(self.model_dir, "model_checkpoint.h5")
            checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=verbose
            )
            callbacks.append(checkpoint)
        
        # Fit the model
        try:
            self.history = self.model.fit(
                X_array,
                y_array,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=callbacks,
                verbose=verbose
            )
            
            self.is_fitted = True
            logger.info(
                f"Fitted TensorFlowDNNRegressor on {X_array.shape[0]} samples "
                f"with {X_array.shape[1]} features for {len(self.history.epoch)} epochs"
            )
            return self
        except Exception as e:
            logger.error(f"Error fitting TensorFlowDNNRegressor: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X: Features as DataFrame or numpy array.
            
        Returns:
            Array of predictions.
            
        Raises:
            ValueError: If model is not fitted or input validation fails.
            TypeError: If input types are not supported.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Validate input
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        elif isinstance(X, np.ndarray):
            X_array = X
        else:
            raise TypeError(
                f"X must be a pandas DataFrame or numpy array, got {type(X)}"
            )
        
        # Check feature count
        if X_array.shape[1] != self.n_features:
            raise ValueError(
                f"Number of features in X ({X_array.shape[1]}) does not match "
                f"number of features the model was trained on ({self.n_features})"
            )
        
        # Make predictions
        try:
            predictions = self.model.predict(X_array, verbose=0)
            # Flatten predictions if they are in shape (n_samples, 1)
            if len(predictions.shape) > 1 and predictions.shape[1] == 1:
                predictions = predictions.flatten()
                
            logger.info(f"Made predictions for {X_array.shape[0]} samples")
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate the model using regression metrics.
        
        Args:
            X: Features as DataFrame or numpy array.
            y: True target values as Series or numpy array.
            
        Returns:
            Dictionary of regression metrics.
            
        Raises:
            ValueError: If model is not fitted or input validation fails.
            TypeError: If input types are not supported.
        """
        y_pred = self.predict(X)
        
        # Convert y to numpy array if it's a pandas Series
        if isinstance(y, pd.Series):
            y_true = y.values
        elif isinstance(y, np.ndarray):
            y_true = y
        else:
            raise TypeError(
                f"y must be a pandas Series or numpy array, got {type(y)}"
            )
        
        # Calculate metrics
        metrics = regression_metrics(y_true, y_pred)
        logger.info(
            f"Evaluated model: MSE={metrics['mse']:.4f}, "
            f"RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}"
        )
        return metrics
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get the training history.
        
        Returns:
            Dictionary with training metrics per epoch.
            
        Raises:
            ValueError: If model is not fitted.
        """
        if not self.is_fitted or self.history is None:
            raise ValueError("Model must be fitted before getting training history")
        
        return self.history.history
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model.
            
        Raises:
            ValueError: If model is not fitted.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before saving")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Ensure the file has .keras extension to use the native Keras format
            if not filepath.endswith('.keras'):
                filepath = filepath + '.keras'
            
            # Save the model using the native Keras format
            self.model.save(filepath, save_format='keras')
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> "TensorFlowDNNRegressor":
        """
        Load a saved model from a file.
        
        Args:
            filepath: Path to the saved model.
            
        Returns:
            Self for method chaining.
            
        Raises:
            FileNotFoundError: If the model file doesn't exist.
            ValueError: If n_features is not set before loading.
        """
        # Check if n_features is set
        if self.n_features == 0:
            raise ValueError(
                "n_features must be set before loading a model. "
                "Please set n_features and feature_names attributes."
            )
            
        try:
            # Handle file extension
            if not filepath.endswith('.keras') and not filepath.endswith('.h5'):
                filepath = filepath + '.keras'
                
            # Load the model
            self.model = load_model(filepath)
            self.is_fitted = True
            
            # If feature_names is not set, generate default names
            if not self.feature_names:
                self.feature_names = [f"feature_{i}" for i in range(self.n_features)]
                
            logger.info(f"Model loaded from {filepath}")
            return self
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model.
        
        Returns:
            Dictionary with model information.
            
        Raises:
            ValueError: If model is not fitted.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before getting summary")
        
        # Get model architecture as string
        model_summary = []
        self.model.summary(print_fn=lambda x: model_summary.append(x))
        
        # Create summary dictionary
        summary = {
            'model_type': 'TensorFlowDNNRegressor',
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs if self.history is None else len(self.history.epoch),
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'model_architecture': '\n'.join(model_summary)
        }
        
        logger.info(f"Generated model summary for TensorFlowDNNRegressor")
        return summary
