"""
TensorFlow DNN Classifier implementation.

This module provides a DNNClassifier class with a standardized
interface for training, prediction, and evaluation.
"""

__author__ = "Usman Ahmad"

from typing import Dict, List, Union, Optional, Tuple, Any, cast, Callable
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

from ml.metrics import classification_metrics

# Configure logging
logger = logging.getLogger(__name__)


class DNNClassifier:
    """
    TensorFlow Deep Neural Network Classifier with standardized interface for training,
    prediction, and evaluation.
    
    This class provides a configurable DNN architecture with options for
    layer sizes, activation functions, regularization, and optimization.
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [128, 64, 32],
        activation: str = 'relu',
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize a DNNClassifier with specified parameters.
        
        Args:
            hidden_layers: List of integers specifying the number of neurons in each hidden layer.
            activation: Activation function to use in hidden layers.
            dropout_rate: Dropout rate for regularization (0 to 1).
            use_batch_norm: Whether to use batch normalization.
            learning_rate: Learning rate for the Adam optimizer.
            batch_size: Batch size for training.
            epochs: Maximum number of epochs for training.
            early_stopping_patience: Number of epochs with no improvement after which training will stop.
            random_state: Seed for random number generation.
        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        
        # Set random seed if provided
        if random_state is not None:
            tf.random.set_seed(random_state)
            np.random.seed(random_state)
        
        # Initialize model and preprocessors
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.num_classes = None
        self.feature_names = None
        self.history = None
        
        logger.info(f"Initialized DNNClassifier with {len(hidden_layers)} hidden layers: {hidden_layers}, "
                   f"activation: {activation}, dropout: {dropout_rate}")
    
    def _build_model(self, input_dim: int, num_classes: int) -> Sequential:
        """
        Build the TensorFlow model architecture.
        
        Args:
            input_dim: Number of input features.
            num_classes: Number of output classes.
            
        Returns:
            Compiled TensorFlow Sequential model.
        """
        model = Sequential()
        
        # Input layer
        model.add(Dense(
            self.hidden_layers[0],
            input_dim=input_dim,
            activation=self.activation,
            kernel_initializer='glorot_uniform'
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
                kernel_initializer='glorot_uniform'
            ))
            
            if self.use_batch_norm:
                model.add(BatchNormalization())
                
            if self.dropout_rate > 0:
                model.add(Dropout(self.dropout_rate))
        
        # Output layer
        if num_classes == 2:
            # Binary classification
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            # Multi-class classification
            model.add(Dense(num_classes, activation='softmax'))
            loss = 'categorical_crossentropy'
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=['accuracy']
        )
        
        logger.info(f"Built model architecture with input_dim={input_dim}, "
                   f"num_classes={num_classes}, loss={loss}")
        
        return model
    
    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, List],
        validation_split: float = 0.2,
        class_weights: Optional[Dict[int, float]] = None
    ) -> Dict[str, Any]:
        """
        Train the DNN classifier on the provided data.
        
        Args:
            X: Feature matrix for training.
            y: Target variable for training.
            validation_split: Fraction of data to use for validation.
            class_weights: Optional dictionary mapping class indices to weights.
            
        Returns:
            Dictionary containing training results:
                - 'model': Trained model
                - 'history': Training history
                - 'feature_names': Feature names if available
        """
        # Validate inputs
        self._validate_inputs(X, y)
        
        # Convert inputs to numpy arrays and get feature names if available
        X_array, feature_names = self._prepare_features(X)
        y_array = self._prepare_target(y)
        
        # Store feature names
        self.feature_names = feature_names
        
        # Log training start
        logger.info(f"Training DNNClassifier on data with shape X: {X_array.shape}, "
                   f"y: {y_array.shape}, num_classes: {self.num_classes}")
        
        results: Dict[str, Any] = {}
        
        try:
            # Build the model
            input_dim = X_array.shape[1]
            self.model = self._build_model(input_dim, self.num_classes)
            
            # Define callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            # Prepare target for training
            y_train = y_array
            if self.num_classes > 2:
                # One-hot encode for multi-class
                y_train = to_categorical(y_array)
            
            # Train the model
            self.history = self.model.fit(
                X_array, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            # Mark model as trained
            self.is_trained = True
            
            # Store results
            results['model'] = self.model
            results['history'] = self.history.history
            results['feature_names'] = self.feature_names
            
            logger.info("DNNClassifier training completed successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training DNNClassifier: {str(e)}")
            raise
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix for prediction.
            
        Returns:
            Array of predicted class labels.
            
        Raises:
            ValueError: If model has not been trained.
        """
        if not self.is_trained or self.model is None:
            logger.error("Model has not been trained. Call train() before predict()")
            raise ValueError("Model has not been trained. Call train() before predict()")
        
        # Prepare features
        X_array, _ = self._prepare_features(X, training=False)
        
        logger.info(f"Making predictions with DNNClassifier on data with shape: {X_array.shape}")
        
        try:
            # Get raw predictions
            if self.num_classes == 2:
                # Binary classification
                y_prob = self.model.predict(X_array)
                y_pred = (y_prob > 0.5).astype(int).flatten()
            else:
                # Multi-class classification
                y_prob = self.model.predict(X_array)
                y_pred = np.argmax(y_prob, axis=1)
            
            # Inverse transform to original labels
            y_pred = self.label_encoder.inverse_transform(y_pred)
            
            logger.info(f"Predictions completed. Output shape: {y_pred.shape}")
            return y_pred
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict class probabilities using the trained model.
        
        Args:
            X: Feature matrix for prediction.
            
        Returns:
            Array of predicted class probabilities.
            
        Raises:
            ValueError: If model has not been trained.
        """
        if not self.is_trained or self.model is None:
            logger.error("Model has not been trained. Call train() before predict_proba()")
            raise ValueError("Model has not been trained. Call train() before predict_proba()")
        
        # Prepare features
        X_array, _ = self._prepare_features(X, training=False)
        
        logger.info(f"Predicting probabilities with DNNClassifier on data with shape: {X_array.shape}")
        
        try:
            # Get probability predictions
            if self.num_classes == 2:
                # Binary classification - return probability of class 1
                y_prob = self.model.predict(X_array).flatten()
                # Convert to 2D array with probabilities for both classes
                y_prob_2d = np.zeros((len(y_prob), 2))
                y_prob_2d[:, 0] = 1 - y_prob
                y_prob_2d[:, 1] = y_prob
                return y_prob_2d
            else:
                # Multi-class classification
                return self.model.predict(X_array)
        except Exception as e:
            logger.error(f"Error predicting probabilities: {str(e)}")
            raise
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, List],
        average: str = 'binary'
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Feature matrix for evaluation.
            y: True target values.
            average: Parameter for precision, recall, and f1 score.
                Options: 'binary', 'micro', 'macro', 'weighted', 'samples'.
                
        Returns:
            Dictionary containing evaluation metrics.
            
        Raises:
            ValueError: If model has not been trained.
        """
        if not self.is_trained or self.model is None:
            logger.error("Model has not been trained. Call train() before evaluate()")
            raise ValueError("Model has not been trained. Call train() before evaluate()")
        
        # Validate inputs
        self._validate_inputs(X, y)
        
        # Prepare data
        X_array, _ = self._prepare_features(X, training=False)
        y_array = self._prepare_target(y, encode=False)
        
        logger.info(f"Evaluating DNNClassifier on data with shape X: {X_array.shape}, "
                   f"y: {y_array.shape}")
        
        try:
            # Make predictions
            y_pred = self.predict(X)
            
            # Get probabilities for ROC AUC (only for binary classification)
            y_prob = None
            if self.num_classes == 2:
                y_prob = self.predict_proba(X)[:, 1]  # Probability of positive class
            
            # Calculate metrics
            metrics = classification_metrics(y_array, y_pred, y_prob, average=average)
            
            # Also get TensorFlow's evaluation metrics
            if self.num_classes > 2:
                y_eval = to_categorical(self.label_encoder.transform(y_array))
            else:
                y_eval = y_array
                
            tf_metrics = self.model.evaluate(X_array, y_eval, verbose=0)
            metrics['loss'] = tf_metrics[0]
            metrics['accuracy_tf'] = tf_metrics[1]
            
            logger.info(f"Evaluation completed. Accuracy: {metrics['accuracy']:.4f}, "
                       f"Precision: {metrics['precision']:.4f}, "
                       f"Recall: {metrics['recall']:.4f}, "
                       f"F1: {metrics['f1']:.4f}")
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
        X_eval: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        n_repeats: int = 5,
        random_state: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get feature importance from the trained model using permutation importance.
        
        For DNNs, this uses a permutation-based approach to estimate feature importance
        by measuring how model performance decreases when a feature is randomly shuffled.
        
        Args:
            feature_names: List of feature names. If None, stored feature names will be used.
            X_eval: Evaluation data to use for permutation importance. If None, synthetic data is generated.
            n_repeats: Number of times to permute each feature for more stable results.
            random_state: Random seed for reproducibility.
            
        Returns:
            DataFrame with feature names and their importance scores,
            sorted by importance in descending order.
            
        Raises:
            ValueError: If model has not been trained or feature_names length doesn't match input features.
        """
        if not self.is_trained or self.model is None:
            logger.error("Model has not been trained. Call train() before get_feature_importance()")
            raise ValueError("Model has not been trained. Call train() before get_feature_importance()")
        
        # Use stored feature names if not provided
        if feature_names is None:
            feature_names = self.feature_names
            
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.model.input_shape[1])]  # type: ignore
        
        # Ensure we have the right number of feature names
        input_dim = self.model.input_shape[1]  # type: ignore
        if len(feature_names) != input_dim:
            error_msg = f"Length mismatch between feature_names ({len(feature_names)}) and input features ({input_dim})"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Set random seed if provided
            if random_state is not None:
                np.random.seed(random_state)
            
            # Create or use evaluation data
            if X_eval is None:
                # Generate synthetic data
                logger.info("No evaluation data provided, generating synthetic data for feature importance")
                sample_size = 1000
                X_eval = np.random.normal(0, 1, (sample_size, input_dim))
            elif isinstance(X_eval, pd.DataFrame):
                X_eval = X_eval.values
            
            # Scale features
            X_eval_scaled = self.scaler.transform(X_eval)
            
            # Get baseline predictions and accuracy
            baseline_pred = self.model.predict(X_eval_scaled, verbose=0)
            
            if self.num_classes == 2:
                # Binary classification
                baseline_pred_class = (baseline_pred > 0.5).astype(int).flatten()
            else:
                # Multi-class classification
                baseline_pred_class = np.argmax(baseline_pred, axis=1)
            
            # Calculate importance for each feature with multiple repeats
            importances = np.zeros(len(feature_names))
            
            for _ in range(n_repeats):
                for i in range(len(feature_names)):
                    # Create a permuted version of the feature
                    X_permuted = X_eval_scaled.copy()
                    X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                    
                    # Get predictions with permuted feature
                    permuted_pred = self.model.predict(X_permuted, verbose=0)
                    
                    if self.num_classes == 2:
                        # Binary classification
                        permuted_pred_class = (permuted_pred > 0.5).astype(int).flatten()
                    else:
                        # Multi-class classification
                        permuted_pred_class = np.argmax(permuted_pred, axis=1)
                    
                    # Calculate accuracy decrease
                    accuracy_decrease = np.mean(baseline_pred_class != permuted_pred_class)
                    importances[i] += accuracy_decrease
            
            # Average importances across repeats
            importances = importances / n_repeats
            
            # Normalize importances to sum to 1
            if np.sum(importances) > 0:
                importances = importances / np.sum(importances)
            
            # Create DataFrame with feature importances
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            
            # Sort by importance in descending order
            importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
            
            logger.info(f"Feature importance extracted for {len(feature_names)} features")
            
            return importance_df
        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
            raise
    
    def _prepare_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        training: bool = True
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        """
        Prepare features for training or prediction.
        
        Args:
            X: Feature matrix.
            training: Whether this is for training (True) or prediction (False).
            
        Returns:
            Tuple of (processed numpy array, feature names if available).
        """
        # Get feature names if available
        feature_names = None
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
        
        # Scale features
        if training:
            X_scaled = self.scaler.fit_transform(X_array)
        else:
            X_scaled = self.scaler.transform(X_array)
        
        return X_scaled, feature_names
    
    def _prepare_target(
        self,
        y: Union[pd.Series, np.ndarray, List],
        encode: bool = True
    ) -> np.ndarray:
        """
        Prepare target variable for training or evaluation.
        
        Args:
            y: Target variable.
            encode: Whether to encode labels (True) or use as-is (False).
            
        Returns:
            Processed numpy array.
        """
        # Convert to numpy array
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        if encode:
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y_array)
            
            # Store number of classes
            self.num_classes = len(self.label_encoder.classes_)
            
            return y_encoded
        else:
            return y_array
    
    def _validate_inputs(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, List]
    ) -> None:
        """
        Validate input data for training and evaluation.
        
        Args:
            X: Feature matrix.
            y: Target variable.
            
        Raises:
            TypeError: If input types are invalid.
            ValueError: If inputs have incompatible shapes.
        """
        # Check X type
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            logger.error("TypeError: X must be a pandas DataFrame or numpy array")
            raise TypeError("X must be a pandas DataFrame or numpy array")
        
        # Check y type
        if not isinstance(y, (pd.Series, np.ndarray, list)):
            logger.error("TypeError: y must be a pandas Series, numpy array, or list")
            raise TypeError("y must be a pandas Series, numpy array, or list")
        
        # Check shapes
        X_shape = X.shape[0]
        y_shape = len(y)
        
        if X_shape != y_shape:
            logger.error(f"ValueError: X and y have incompatible shapes: {X_shape} vs {y_shape}")
            raise ValueError(f"X and y have incompatible shapes: {X_shape} vs {y_shape}")
    
    def get_model_summary(self) -> str:
        """
        Get a string representation of the model architecture.
        
        Returns:
            String with model summary.
            
        Raises:
            ValueError: If model has not been built.
        """
        if self.model is None:
            logger.error("Model has not been built yet")
            raise ValueError("Model has not been built yet")
        
        # Capture model summary as string
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return "\n".join(summary_lines)
