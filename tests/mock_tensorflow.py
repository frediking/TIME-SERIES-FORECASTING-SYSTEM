"""
Mock TensorFlow module for testing purposes.
This allows tests to run without requiring TensorFlow to be installed.
"""
import sys
from unittest.mock import MagicMock

# Create a mock TensorFlow module
mock_tf = MagicMock()
mock_tf.__version__ = '2.12.0'

# Create mock keras module
mock_keras = MagicMock()
mock_tf.keras = mock_keras

# Create mock Sequential model
mock_sequential = MagicMock()
mock_keras.models = MagicMock()
mock_keras.models.Sequential = mock_sequential

# Create mock layers
mock_layers = MagicMock()
mock_keras.layers = mock_layers

# Add common layer types
mock_layers.LSTM = MagicMock()
mock_layers.Dense = MagicMock()
mock_layers.Dropout = MagicMock()

# Create mock optimizers
mock_optimizers = MagicMock()
mock_keras.optimizers = mock_optimizers

# Add common optimizers
mock_optimizers.Adam = MagicMock()

# Create mock losses
mock_losses = MagicMock()
mock_keras.losses = mock_losses

# Add common losses
mock_losses.MeanSquaredError = MagicMock()
mock_losses.MeanAbsoluteError = MagicMock()

# Create mock callbacks
mock_callbacks = MagicMock()
mock_keras.callbacks = mock_callbacks

# Add common callbacks
mock_callbacks.EarlyStopping = MagicMock()
mock_callbacks.ModelCheckpoint = MagicMock()

# Add save and load functions
mock_keras.models.save_model = MagicMock()
mock_keras.models.load_model = MagicMock()

# Register the mock modules
sys.modules['tensorflow'] = mock_tf
sys.modules['tensorflow.keras'] = mock_keras
sys.modules['tensorflow.keras.models'] = mock_keras.models
sys.modules['tensorflow.keras.layers'] = mock_keras.layers
sys.modules['tensorflow.keras.optimizers'] = mock_keras.optimizers
sys.modules['tensorflow.keras.losses'] = mock_keras.losses
sys.modules['tensorflow.keras.callbacks'] = mock_keras.callbacks
