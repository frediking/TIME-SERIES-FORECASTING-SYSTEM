"""
Mock Transformers module for testing purposes.
This allows tests to run without requiring the full Transformers library.
"""
import sys
from unittest.mock import MagicMock

# Create mock transformers module
mock_transformers = MagicMock()
mock_transformers.__version__ = '4.30.0'

# Create mock TimeSeriesTransformerModel
mock_ts_model = MagicMock()
mock_transformers.TimeSeriesTransformerModel = mock_ts_model

# Create mock TimeSeriesTransformerConfig
mock_ts_config = MagicMock()
mock_transformers.TimeSeriesTransformerConfig = mock_ts_config

# Register the mock module
sys.modules['transformers'] = mock_transformers
