"""
Configuration for pytest.
This file contains fixtures and configuration settings for pytest.
"""
import os
import sys
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Check for available dependencies
def is_tensorflow_available():
    """Check if TensorFlow is available."""
    try:
        import tensorflow
        return True
    except ImportError:
        return False

def is_torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False

def is_transformers_available():
    """Check if Transformers is available."""
    try:
        import transformers
        return True
    except ImportError:
        return False

def is_pmdarima_available():
    """Check if pmdarima is available."""
    try:
        import pmdarima
        return True
    except ImportError:
        return False

# Register markers for skipping tests based on dependencies
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "requires_tensorflow: mark test as requiring tensorflow")
    config.addinivalue_line("markers", "requires_torch: mark test as requiring pytorch")
    config.addinivalue_line("markers", "requires_transformers: mark test as requiring transformers")
    config.addinivalue_line("markers", "requires_pmdarima: mark test as requiring pmdarima")

# Skip tests that require unavailable dependencies
def pytest_runtest_setup(item):
    """Skip tests that require unavailable dependencies."""
    markers = list(item.iter_markers())
    for marker in markers:
        if marker.name == 'requires_tensorflow' and not is_tensorflow_available():
            pytest.skip("Test requires TensorFlow")
        elif marker.name == 'requires_torch' and not is_torch_available():
            pytest.skip("Test requires PyTorch")
        elif marker.name == 'requires_transformers' and not is_transformers_available():
            pytest.skip("Test requires Transformers")
        elif marker.name == 'requires_pmdarima' and not is_pmdarima_available():
            pytest.skip("Test requires pmdarima")
