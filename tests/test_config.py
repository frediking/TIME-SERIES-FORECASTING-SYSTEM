"""
Unit tests for the configuration system.
"""
import pytest
from pathlib import Path
import tempfile
import yaml
from config.config import Config, DEFAULT_CONFIG

@pytest.fixture
def config():
    return Config()

def test_config_initialization(config):
    """Test basic configuration initialization."""
    config_dict = config.get_config()
    assert config_dict is not None
    assert 'paths' in config_dict
    assert 'models' in config_dict
    assert 'monitoring' in config_dict

def test_config_get_set(config):
    """Test getting and setting configuration values."""
    # Test setting new value
    config.set('test.key', 'value')
    assert config.get('test.key') == 'value'
    
    # Test nested keys
    config.set('deep.nested.key', 123)
    assert config.get('deep.nested.key') == 123
    
    # Test default value
    assert config.get('nonexistent.key', 'default') == 'default'

def test_model_params(config):
    """Test model parameter access."""
    xgb_params = config.get('models.xgboost')
    assert xgb_params is not None
    assert 'learning_rate' in xgb_params

def test_config_file_loading(tmp_path):
    """Test loading configuration from file."""
    config_data = {
        'test': {'key': 'value'},
        'models': {'custom': {'param': 1}}
    }
    
    # Create temporary config file
    config_file = tmp_path / 'test_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    # Test loading
    config = Config(str(config_file))
    assert config.get('test.key') == 'value'
    assert config.get('models.custom.param') == 1
