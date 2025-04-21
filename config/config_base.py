"""
Base configuration class for the forecasting system.
Provides core configuration functionality and interface.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigBase:
    """Base configuration class with core functionality."""
    
    def __init__(self):
        """Initialize base configuration."""
        self.base_dir = Path(__file__).parent.parent
        self._config_dict: Dict[str, Any] = {}  # Changed name to avoid confusion
        
        # Initialize base paths
        self._config_dict['paths'] = {
            'base_dir': str(self.base_dir),
            'models_dir': str(self.base_dir / 'models'),
            'data_dir': str(self.base_dir / 'data'),
            'logs_dir': str(self.base_dir / 'logs'),
            'results_dir': str(self.base_dir / 'results')
        }
        
        # Initialize monitoring settings
        self._config_dict['monitoring'] = {
            'storage_dir': str(self.base_dir / 'monitoring_data'),
            'drift_detection_path': str(self.base_dir / 'drift_detection'),
            'max_cache_size': 1000,
            'max_records_per_series': 10000
        }
        
        # Initialize model parameters
        self._config_dict['models'] = {
            'xgboost': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100
            },
            'lstm': {
                'units': 64,
                'dropout': 0.2,
                'epochs': 50
            },
            'transformer': {
                'd_model': 32,
                'n_heads': 4,
                'n_layers': 3,
                'dropout': 0.1,
                'prediction_length': 24,
                'batch_size': 32,
                'learning_rate': 1e-4
            }
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        return self._config_dict
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration dictionary."""
        self._config_dict.update(new_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        try:
            value = self._config_dict
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        parts = key.split('.')
        current = self._config_dict
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value

# Create global config instance
config = ConfigBase()

# Allow environment variable overrides
for key, value in os.environ.items():
    if key.startswith('FORECAST_'):
        config_key = key.replace('FORECAST_', '').lower().replace('_', '.')
        config.set(config_key, value)