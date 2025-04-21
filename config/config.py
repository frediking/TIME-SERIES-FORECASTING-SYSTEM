import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


# Add this near the top of the file after imports
DEFAULT_CONFIG = {
    "models": {
        "xgboost": {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "verbosity": 0,
            "learning_rate": 0.1
        },
        "lightgbm": {
            "objective": "regression",
            "metric": "mse",
            "verbosity": -1
        },
        "lstm": {
            "units": 50,
            "activation": "relu",
            "optimizer": "adam"
        }
    },
    "training": {
        "batch_size": 32,
        "epochs": 100,
        "validation_split": 0.2         
    },
    "paths": {  
        "data": "./data",
        "models": "./models",
        "logs": "./logs"
    },
    "monitoring": { 
        "enabled": True,
        "interval": 60,
        "metrics": ["loss", "accuracy", "rmse"]
    }
}



logger = logging.getLogger(__name__)

class Config:
    """
    Centralized configuration management.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Parameters
        ----------
        config_path : str, optional
            Path to config file (if None, uses default config)
        """
        self.config_path = config_path
        self.data = DEFAULT_CONFIG.copy()
        
        if config_path is not None:
            self.load(config_path)

    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns
        -------
        dict
            Current configuration
        """
        return self.data
    
    def load(self, config_path: str) -> None:
        """
        Load configuration from file.
        
        Parameters
        ----------
        config_path : str
            Path to config file (YAML or JSON)
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                self.data = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                self.data = json.load(f)
            else:
                raise ValueError("Unsupported config file format")
        
        logger.info(f"Loaded config from {config_path}")
    
    def save(self, config_path: str) -> None:
        """
        Save configuration to file.
        
        Parameters
        ----------
        config_path : str
            Path to config file (YAML or JSON)
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                yaml.dump(self.data, f)
            elif config_path.suffix == '.json':
                json.dump(self.data, f, indent=4)
            else:
                raise ValueError("Unsupported config file format")
        
        logger.info(f"Saved config to {config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Parameters
        ----------
        key : str
            Dot notation key (e.g. 'paths.models_dir')
        default : Any, optional
            Default value if key not found
            
        Returns
        -------
        Any
            Configuration value
        """
        keys = key.split('.')
        val = self.data
        
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Parameters
        ----------
        key : str
            Dot notation key (e.g. 'paths.models_dir')
        value : Any
            Value to set
        """
        keys = key.split('.')
        current = self.data
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """
        Get model parameters from config.
        
        Parameters
        ----------
        model_type : str
            Type of model (e.g. 'xgboost', 'lstm')
            
        Returns
        -------
        dict
            Model parameters
        """
        try:
            return self.get(f'models.{model_type}', {})
        except Exception as e:
            logger.error(f"Error getting model params: {str(e)}")
            return {}
    
    def get_model_version(self) -> str:
        """Get current model version from config."""
        return self.get('model.version', '1.0.0')
        
    def increment_version(self, level: str = 'patch') -> None:
        """Increment version number.
        
        Args:
            level: Which part to increment ('major', 'minor', 'patch')
        """
        version = self.get_model_version()
        major, minor, patch = map(int, version.split('.'))
        
        if level == 'major':
            major += 1
            minor = 0
            patch = 0
        elif level == 'minor':
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
            
        new_version = f"{major}.{minor}.{patch}"
        self.set('model.version', new_version)

# Create global config instance
config = Config()
