"""
Logging configuration for the forecasting system.
Provides centralized logging setup with file and console handlers.
"""
import os
import logging
import logging.handlers
from pathlib import Path
import sys
from datetime import datetime
import json
from typing import Dict, Any, Optional

# Import config if available
try:
    sys.path.append(str(Path(__file__).parent.parent))
    from config import config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Default log directory
DEFAULT_LOG_DIR = "logs"

def setup_logging(
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    log_format: Optional[str] = None,
    log_file_prefix: str = "forecasting",
    enable_console: bool = True,
    enable_file: bool = True,
    max_file_size_mb: int = 10,
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging for the application.
    
    Parameters
    ----------
    log_dir : str, optional
        Directory to store log files
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_format : str, optional
        Custom log format string
    log_file_prefix : str
        Prefix for log file names
    enable_console : bool
        Whether to enable console logging
    enable_file : bool
        Whether to enable file logging
    max_file_size_mb : int
        Maximum size of log files in MB before rotation
    backup_count : int
        Number of backup log files to keep
        
    Returns
    -------
    logging.Logger
        Root logger
    """
    # Get log directory from config if available
    if CONFIG_AVAILABLE and log_dir is None:
        log_dir = config.get('logging.log_dir', DEFAULT_LOG_DIR)
    elif log_dir is None:
        log_dir = DEFAULT_LOG_DIR
    
    # Get log level from config if available
    if CONFIG_AVAILABLE and log_level is None:
        log_level = config.get('logging.log_level', "INFO")
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Get numeric log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Define log format
    if log_format is None:
        log_format = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
    formatter = logging.Formatter(log_format)
    
    # Add console handler if enabled
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if enabled
    if enable_file:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"{log_file_prefix}_{timestamp}.log")
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log initial message
    root_logger.info(f"Logging initialized at level {log_level}")
    
    return root_logger


class JsonFormatter(logging.Formatter):
    """
    Formatter for JSON-structured logs.
    Useful for log aggregation systems like ELK stack.
    """
    
    def __init__(self, **kwargs):
        """Initialize JSON formatter."""
        self.json_fields = kwargs
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'line': record.lineno
        }
        
        # Add exception info if available
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add custom fields
        for field, value in self.json_fields.items():
            if callable(value):
                log_data[field] = value()
            else:
                log_data[field] = value
        
        # Add extra fields from record
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        return json.dumps(log_data)


class ModelMonitorHandler(logging.Handler):
    """
    Custom handler for model monitoring.
    Captures model performance metrics and other monitoring data.
    """
    
    def __init__(self, metrics_dir: str = "metrics"):
        """
        Initialize model monitor handler.
        
        Parameters
        ----------
        metrics_dir : str
            Directory to store metrics files
        """
        super().__init__()
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a record.
        
        Parameters
        ----------
        record : logging.LogRecord
            Log record
        """
        # Only process records with metrics
        if not hasattr(record, 'metrics'):
            return
        
        try:
            # Get metrics data
            metrics = record.metrics
            
            # Add timestamp
            metrics['timestamp'] = datetime.fromtimestamp(record.created).isoformat()
            
            # Add model info if available
            if hasattr(record, 'model_name'):
                metrics['model_name'] = record.model_name
            
            if hasattr(record, 'model_version'):
                metrics['model_version'] = record.model_version
            
            # Write metrics to file
            model_name = getattr(record, 'model_name', 'unknown')
            timestamp = datetime.now().strftime("%Y%m%d")
            metrics_file = os.path.join(self.metrics_dir, f"{model_name}_{timestamp}.jsonl")
            
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
                
        except Exception as e:
            # Don't raise exceptions from handler
            sys.stderr.write(f"Error in ModelMonitorHandler: {str(e)}\n")


def get_model_logger(
    model_name: str,
    metrics_dir: Optional[str] = None
) -> logging.Logger:
    """
    Get logger for model monitoring.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    metrics_dir : str, optional
        Directory to store metrics files
        
    Returns
    -------
    logging.Logger
        Model logger
    """
    # Get metrics directory from config if available
    if CONFIG_AVAILABLE and metrics_dir is None:
        metrics_dir = config.get('logging.metrics_dir', "metrics")
    elif metrics_dir is None:
        metrics_dir = "metrics"
    
    # Create logger
    logger = logging.getLogger(f"model.{model_name}")
    
    # Add model monitor handler if not already present
    has_monitor_handler = any(
        isinstance(handler, ModelMonitorHandler) 
        for handler in logger.handlers
    )
    
    if not has_monitor_handler:
        handler = ModelMonitorHandler(metrics_dir)
        logger.addHandler(handler)
    
    return logger


def log_model_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    level: int = logging.INFO
) -> None:
    """
    Log model metrics.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    metrics : dict
        Metrics to log
    model_name : str, optional
        Name of the model
    model_version : str, optional
        Version of the model
    level : int
        Log level
    """
    # Create record with metrics
    record = logging.LogRecord(
        name=logger.name,
        level=level,
        pathname='',
        lineno=0,
        msg=f"Model metrics: {metrics}",
        args=(),
        exc_info=None
    )
    
    # Add metrics and model info
    record.metrics = metrics
    
    if model_name:
        record.model_name = model_name
    
    if model_version:
        record.model_version = model_version
    
    # Process record
    logger.handle(record)
