"""
Model drift detection module for the forecasting system.

This module provides functionality to detect model drift and degradation over time,
allowing for automated alerts and retraining triggers when model performance
falls below acceptable thresholds.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from config.config_base import ConfigBase
from monitoring.model_monitor import get_monitor, PredictionRecord
from scripts.system_maintenance.logging_config import setup_logging

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class DriftMetrics:
    """Metrics for model drift detection."""
    # Statistical metrics
    ks_statistic: float = 0.0
    ks_pvalue: float = 1.0
    psi_value: float = 0.0  # Population Stability Index
    
    # Performance metrics
    rmse_ratio: float = 1.0  # Current RMSE / Baseline RMSE
    mae_ratio: float = 1.0   # Current MAE / Baseline MAE
    mape_ratio: float = 1.0  # Current MAPE / Baseline MAPE
    
    # Trend metrics
    rmse_trend: float = 0.0  # Slope of RMSE over time
    mae_trend: float = 0.0   # Slope of MAE over time
    mape_trend: float = 0.0  # Slope of MAPE over time
    
    # Drift detection flags
    distribution_drift: bool = False
    performance_drift: bool = False
    trend_drift: bool = False
    
    # Overall drift status
    drift_detected: bool = False
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {k: v for k, v in self.__dict__.items()}
        result['timestamp'] = result['timestamp'].isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DriftMetrics':
        """Create from dictionary."""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class ModelDriftDetector:
    """
    Detects model drift by analyzing prediction performance over time.
    
    This class monitors model predictions and compares recent performance
    against baseline metrics to detect potential model drift or degradation.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        baseline_window: int = 30,  # Days for baseline
        detection_window: int = 7,  # Days for current metrics
        thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the model drift detector.
        
        Args:
            storage_path: Path to store drift detection data
            baseline_window: Number of days to use for baseline metrics
            detection_window: Number of days to use for current metrics
            thresholds: Custom thresholds for drift detection
        """
        # Initialize configuration
        self.config = ConfigBase()
        
        # Use config or default path
        self.storage_path = storage_path or self.config.get(
            'monitoring.drift_detection_path', 
            'drift_detection'
        )
        os.makedirs(self.storage_path, exist_ok=True)
        
        self.baseline_window = baseline_window
        self.detection_window = detection_window
        
        # Default thresholds
        self.thresholds = {
            'ks_pvalue': 0.05,        # p-value threshold for KS test
            'psi_threshold': 0.2,      # PSI threshold
            'performance_ratio': 1.2,  # 20% degradation in performance
            'trend_slope': 0.01,       # Positive slope indicating degradation
            **(thresholds or {})
        }
        
        # Get monitor instance
        self.monitor = get_monitor()
        
        # Store drift metrics
        self.drift_metrics: Dict[str, Dict[str, List[DriftMetrics]]] = {}
        
        # Load existing drift metrics if available
        self._load_drift_metrics()
    
    def _load_drift_metrics(self) -> None:
        """Load drift metrics from storage."""
        metrics_path = os.path.join(self.storage_path, 'drift_metrics.json')
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                
                # Convert to DriftMetrics objects
                for model_name, series_data in data.items():
                    if model_name not in self.drift_metrics:
                        self.drift_metrics[model_name] = {}
                    
                    for series_id, metrics_list in series_data.items():
                        self.drift_metrics[model_name][series_id] = [
                            DriftMetrics.from_dict(m) for m in metrics_list
                        ]
                
                logger.info(f"Loaded drift metrics from {metrics_path}")
            except Exception as e:
                logger.error(f"Error loading drift metrics: {e}")
    
    def _save_drift_metrics(self) -> None:
        """Save drift metrics to storage."""
        metrics_path = os.path.join(self.storage_path, 'drift_metrics.json')
        try:
            # Convert to serializable format
            data = {}
            for model_name, series_data in self.drift_metrics.items():
                data[model_name] = {}
                for series_id, metrics_list in series_data.items():
                    data[model_name][series_id] = [
                        m.to_dict() for m in metrics_list
                    ]
            
            with open(metrics_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved drift metrics to {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving drift metrics: {e}")
    
    def calculate_psi(self, baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI) between baseline and current distributions.
        
        PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
        
        Args:
            baseline: Baseline distribution values
            current: Current distribution values
            bins: Number of bins for histogram
        
        Returns:
            PSI value (higher values indicate more drift)
        """
        # Create histograms
        baseline_hist, bin_edges = np.histogram(baseline, bins=bins, density=True)
        current_hist, _ = np.histogram(current, bins=bin_edges, density=True)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        baseline_hist = np.maximum(baseline_hist, epsilon)
        current_hist = np.maximum(current_hist, epsilon)
        
        # Normalize histograms
        baseline_pct = baseline_hist / baseline_hist.sum()
        current_pct = current_hist / current_hist.sum()
        
        # Calculate PSI
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        
        return psi
    
    def detect_drift(
        self, 
        model_name: str, 
        series_id: str,
        force_recalculation: bool = False
    ) -> DriftMetrics:
        """
        Detect drift for a specific model and series.
        
        Args:
            model_name: Name of the model
            series_id: ID of the time series
            force_recalculation: Force recalculation even if recent metrics exist
        
        Returns:
            DriftMetrics object with drift detection results
        """
        # Check if we have recent drift metrics (within the last day)
        if not force_recalculation and model_name in self.drift_metrics:
            if series_id in self.drift_metrics[model_name]:
                metrics_list = self.drift_metrics[model_name][series_id]
                if metrics_list and (datetime.now() - metrics_list[-1].timestamp).days < 1:
                    logger.info(f"Using recent drift metrics for {model_name}/{series_id}")
                    return metrics_list[-1]
        
        # Get predictions from monitor
        if model_name not in self.monitor.predictions or series_id not in self.monitor.predictions[model_name]:
            logger.warning(f"No predictions found for {model_name}/{series_id}")
            return DriftMetrics()
        
        predictions = self.monitor.predictions[model_name][series_id]
        if not predictions:
            logger.warning(f"Empty predictions for {model_name}/{series_id}")
            return DriftMetrics()
        
        # Convert to dataframe
        df = self.monitor.get_predictions_df(model_name, series_id)
        
        # Calculate baseline and current windows
        now = datetime.now()
        baseline_start = now - timedelta(days=self.baseline_window + self.detection_window)
        baseline_end = now - timedelta(days=self.detection_window)
        current_start = now - timedelta(days=self.detection_window)
        
        # Filter dataframes
        baseline_df = df[(df['timestamp'] >= baseline_start) & (df['timestamp'] < baseline_end)]
        current_df = df[(df['timestamp'] >= current_start)]
        
        # Check if we have enough data
        if len(baseline_df) < 10 or len(current_df) < 5:
            logger.warning(f"Insufficient data for drift detection: baseline={len(baseline_df)}, current={len(current_df)}")
            return DriftMetrics()
        
        # Calculate drift metrics
        drift_metrics = DriftMetrics()
        
        # 1. Distribution drift using KS test
        ks_statistic, ks_pvalue = stats.ks_2samp(
            baseline_df['error'].values, 
            current_df['error'].values
        )
        drift_metrics.ks_statistic = ks_statistic
        drift_metrics.ks_pvalue = ks_pvalue
        
        # 2. Calculate PSI
        psi_value = self.calculate_psi(
            baseline_df['error'].values, 
            current_df['error'].values
        )
        drift_metrics.psi_value = psi_value
        
        # 3. Performance metrics
        baseline_metrics = self._calculate_metrics(baseline_df)
        current_metrics = self._calculate_metrics(current_df)
        
        drift_metrics.rmse_ratio = current_metrics['rmse'] / baseline_metrics['rmse'] if baseline_metrics['rmse'] > 0 else 1.0
        drift_metrics.mae_ratio = current_metrics['mae'] / baseline_metrics['mae'] if baseline_metrics['mae'] > 0 else 1.0
        drift_metrics.mape_ratio = current_metrics['mape'] / baseline_metrics['mape'] if baseline_metrics['mape'] > 0 else 1.0
        
        # 4. Trend analysis
        # Group by day and calculate metrics
        df['date'] = df['timestamp'].dt.date
        trend_df = df.groupby('date').agg({
            'error': lambda x: np.sqrt(np.mean(np.square(x))),  # RMSE
        }).reset_index()
        
        if len(trend_df) >= 7:  # Need at least a week of data
            # Fit linear trend
            X = np.arange(len(trend_df)).reshape(-1, 1)
            y = trend_df['error'].values
            
            model = sm.OLS(y, sm.add_constant(X)).fit()
            drift_metrics.rmse_trend = model.params[1]  # Slope coefficient
        
        # 5. Set drift flags
        drift_metrics.distribution_drift = (
            drift_metrics.ks_pvalue < self.thresholds['ks_pvalue'] or 
            drift_metrics.psi_value > self.thresholds['psi_threshold']
        )
        
        drift_metrics.performance_drift = (
            drift_metrics.rmse_ratio > self.thresholds['performance_ratio'] or
            drift_metrics.mae_ratio > self.thresholds['performance_ratio'] or
            drift_metrics.mape_ratio > self.thresholds['performance_ratio']
        )
        
        drift_metrics.trend_drift = drift_metrics.rmse_trend > self.thresholds['trend_slope']
        
        # Overall drift status
        drift_metrics.drift_detected = (
            drift_metrics.distribution_drift or
            drift_metrics.performance_drift or
            drift_metrics.trend_drift
        )
        
        # Store drift metrics
        if model_name not in self.drift_metrics:
            self.drift_metrics[model_name] = {}
        
        if series_id not in self.drift_metrics[model_name]:
            self.drift_metrics[model_name][series_id] = []
        
        self.drift_metrics[model_name][series_id].append(drift_metrics)
        
        # Save drift metrics
        self._save_drift_metrics()
        
        # Log drift detection
        if drift_metrics.drift_detected:
            logger.warning(
                f"Model drift detected for {model_name}/{series_id}: "
                f"distribution={drift_metrics.distribution_drift}, "
                f"performance={drift_metrics.performance_drift}, "
                f"trend={drift_metrics.trend_drift}"
            )
        else:
            logger.info(f"No drift detected for {model_name}/{series_id}")
        
        return drift_metrics
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics from prediction dataframe."""
        errors = df['error'].values
        actuals = df['actual_value'].values
        
        # Avoid division by zero
        actuals_nonzero = np.where(np.abs(actuals) < 1e-10, 1e-10, actuals)
        
        metrics = {
            'rmse': np.sqrt(np.mean(np.square(errors))),
            'mae': np.mean(np.abs(errors)),
            'mape': np.mean(np.abs(errors / actuals_nonzero)) * 100
        }
        
        return metrics
    
    def generate_drift_report(
        self, 
        model_name: str, 
        series_id: str,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Generate a drift report for a specific model and series.
        
        Args:
            model_name: Name of the model
            series_id: ID of the time series
            output_dir: Directory to save the report
        
        Returns:
            Path to the generated report
        """
        # Detect drift if not already done
        drift_metrics = self.detect_drift(model_name, series_id)
        
        # Get predictions dataframe
        df = self.monitor.get_predictions_df(model_name, series_id)
        
        # Create output directory
        output_dir = output_dir or os.path.join(self.storage_path, 'reports')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(
            output_dir, 
            f"drift_report_{model_name}_{series_id}_{timestamp}.html"
        )
        
        # Create report
        with open(report_path, 'w') as f:
            f.write(f"""
            <html>
            <head>
                <title>Model Drift Report: {model_name}/{series_id}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .metrics {{ display: flex; flex-wrap: wrap; }}
                    .metric-card {{ 
                        background: #f5f5f5; 
                        border-radius: 5px; 
                        padding: 15px; 
                        margin: 10px; 
                        width: 200px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                    .drift-detected {{ color: #d9534f; }}
                    .no-drift {{ color: #5cb85c; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                </style>
            </head>
            <body>
                <h1>Model Drift Report</h1>
                <p><strong>Model:</strong> {model_name}</p>
                <p><strong>Series:</strong> {series_id}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Drift Status</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <h3>Overall Drift</h3>
                        <div class="metric-value {('drift-detected' if drift_metrics.drift_detected else 'no-drift')}">
                            {'Detected' if drift_metrics.drift_detected else 'Not Detected'}
                        </div>
                    </div>
                    <div class="metric-card">
                        <h3>Distribution Drift</h3>
                        <div class="metric-value {('drift-detected' if drift_metrics.distribution_drift else 'no-drift')}">
                            {'Detected' if drift_metrics.distribution_drift else 'Not Detected'}
                        </div>
                    </div>
                    <div class="metric-card">
                        <h3>Performance Drift</h3>
                        <div class="metric-value {('drift-detected' if drift_metrics.performance_drift else 'no-drift')}">
                            {'Detected' if drift_metrics.performance_drift else 'Not Detected'}
                        </div>
                    </div>
                    <div class="metric-card">
                        <h3>Trend Drift</h3>
                        <div class="metric-value {('drift-detected' if drift_metrics.trend_drift else 'no-drift')}">
                            {'Detected' if drift_metrics.trend_drift else 'Not Detected'}
                        </div>
                    </div>
                </div>
                
                <h2>Drift Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Threshold</th>
                        <th>Status</th>
                    </tr>
                    <tr>
                        <td>KS Test p-value</td>
                        <td>{drift_metrics.ks_pvalue:.4f}</td>
                        <td>&lt; {self.thresholds['ks_pvalue']}</td>
                        <td>{('Drift' if drift_metrics.ks_pvalue < self.thresholds['ks_pvalue'] else 'No Drift')}</td>
                    </tr>
                    <tr>
                        <td>Population Stability Index (PSI)</td>
                        <td>{drift_metrics.psi_value:.4f}</td>
                        <td>&gt; {self.thresholds['psi_threshold']}</td>
                        <td>{('Drift' if drift_metrics.psi_value > self.thresholds['psi_threshold'] else 'No Drift')}</td>
                    </tr>
                    <tr>
                        <td>RMSE Ratio (Current/Baseline)</td>
                        <td>{drift_metrics.rmse_ratio:.4f}</td>
                        <td>&gt; {self.thresholds['performance_ratio']}</td>
                        <td>{('Drift' if drift_metrics.rmse_ratio > self.thresholds['performance_ratio'] else 'No Drift')}</td>
                    </tr>
                    <tr>
                        <td>MAE Ratio (Current/Baseline)</td>
                        <td>{drift_metrics.mae_ratio:.4f}</td>
                        <td>&gt; {self.thresholds['performance_ratio']}</td>
                        <td>{('Drift' if drift_metrics.mae_ratio > self.thresholds['performance_ratio'] else 'No Drift')}</td>
                    </tr>
                    <tr>
                        <td>MAPE Ratio (Current/Baseline)</td>
                        <td>{drift_metrics.mape_ratio:.4f}</td>
                        <td>&gt; {self.thresholds['performance_ratio']}</td>
                        <td>{('Drift' if drift_metrics.mape_ratio > self.thresholds['performance_ratio'] else 'No Drift')}</td>
                    </tr>
                    <tr>
                        <td>RMSE Trend Slope</td>
                        <td>{drift_metrics.rmse_trend:.6f}</td>
                        <td>&gt; {self.thresholds['trend_slope']}</td>
                        <td>{('Drift' if drift_metrics.rmse_trend > self.thresholds['trend_slope'] else 'No Drift')}</td>
                    </tr>
                </table>
                
                <h2>Recommendations</h2>
                <ul>
                    {'<li><strong>Retrain Model:</strong> Performance has degraded significantly. Consider retraining the model with recent data.</li>' if drift_metrics.performance_drift else ''}
                    {'<li><strong>Investigate Data Sources:</strong> Distribution drift detected. Check for changes in data sources or patterns.</li>' if drift_metrics.distribution_drift else ''}
                    {'<li><strong>Monitor Closely:</strong> Trend drift detected. Continue monitoring as performance may degrade further.</li>' if drift_metrics.trend_drift else ''}
                    {'<li><strong>No Action Required:</strong> Model is performing as expected.</li>' if not drift_metrics.drift_detected else ''}
                </ul>
            </body>
            </html>
            """)
        
        logger.info(f"Generated drift report at {report_path}")
        return report_path
    
    def visualize_drift(
        self, 
        model_name: str, 
        series_id: str,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Generate visualizations for model drift.
        
        Args:
            model_name: Name of the model
            series_id: ID of the time series
            output_dir: Directory to save visualizations
        
        Returns:
            Path to the generated visualization file
        """
        # Get predictions dataframe
        df = self.monitor.get_predictions_df(model_name, series_id)
        
        if df.empty:
            logger.warning(f"No predictions found for {model_name}/{series_id}")
            return ""
        
        # Create output directory
        output_dir = output_dir or os.path.join(self.storage_path, 'visualizations')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualization filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_path = os.path.join(
            output_dir, 
            f"drift_viz_{model_name}_{series_id}_{timestamp}.png"
        )
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        
        # 1. Error over time
        df['date'] = df['timestamp'].dt.date
        daily_errors = df.groupby('date').agg({
            'error': ['mean', 'std', lambda x: np.sqrt(np.mean(np.square(x)))],
            'timestamp': 'first'
        })
        daily_errors.columns = ['mean_error', 'std_error', 'rmse', 'timestamp']
        daily_errors = daily_errors.reset_index()
        
        axs[0].plot(daily_errors['date'], daily_errors['rmse'], 'o-', label='RMSE')
        axs[0].plot(daily_errors['date'], daily_errors['mean_error'], 'o-', label='Mean Error')
        axs[0].fill_between(
            daily_errors['date'],
            daily_errors['mean_error'] - daily_errors['std_error'],
            daily_errors['mean_error'] + daily_errors['std_error'],
            alpha=0.2
        )
        axs[0].set_title('Prediction Error Over Time')
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Error')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        # 2. Error distribution comparison (baseline vs recent)
        now = datetime.now()
        baseline_start = now - timedelta(days=self.baseline_window + self.detection_window)
        baseline_end = now - timedelta(days=self.detection_window)
        current_start = now - timedelta(days=self.detection_window)
        
        baseline_df = df[(df['timestamp'] >= baseline_start) & (df['timestamp'] < baseline_end)]
        current_df = df[(df['timestamp'] >= current_start)]
        
        if not baseline_df.empty and not current_df.empty:
            axs[1].hist(
                baseline_df['error'], 
                bins=30, 
                alpha=0.5, 
                label=f'Baseline ({baseline_start.date()} to {baseline_end.date()})'
            )
            axs[1].hist(
                current_df['error'], 
                bins=30, 
                alpha=0.5, 
                label=f'Recent ({current_start.date()} to {now.date()})'
            )
            axs[1].set_title('Error Distribution Comparison')
            axs[1].set_xlabel('Error')
            axs[1].set_ylabel('Frequency')
            axs[1].legend()
            axs[1].grid(True, alpha=0.3)
        
        # 3. Actual vs Predicted over time
        recent_df = df.sort_values('timestamp').tail(100)  # Last 100 predictions
        axs[2].plot(recent_df['timestamp'], recent_df['actual_value'], 'o-', label='Actual')
        axs[2].plot(recent_df['timestamp'], recent_df['predicted_value'], 'o-', label='Predicted')
        axs[2].set_title('Actual vs Predicted Values (Recent)')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Value')
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)
        
        # Add drift detection status
        drift_metrics = self.detect_drift(model_name, series_id)
        drift_status = "DRIFT DETECTED" if drift_metrics.drift_detected else "NO DRIFT DETECTED"
        fig.suptitle(
            f"Model Drift Analysis: {model_name}/{series_id}\n{drift_status}",
            fontsize=16
        )
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_path)
        plt.close()
        
        logger.info(f"Generated drift visualization at {viz_path}")
        return viz_path
    
    def check_all_models(self) -> Dict[str, Dict[str, DriftMetrics]]:
        """
        Check drift for all models and series in the monitor.
        
        Returns:
            Dictionary mapping model names and series IDs to drift metrics
        """
        results = {}
        
        for model_name, series_dict in self.monitor.predictions.items():
            results[model_name] = {}
            
            for series_id in series_dict.keys():
                drift_metrics = self.detect_drift(model_name, series_id)
                results[model_name][series_id] = drift_metrics
        
        return results
    
    def get_drift_history(
        self, 
        model_name: str, 
        series_id: str
    ) -> List[DriftMetrics]:
        """
        Get drift history for a specific model and series.
        
        Args:
            model_name: Name of the model
            series_id: ID of the time series
        
        Returns:
            List of DriftMetrics objects
        """
        if model_name in self.drift_metrics and series_id in self.drift_metrics[model_name]:
            return self.drift_metrics[model_name][series_id]
        return []


# Singleton instance
_drift_detector = None

def get_drift_detector() -> ModelDriftDetector:
    """Get the singleton drift detector instance."""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = ModelDriftDetector()
    return _drift_detector
