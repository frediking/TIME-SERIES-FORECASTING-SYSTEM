import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from dataclasses import asdict
import numpy as np
import pandas as pd

@dataclass
class PredictionRecord:
    model_name: str
    series_id: str
    timestamp: datetime
    predicted_value: float
    actual_value: Optional[float] = None
    prediction_interval_lower: Optional[float] = None
    prediction_interval_upper: Optional[float] = None
    features: Optional[Dict[str, float]] = None
    model_version: Optional[str] = None
    region: Optional[str] = None
    prediction_horizon: Optional[int] = None
    error: Optional[float] = None

class ModelMonitor:
    """
    Tracks model predictions and performance metrics.
    
    Methods:
        record_prediction: Stores a prediction with metadata
        check_drift: Runs drift detection on recent predictions
    """
    
    def __init__(self, storage_dir=None):
        self.storage_dir = Path(storage_dir) if storage_dir else Path("./monitoring_data")
        self.predictions = {}  # {model_name: {series_id: [records]}}
        self.metrics = {}  # {model_name: {series_id: {metrics}}}
        
    def record_prediction(self, record):
        """Record a prediction and update metrics."""
        if record.model_name not in self.predictions:
            self.predictions[record.model_name] = {}
        if record.series_id not in self.predictions[record.model_name]:
            self.predictions[record.model_name][record.series_id] = []
        self.predictions[record.model_name][record.series_id].append(record)
        
        # Initialize metrics
        if record.model_name not in self.metrics:
            self.metrics[record.model_name] = {}
        if record.series_id not in self.metrics[record.model_name]:
            self.metrics[record.model_name][record.series_id] = {
                'RMSE': 0,
                'MAE': 0,
                'MAPE': 0,
                'count': 0
            }
        
        # Update count
        self.metrics[record.model_name][record.series_id]['count'] += 1
        
        # Calculate metrics
        if record.actual_value is not None and record.predicted_value is not None:
            error = record.actual_value - record.predicted_value
            self.metrics[record.model_name][record.series_id]['RMSE'] = error**2
            self.metrics[record.model_name][record.series_id]['MAE'] = abs(error)
            if record.actual_value != 0:  # Avoid division by zero
                self.metrics[record.model_name][record.series_id]['MAPE'] = (
                    abs(error)/record.actual_value * 100
                )

    def get_metrics(self, model_name: str, series_id: str) -> Dict[str, float]:
        """Get metrics for a specific model and series."""
        return self.metrics.get(model_name, {}).get(series_id, {})
            
    def get_predictions(
        self,
        model_name: str,
        series_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[PredictionRecord]:
        """Get predictions for a model/series within a time range."""
        preds = self.predictions.get(model_name, {}).get(series_id, [])
        if start_time or end_time:
            return [
                p for p in preds
                if (not start_time or p.timestamp >= start_time) and
                   (not end_time or p.timestamp <= end_time)
            ]
        return preds

    
    # Add after get_predictions() method
    def get_predictions_df(self, model_name=None, series=None) -> pd.DataFrame:
        """Convert stored predictions to pandas DataFrame with datetime index.
        
        Returns:
            DataFrame with columns:
            - model: Model name
            - series: Series ID  
            - timestamp: Prediction time
            - actual: Actual values
            - predicted: Predicted values
            - feature_*: All recorded features
        """
        records = []
        for m_name, series_data in self.predictions.items():
            if model_name and m_name != model_name:
                continue
            for s_name, pred_records in series_data.items():
                if series and s_name != series:
                    continue
                for record in pred_records:
                    row = {
                        'model': m_name,
                        'series': s_name,
                        'timestamp': record.timestamp,
                        'actual': record.actual_value,
                        'predicted': record.predicted_value
                    }
                    row.update({f'feature_{k}': v for k, v in record.features.items()})
                    records.append(row)
        if not records:
            return pd.DataFrame()
            
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.set_index('timestamp')
    
    # Modify save() method
    def save(self):
        """Save monitor state to disk."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.storage_dir / "monitor_state.json"
        with open(state_path, 'w') as f:
            json.dump({
                'predictions': {
                    model_name: {
                        series_id: [
                            {
                                **asdict(record),
                                'timestamp': record.timestamp.isoformat()  # Convert datetime
                            }
                            for record in records
                        ]
                        for series_id, records in series_data.items()
                    }
                    for model_name, series_data in self.predictions.items()
                },
                'metrics': self.metrics
            }, f, indent=2)



    def load(self):
        """Load monitor state from disk."""
        state_path = self.storage_dir / "monitor_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
                self.predictions = {
                    model: {
                        series: [PredictionRecord(**p) for p in preds]
                        for series, preds in series_data.items()
                    }
                    for model, series_data in state['predictions'].items()
                }
                self.metrics = state['metrics']

    def generate_report(
        self,
        model_name: str,
        series_id: str,
        output_dir: Optional[str] = None
    ) -> Path:
        """Generate report for specific model and series."""
        output_dir = Path(output_dir) if output_dir else self.storage_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / f"{model_name}_performance_report.html"
        
        # Get metrics
        metrics = self.metrics.get(model_name, {}).get(series_id, {})
        
        # Generate report content
        report_content = f"""
        <html>
        <head><title>Performance Report</title></head>
        <body>
            <h1>Performance Report for {model_name}</h1>
            <h2>Series: {series_id}</h2>
            <p>RMSE: {metrics.get('RMSE', 'N/A')}</p>
            <p>MAE: {metrics.get('MAE', 'N/A')}</p>
            <p>MAPE: {metrics.get('MAPE', 'N/A')}%</p>
            <p>Total predictions: {len(self.predictions.get(model_name, {}).get(series_id, []))}</p>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        return report_path

    def check_drift(self) -> Dict[str, Dict[str, Any]]:
        """Check for model drift across all monitored metrics."""
        drift_results = {}
        
        for model_name, series_data in self.predictions.items():
            drift_results[model_name] = {}
            
            for series_id, records in series_data.items():
                if len(records) < 100:
                    continue
                    
                window_size = max(100, len(records) // 10)
                recent = records[-window_size:]
                baseline = records[:-window_size]
                
                # 1. Prediction drift
                baseline_preds = [r.predicted_value for r in baseline]
                recent_preds = [r.predicted_value for r in recent]
                pred_psi = self._calculate_psi(baseline_preds, recent_preds)
                
                # 2. Feature shifts
                feature_shifts = {}
                if records[0].features:
                    for feat in records[0].features:
                        baseline_feats = [r.features[feat] for r in baseline]
                        recent_feats = [r.features[feat] for r in recent]
                        feature_shifts[feat] = self._calculate_psi(baseline_feats, recent_feats)
                
                # 3. Performance degradation
                perf_degradation = False
                if records[0].actual_value is not None:
                    # Calculate baseline MAE (more stable than MAPE)
                    baseline_errors = [abs(r.actual_value - r.predicted_value) for r in baseline]
                    baseline_mae = sum(baseline_errors)/len(baseline_errors)
                    
                    # Calculate recent MAE
                    recent_errors = [abs(r.actual_value - r.predicted_value) for r in recent]
                    recent_mae = sum(recent_errors)/len(recent_errors)
                    
                    # Degradation if MAE increases by 50% or more
                    perf_degradation = recent_mae > baseline_mae * 1.5
                
                drift_results[model_name][series_id] = {
                    'prediction_drift': pred_psi > 0.1,  # Lower threshold
                    'feature_shift': {f: psi > 0.1 for f, psi in feature_shifts.items()},
                    'performance_degradation': perf_degradation
                }
        
        return drift_results
    def _calculate_psi(self, baseline: List[float], recent: List[float], bins: int = 10) -> float:
        """Calculate Population Stability Index between two distributions."""
        # Bin both distributions
        min_val = min(min(baseline), min(recent))
        max_val = max(max(baseline), max(recent))
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        baseline_hist = np.histogram(baseline, bins=bin_edges)[0]
        recent_hist = np.histogram(recent, bins=bin_edges)[0]
        
        # Add small value to avoid division by zero
        baseline_hist = baseline_hist + 0.0001
        recent_hist = recent_hist + 0.0001
        
        # Normalize to probabilities
        baseline_probs = baseline_hist / sum(baseline_hist)
        recent_probs = recent_hist / sum(recent_hist)
        
        # Calculate PSI
        psi = sum((recent_probs - baseline_probs) * np.log(recent_probs / baseline_probs))
        return psi


# Singleton implementation
_monitor_instance = None

def get_monitor() -> ModelMonitor:
    """Get the singleton ModelMonitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ModelMonitor()
    return _monitor_instance