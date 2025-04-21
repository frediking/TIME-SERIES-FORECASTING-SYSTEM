import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sequences(data: pd.Series, 
                    seq_length: int, 
                    target_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction.
    
    Args:
        data: Time series data
        seq_length: Length of input sequence
        target_horizon: Number of future steps to predict
    """
    sequences, targets = [], []
    for i in range(len(data) - seq_length - target_horizon + 1):
        seq = data[i:(i + seq_length)]
        target = data[(i + seq_length):(i + seq_length + target_horizon)]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def calculate_metrics(actual: np.ndarray, 
                     predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate various forecasting metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'MSE': mean_squared_error(actual, predicted),
        'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
        'MAE': mean_absolute_error(actual, predicted),
        'R2': r2_score(actual, predicted)
    }
    
    # Calculate MAPE only if there are no zeros in actual values
    if not np.any(actual == 0):
        metrics['MAPE'] = np.mean(np.abs((actual - predicted) / actual)) * 100
    else:
        # Use a small epsilon to avoid division by zero
        epsilon = 1e-10
        metrics['MAPE'] = np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100
    
    return metrics

def plot_forecast(y_true, forecasts, title='Forecast Comparison'):
    """
    Plot actual vs predicted values with confidence intervals.
    
    Args:
        y_true: Actual values
        forecasts: Predicted values
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Plot actual values
    fig.add_trace(go.Scatter(
        y=y_true,
        name='Actual',
        line=dict(color='blue')
    ))
    
    # Plot predictions
    fig.add_trace(go.Scatter(
        y=forecasts,
        name='Forecast',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time Steps',
        yaxis_title='Value',
        template='plotly_white'
    )
    
    return fig

def plot_forecast_plotly(y_true, forecasts, title='Forecast Comparison'):
    """
    Plot actual vs forecasted values using Plotly.
    
    Args:
        y_true: Actual values
        forecasts: Dictionary of forecasts from different models
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        y=y_true,
        mode='lines',
        name='Actual',
        line=dict(width=2)
    ))
    
    # Add forecasts from different models
    for model_name, y_pred in forecasts.items():
        fig.add_trace(go.Scatter(
            y=y_pred,
            mode='lines',
            name=f'{model_name}',
            line=dict(width=1.5)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time Steps',
        yaxis_title='Value',
        legend=dict(x=0, y=1),
        template='plotly_white'
    )
    
    return fig

def plot_feature_importance(importance, feature_names, title='Feature Importance'):
    """
    Plot feature importance from tree-based models.
    
    Args:
        importance: Feature importance values
        feature_names: Names of features
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(title)
    plt.tight_layout()
    
    return plt

def plot_feature_importance_plotly(importance, feature_names, title='Feature Importance'):
    """
    Plot feature importance from tree-based models using Plotly.
    
    Args:
        importance: Feature importance values
        feature_names: Names of features
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=importance_df['Feature'],
        x=importance_df['Importance'],
        orientation='h'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Importance',
        yaxis_title='Feature',
        template='plotly_white'
    )
    
    return fig

def create_evaluation_report(agent, test_data, target_column):
    """
    Create comprehensive evaluation report for all models.
    
    Args:
        agent: ForecastingAgent instance
        test_data: Test dataset
        target_column: Target column name
        
    Returns:
        Dictionary of metrics for each model and series
    """
    report = {}
    
    for name, models in agent.models.items():
        # Get actual values
        y_true = test_data[test_data['series_id'] == name[0]][test_data['region'] == name[1]][target_column].values
        
        # Get predictions from each model
        forecasts = agent.forecast(horizon=len(y_true))
        y_pred = forecasts[name]
        
        # Calculate metrics for each model
        model_metrics = {}
        for model_name, predictions in y_pred.items():
            # Trim predictions to match actual length if needed
            pred = predictions[:len(y_true)]
            metrics = calculate_metrics(y_true, pred)
            model_metrics[model_name] = metrics
        
        # Store metrics for this series
        report[name] = model_metrics
    
    return report

def save_evaluation_results(report, path='evaluation_results.json'):
    """
    Save evaluation results to a JSON file.
    
    Args:
        report: Evaluation report dictionary
        path: Path to save the results
    """
    # Convert tuple keys to strings
    serializable_report = {}
    for name, metrics in report.items():
        key = f"{name[0]}_{name[1]}"
        serializable_report[key] = metrics
    
    # Save to JSON
    import json
    with open(path, 'w') as f:
        json.dump(serializable_report, f, indent=4)
    
    print(f"Evaluation results saved to {path}")

def decompose_time_series(series, period=24, model='additive'):
    """
    Decompose time series into trend, seasonal, and residual components.
    
    Args:
        series: Time series data
        period: Seasonality period
        model: Decomposition model ('additive' or 'multiplicative')
        
    Returns:
        Decomposition result
    """
    decomposition = seasonal_decompose(series, model=model, period=period)
    return decomposition

def plot_decomposition(decomposition, title='Time Series Decomposition'):
    """
    Plot time series decomposition.
    
    Args:
        decomposition: Decomposition result
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
    
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    ax1.set_xlabel('')
    
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    ax2.set_xlabel('')
    
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    ax3.set_xlabel('')
    
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)
    
    return fig

def plot_decomposition_plotly(decomposition, title='Time Series Decomposition'):
    """
    Plot time series decomposition using Plotly.
    
    Args:
        decomposition: Decomposition result
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Create subplots
    fig = go.Figure()
    
    # Add observed data
    fig.add_trace(go.Scatter(
        y=decomposition.observed,
        mode='lines',
        name='Observed'
    ))
    
    # Add trend
    fig.add_trace(go.Scatter(
        y=decomposition.trend,
        mode='lines',
        name='Trend'
    ))
    
    # Add seasonal component
    fig.add_trace(go.Scatter(
        y=decomposition.seasonal,
        mode='lines',
        name='Seasonal'
    ))
    
    # Add residual component
    fig.add_trace(go.Scatter(
        y=decomposition.resid,
        mode='lines',
        name='Residual'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Value',
        template='plotly_white'
    )
    
    return fig

def analyze_seasonality(data: pd.Series, 
                       period: int = None) -> Dict[str, pd.Series]:
    """
    Perform seasonal decomposition of time series.
    """
    if period is None:
        # Attempt to automatically detect seasonality
        period = detect_seasonality(data)
    
    try:
        result = seasonal_decompose(
            data,
            period=period,
            model='additive'
        )
        return {
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.resid
        }
    except Exception as e:
        logger.error(f"Seasonality analysis failed: {str(e)}")
        return None

def detect_seasonality(data: pd.Series) -> int:
    """
    Automatically detect seasonality period using FFT.
    """
    fft = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(data))
    magnitudes = np.abs(fft)
    
    # Exclude the zero frequency
    frequencies = frequencies[1:]
    magnitudes = magnitudes[1:]
    
    # Find the frequency with maximum magnitude
    peak_frequency_idx = np.argmax(magnitudes)
    period = int(1 / abs(frequencies[peak_frequency_idx]))
    
    return period

def evaluate_forecast(actual: np.ndarray, 
                     predicted: np.ndarray, 
                     plot: bool = True) -> Dict:
    """
    Comprehensive forecast evaluation.
    """
    # Calculate metrics
    metrics = calculate_metrics(actual, predicted)
    
    # Generate plots if requested
    if plot:
        fig = plot_forecast_plotly(actual, predicted)
        fig.show()
    
    # Calculate additional statistics
    metrics.update({
        'bias': np.mean(predicted - actual),
        'forecast_accuracy': 1 - np.mean(np.abs((actual - predicted) / actual))
    })
    
    return metrics

def save_evaluation_results(results: Dict, 
                          filepath: str = 'evaluation_results.json'):
    """
    Save evaluation results to file.
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        pd.DataFrame(results).to_json(output_path)
        logger.info(f"Evaluation results saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {str(e)}")