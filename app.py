import os
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHER"] = "false"

import torch
torch._C._disable_streamlit_watcher = True  # Prevent Streamlit from inspecting torch internals

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import logging
from datetime import datetime
from pathlib import Path
import json
import matplotlib.pyplot as plt

# Add parent directory to path to import agent modules
sys.path.append(str(Path(__file__).parent.parent))
from config.config import Config
from agent.ensemble import ForecastingEnsemble
from models.registry.model_registry import registry
from scripts.system_maintenance.logging_config import setup_logging
from monitoring.model_monitor import get_monitor, PredictionRecord

# Configure logging
setup_logging(
    log_dir=Config().get('logging.log_dir', 'logs'),
    log_level=Config().get('logging.log_level', 'INFO')
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Time Series Forecasting Agent",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = ForecastingEnsemble()
    logger.info("Initialized ForecastingEnsemble in session state")
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = None
if 'selected_series' not in st.session_state:
    st.session_state.selected_series = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'monitor' not in st.session_state:
    st.session_state.monitor = get_monitor()

# Initialize monitor
monitor = get_monitor()

def display_monitoring_metrics():
    """Display model monitoring metrics in Streamlit."""
    st.header("Model Monitoring")
    
    # Get recent predictions
    recent_metrics = monitor.get_recent_metrics()
    if recent_metrics:
        st.write("### Recent Model Performance")
        st.dataframe(recent_metrics)
    
    # Show prediction plots
    st.write("### Prediction Analysis")
    fig = monitor.plot_performance_over_time()
    st.plotly_chart(fig)

def load_data(uploaded_file):
    """
    Load and preprocess data from uploaded file.
    
    Parameters
    ----------
    uploaded_file : UploadedFile
        Uploaded file object
        
    Returns
    -------
    pd.DataFrame
        Loaded and preprocessed dataframe
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        # Check for required columns
        required_columns = ['timestamp', 'series_id', 'region']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}", exc_info=True)
        st.error(f"Error loading data: {str(e)}")
        return None

def train_models(df, target_column, use_hyperparameters=False, hyperparameters_path=None):
    """
    Train forecasting models.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Target column name
    use_hyperparameters : bool
        Whether to use hyperparameters
    hyperparameters_path : str, optional
        Path to hyperparameters file
        
    Returns
    -------
    dict
        Trained models
    """
    try:
        with st.spinner("Training models... This may take a few minutes."):
            models = st.session_state.agent.train_model(
                df=df,
                target_column=target_column,
                use_hyperparameters=use_hyperparameters,
                hyperparameters_path=hyperparameters_path
            )
            
            st.session_state.trained = True
            logger.info(f"Models trained successfully for {len(models)} series")
            return models
    
    except Exception as e:
        logger.error(f"Error training models: {str(e)}", exc_info=True)
        st.error(f"Error training models: {str(e)}")
        return None

def generate_forecasts(horizon):
    """
    Generate forecasts.
    
    Parameters
    ----------
    horizon : int
        Forecast horizon
        
    Returns
    -------
    dict
        Forecasts
    """
    try:
        if not st.session_state.trained:
            st.error("Models not trained yet!")
            return None
            
        st.write("## Prediction Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Get prediction start time
        start_time = datetime.now()
        status_text.text(f"Starting predictions at {start_time.strftime('%H:%M:%S')}")
        
        # Generate forecasts
        forecasts = st.session_state.agent.predict(horizon)
        
        # Show completion
        duration = (datetime.now() - start_time).total_seconds()
        progress_bar.progress(100)
        status_text.text(f"✓ Predictions completed in {duration:.2f} seconds")
        
        # Show prediction stats
        with st.expander("Prediction Details"):
            st.write(f"**Forecast Horizon:** {horizon} steps")
            st.write(f"**Models Used:** {len(st.session_state.agent.models)} models")
            st.write(f"**Time Series Count:** {len(forecasts)} series")
            
            # Show sample prediction
            if forecasts:
                sample_series = next(iter(forecasts.keys()))
                st.write("**Sample Prediction:**")
                st.dataframe(forecasts[sample_series].head())
        
        logger.info(f"Generated forecasts for horizon {horizon}")
        return forecasts
        
    except Exception as e:
        logger.error(f"Error generating forecasts: {str(e)}", exc_info=True)
        st.error(f"Error generating forecasts: {str(e)}")
        return None

def plot_series_forecast(series_name, forecasts, history):
    """
    Plot forecast for a specific series.
    
    Parameters
    ----------
    series_name : tuple
        (series_id, region)
    forecasts : dict
        Forecasts dictionary
    history : dict
        History dictionary
    """
    try:
        if series_name not in forecasts:
            st.warning(f"No forecast available for {series_name}")
            return
        
        # Get historical data
        historical_data = history[series_name]['test_data']
        target_column = st.session_state.agent.target_column
        
        # Create figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_data['timestamp'],
            y=historical_data[target_column],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Get forecast timestamps
        last_timestamp = historical_data['timestamp'].iloc[-1]
        forecast_timestamps = pd.date_range(
            start=last_timestamp,
            periods=len(forecasts[series_name]['ensemble']) + 1,
            freq=pd.infer_freq(historical_data['timestamp'])
        )[1:]
        
        # Add forecasts for each model
        colors = {
            'xgboost': 'green',
            'lightgbm': 'orange',
            'lstm': 'red',
            'statistical': 'purple',
            'ensemble': 'black'
        }
        
        for model_name, forecast in forecasts[series_name].items():
            fig.add_trace(go.Scatter(
                x=forecast_timestamps,
                y=forecast,
                mode='lines',
                name=f'{model_name.capitalize()} Forecast',
                line=dict(color=colors.get(model_name, 'gray'))
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Forecast for {series_name[0]} - {series_name[1]}",
            xaxis_title="Timestamp",
            yaxis_title=target_column,
            legend_title="Models",
            height=600
        )
        
        # Log forecast to monitoring system
        model_version = Config().get_model_version()
        for model_name, forecast in forecasts[series_name].items():
            st.session_state.monitor.log_prediction(
                PredictionRecord(
                    timestamp=datetime.now().isoformat(),
                    model_name=f"{model_name}_{series_name[0]}_{series_name[1]}",
                    model_version=f"v{model_version}",
                    series_id=series_name[0],
                    region=series_name[1],
                    prediction_horizon=len(forecast),
                    predicted_value=float(forecast[0]),
                    prediction_interval_lower=None,
                    prediction_interval_upper=None
                )
            )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting forecast: {str(e)}", exc_info=True)
        st.error(f"Error plotting forecast: {str(e)}")
        return None

def evaluate_models(forecasts, history):
    """
    Evaluate model performance.
    
    Parameters
    ----------
    forecasts : dict
        Forecasts dictionary
    history : dict
        History dictionary
        
    Returns
    -------
    dict
        Evaluation results
    """
    try:
        results = {}
        target_column = st.session_state.agent.target_column
        
        for name, series_forecasts in forecasts.items():
            # Get actual values
            actual = history[name]['test_data'][target_column].values
            
            # Calculate metrics for each model
            results[name] = {}
            
            for model_name, forecast in series_forecasts.items():
                # Limit forecast to actual length
                forecast = forecast[:len(actual)]
                
                # Calculate metrics
                mse = np.mean((actual - forecast) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(actual - forecast))
                
                # Calculate MAPE (avoiding division by zero)
                mask = actual != 0
                mape = np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100 if mask.any() else np.nan
                
                results[name][model_name] = {
                    'MSE': float(mse),
                    'RMSE': float(rmse),
                    'MAE': float(mae),
                    'MAPE': float(mape) if not np.isnan(mape) else None
                }
        
        logger.info(f"Model evaluation completed for {len(results)} series")
        return results
    
    except Exception as e:
        logger.error(f"Error evaluating models: {str(e)}", exc_info=True)
        st.error(f"Error evaluating models: {str(e)}")
        return None

def save_models():
    """Save trained models to disk."""
    try:
        with st.spinner("Saving models..."):
            models_dir = Config().get('paths.models_dir', 'models')
            st.session_state.agent.save_models(path=models_dir)
            st.success(f"Models saved to {models_dir}")
            logger.info(f"Models saved to {models_dir}")
    
    except Exception as e:
        logger.error(f"Error saving models: {str(e)}", exc_info=True)
        st.error(f"Error saving models: {str(e)}")

def load_models():
    """Load trained models from disk."""
    try:
        with st.spinner("Loading models..."):
            models_dir = Config().get('paths.models_dir', 'models')
            models = st.session_state.agent.load_models(path=models_dir)
            
            if models:
                st.session_state.trained = True
                st.success(f"Loaded models for {len(models)} series")
                logger.info(f"Loaded models for {len(models)} series")
            else:
                st.warning("No models found")
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}", exc_info=True)
        st.error(f"Error loading models: {str(e)}")

def show_monitoring_dashboard():
    """Enhanced monitoring dashboard with drift detection."""
    st.header("Real-time Prediction Monitoring")
    
    # Get detailed model metrics
    model_metrics = monitor.get_detailed_model_metrics()
    
    if not model_metrics:
        st.warning("No trained models found. Please train models first to see monitoring data.")
        # Show basic system stats even without models
        st.subheader("System Resources")
        cpu, mem = monitor.get_system_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CPU Usage", f"{cpu}%")
        with col2:
            st.metric("Memory Usage", f"{mem}%")
        return
    
    # Show metrics in tabs - one tab per model type
    tab_names = list(model_metrics.keys())
    tabs = st.tabs(tab_names)
    
    for tab, model_type in zip(tabs, tab_names):
        with tab:
            metrics = model_metrics[model_type]
            
            # Key metrics summary
            cols = st.columns(5)  
            cols[0].metric("Total Predictions", metrics['total_predictions'])
            cols[1].metric("Avg Latency", f"{metrics['avg_latency']:.1f} ms")
            cols[2].metric("Success Rate", f"{metrics['success_rate']*100:.1f}%")
            cols[3].metric("Last Used", metrics['last_used'])
            cols[4].metric("Days Since Training", metrics.get('days_since_training', 'N/A'))
            
            # New confidence display
            if metrics.get('avg_confidence'):
                st.progress(float(metrics['avg_confidence']), 
                           text=f"Avg Confidence: {metrics['avg_confidence']*100:.1f}%")
            
            # Detailed metrics
            with st.expander("Detailed Metrics"):
                st.write("#### Accuracy Metrics")
                st.dataframe(metrics['accuracy_metrics'])
                
                # New feature importance visualization
                if metrics.get('feature_importance'):
                    st.write("#### Top Features")
                    importance_df = pd.DataFrame.from_dict(
                        metrics['feature_importance'], 
                        orient='index', 
                        columns=['importance']
                    ).sort_values('importance', ascending=False)
                    st.dataframe(importance_df.head(10))
                
                st.write("#### Latency Distribution")
                st.plotly_chart(metrics['latency_chart'])
                
                st.write("#### Recent Predictions")
                st.dataframe(metrics['recent_predictions'])
    
    # Drift monitoring section
    st.header("Drift Monitoring")
    drift_results = monitor.check_drift()
    
    if drift_results:
        cols = st.columns(3)
        cols[0].metric("Prediction Drift", 
                      "Detected" if drift_results['prediction_drift'] else "Normal",
                      delta="↑" if drift_results['prediction_drift'] else "→")
                      
        cols[1].metric("Performance Degradation",
                      "Detected" if drift_results['performance_degradation'] else "Normal",
                      delta="↑" if drift_results['performance_degradation'] else "→")
                      
        with cols[2].expander("Feature Shifts"):
            for feature, shifted in drift_results['feature_shifts'].items():
                st.write(f"{feature}: {'was shifted' if shifted else 'was stable'}")
    
    # Add alert thresholds configuration
    with st.sidebar.expander("Alert Settings"):
        st.slider("Drift Sensitivity", 1.0, 5.0, 3.0)
        st.slider("Degradation Threshold", 1.0, 3.0, 1.5)
    
    # System Resources
    st.subheader("System Resources")
    cpu, mem = monitor.get_system_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("CPU Usage", f"{cpu}%")
    with col2:
        st.metric("Memory Usage", f"{mem}%")

# Main app layout
st.title("Time Series Forecasting Agent")
with st.expander("About this app & Quick Start Guide"):
    st.markdown("""
    **A production-grade forecasting system that:**
    - Trains multiple ML models on your time series data
    - Generates ensemble forecasts with confidence intervals
    - Tracks model versions and performance over time
    - Monitors for data drift and model degradation
    
    **Navigation:**
    1. 📁 **Upload Data** (Sidebar) - Start with your time series CSV/Excel
    2. ⚙️ **Configure Models** - Select algorithms and parameters
    3. 🏋️ **Train Models** - Build forecasting pipeline
    4. 🔮 **Generate Forecasts** - Get future predictions
    5. 📊 **Evaluate** - Compare model performance
    6. 🚨 **Monitor** - Track production metrics
    """)

# Sidebar
with st.sidebar:
    st.header("Configuration", help="Global settings for the forecasting system")
    
    # Data upload
    st.subheader("Data", help="Upload your time series data in CSV or Excel format")
    uploaded_file = st.file_uploader(
        "Upload data (CSV or Excel)", 
        type=["csv", "xlsx", "xls"],
        help="Supports files with datetime index and numeric columns"
    )

if uploaded_file is not None:
    if st.session_state.data is None or st.sidebar.button("Reload Data", 
        help="Refresh and re-process the uploaded dataset"):
        st.session_state.data = load_data(uploaded_file)
        st.session_state.trained = False
        st.session_state.forecasts = None
        st.session_state.evaluation_results = None

# Model training
if st.session_state.data is not None:
    st.sidebar.subheader("Model Training", 
        help="Configure and train forecasting models")
    
    # Target column selection
    numeric_columns = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
    target_column = st.sidebar.selectbox("Target Column", numeric_columns)
    
    # Hyperparameter options
    use_hyperparameters = st.sidebar.checkbox("Use Optimized Hyperparameters")
    hyperparameters_path = None
    
    if use_hyperparameters:
        hyperparameters_path = st.sidebar.text_input(
            "Hyperparameters Path", 
            value=Config().get('paths.hyperparameters_file', 'models/hyperparameters.json')
        )
    
    # Train button
    if st.sidebar.button("Train Models", 
        help="Build all selected models with current configuration"):
        st.session_state.models = train_models(
            st.session_state.data, 
            target_column, 
            use_hyperparameters, 
            hyperparameters_path
        )

# Forecasting
if st.session_state.trained:
    st.sidebar.subheader("Forecasting")
    
    # Forecast horizon
    horizon = st.sidebar.slider("Forecast Horizon", 1, 168, 24)
    
    # Generate forecasts button
    if st.sidebar.button("Generate Forecasts"):
        st.session_state.forecasts = generate_forecasts(horizon)
        
        if st.session_state.forecasts:
            st.session_state.evaluation_results = evaluate_models(
                st.session_state.forecasts, 
                st.session_state.agent.history
            )

# Save/Load models
st.sidebar.subheader("Model Management")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("Save Models") and st.session_state.trained:
        save_models()

with col2:
    if st.button("Load Models"):
        load_models()

# Monitoring
st.sidebar.subheader("Monitoring")
if st.sidebar.button("Show Monitoring Dashboard"):
    st.session_state.selected_tab = "monitoring"

# Evaluation Section
if st.session_state.evaluation_results:
    st.subheader("Model Evaluation", 
        help="Compare model performance across key metrics: RMSE, MAE, R². Lower error values indicate better performance.")
    
    st.metric("Best Model", st.session_state.evaluation_results['best_model'],
        help="Selected based on lowest RMSE on a 20% holdout validation set. Consider business requirements before final deployment.")
    
    st.dataframe(st.session_state.evaluation_results['metrics'],
        help="""Key metrics:
        - RMSE: Root Mean Squared Error (sensitive to outliers)
        - MAE: Mean Absolute Error (easier to interpret)
        - R²: Variance explained (0-1 scale)
        - Inference Time: Milliseconds per prediction""")

# Monitoring Dashboard
if st.session_state.trained:
    st.subheader("Production Monitoring",
        help="""Track model health indicators:
        - Accuracy Drift: % change from validation baseline
        - Data Drift: Feature distribution changes
        - Traffic Patterns: Prediction volume anomalies""")
    
    st.plotly_chart(st.session_state.monitor.get_performance_trends(),
        use_container_width=True,
        help="""Performance over last 30 days:
        - Green: Within expected range
        - Yellow: Warning threshold (10% degradation)
        - Red: Critical threshold (25% degradation)""")

# Model Configuration
model_options = {
    "XGBoost": {"help": "Best for: Tabular data with clear features. Pros: Fast training, handles missing values. Cons: Limited sequence awareness."},
    "LSTM": {"help": "Best for: Complex temporal patterns. Pros: Learns long-term dependencies. Cons: Needs large datasets (>10k samples)."},
    "Prophet": {"help": "Best for: Business time series with holidays. Pros: Interpretable components. Cons: Rigid model structure."}
}

for model in model_options:
    st.sidebar.checkbox(model, value=True,
        help=model_options[model]["help"],
        key=f"model_{model.lower()}")

# Main content
if st.session_state.data is not None:
    st.subheader("Data Preview")
    st.dataframe(st.session_state.data.head())
    
    # Data statistics
    st.subheader("Data Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Rows:** {st.session_state.data.shape[0]}")
        st.write(f"**Columns:** {st.session_state.data.shape[1]}")
    
    with col2:
        series_count = st.session_state.data.groupby(['series_id', 'region']).ngroups
        st.write(f"**Time Series:** {series_count}")
        
        if 'timestamp' in st.session_state.data.columns:
            time_range = f"{st.session_state.data['timestamp'].min()} to {st.session_state.data['timestamp'].max()}"
            st.write(f"**Time Range:** {time_range}")

# Show forecasts if available
if st.session_state.forecasts:
    st.subheader("Forecasts")
    
    # Series selection
    series_names = list(st.session_state.forecasts.keys())
    selected_series = st.selectbox("Select Series", series_names, format_func=lambda x: f"{x[0]} - {x[1]}")
    st.session_state.selected_series = selected_series
    
    # Plot forecast
    fig = plot_series_forecast(
        selected_series, 
        st.session_state.forecasts, 
        st.session_state.agent.history
    )
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Show evaluation results
    if st.session_state.evaluation_results:
        st.subheader("Model Evaluation")
        
        if selected_series in st.session_state.evaluation_results:
            results = st.session_state.evaluation_results[selected_series]
            
            # Create metrics dataframe
            metrics_df = pd.DataFrame.from_dict(results, orient='index')
            
            # Show metrics table
            st.dataframe(metrics_df)
            
            # Plot metrics comparison
            st.subheader("Metrics Comparison")
            
            # Create bar chart for RMSE
            rmse_data = {model: metrics['RMSE'] for model, metrics in results.items()}
            rmse_df = pd.DataFrame(list(rmse_data.items()), columns=['Model', 'RMSE'])
            
            fig = px.bar(
                rmse_df, 
                x='Model', 
                y='RMSE', 
                title='RMSE by Model',
                color='Model'
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Show monitoring dashboard if selected
if hasattr(st.session_state, 'selected_tab') and st.session_state.selected_tab == "monitoring":
    show_monitoring_dashboard()