#!/usr/bin/env python3
"""
Script to run hyperparameter tuning and train models with optimized parameters.
This script demonstrates the complete workflow:
1. Run hyperparameter tuning
2. Train models with optimized parameters
3. Evaluate model performance
"""
import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse
from datetime import datetime

# Add parent directory to path to import agent modules
import sys
sys.path.append(str(Path(__file__).parent))
from agent.agent_core import ForecastingAgent
from models.hyperparameter_tuning import HyperparameterTuner
from agent.utils import calculate_metrics, plot_forecast, create_evaluation_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_dataset(file_path):
    """
    Load and prepare the dataset for forecasting.
    
    Parameters
    ----------
    file_path : str
        Path to the dataset file
        
    Returns
    -------
    pd.DataFrame
        Prepared dataframe
    """
    logger.info(f"Loading dataset from {file_path}")
    
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Dataset loaded successfully with shape {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def run_hyperparameter_tuning(dataset_path, target_column='value', n_trials=50, output_dir='models'):
    """
    Run hyperparameter tuning for all models.
    
    Parameters
    ----------
    dataset_path : str
        Path to the dataset file
    target_column : str
        Name of the target column
    n_trials : int
        Number of trials for optimization
    output_dir : str
        Directory to save the best parameters
        
    Returns
    -------
    dict
        Best hyperparameters
    """
    # Load the dataset
    df = load_dataset(dataset_path)
    if df is None:
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create hyperparameter tuner
    logger.info(f"Creating hyperparameter tuner with {n_trials} trials")
    tuner = HyperparameterTuner(df, target_column, n_trials=n_trials)
    
    # Run tuning
    logger.info("Starting hyperparameter tuning")
    best_params = tuner.tune_all()
    
    # Save best parameters
    params_path = os.path.join(output_dir, 'best_params.json')
    tuner.save_best_params(params_path)
    
    logger.info(f"Best parameters saved to {params_path}")
    
    return best_params

def train_with_optimized_params(dataset_path, target_column='value', params_path='models/best_params.json', output_dir='models'):
    """
    Train forecasting models with optimized hyperparameters.
    
    Parameters
    ----------
    dataset_path : str
        Path to the dataset file
    target_column : str
        Name of the target column
    params_path : str
        Path to the best parameters file
    output_dir : str
        Directory to save the trained models
        
    Returns
    -------
    ForecastingAgent
        Trained forecasting agent
    """
    # Load the dataset
    df = load_dataset(dataset_path)
    if df is None:
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and train the forecasting agent
    logger.info("Creating and training forecasting agent with optimized parameters")
    agent = ForecastingAgent()
    
    try:
        # Train the model with optimized parameters
        agent.train_model(df, target_column, use_hyperparameters=True, hyperparameters_path=params_path)
        
        # Save the trained models
        agent.save_models(path=output_dir)
        
        logger.info(f"Models trained with optimized parameters and saved to {output_dir}")
        
        return agent
    
    except Exception as e:
        logger.error(f"Error training with optimized parameters: {str(e)}")
        return None

def evaluate_optimized_models(agent, dataset_path, target_column='value', output_dir='evaluation_results'):
    """
    Evaluate models trained with optimized hyperparameters.
    
    Parameters
    ----------
    agent : ForecastingAgent
        Trained forecasting agent
    dataset_path : str
        Path to the dataset file
    target_column : str
        Name of the target column
    output_dir : str
        Directory to save the evaluation results
        
    Returns
    -------
    dict
        Evaluation metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate forecasts for different horizons
    horizons = [24, 48, 168]  # 1 day, 2 days, 1 week
    
    all_results = {}
    
    for horizon in horizons:
        logger.info(f"Generating forecasts for horizon {horizon}")
        
        # Generate forecasts
        forecasts = agent.forecast(horizon=horizon)
        
        # Calculate metrics for each series and model
        horizon_results = {}
        
        for name, models_pred in forecasts.items():
            # Get test data for this series
            test_data = agent.history[name]['test_data']
            
            # Get actual values
            y_true = test_data[target_column].values[:horizon]
            
            # Calculate metrics for each model
            model_metrics = {}
            for model_name, predictions in models_pred.items():
                # Trim predictions to match actual length
                pred = predictions[:len(y_true)]
                metrics = calculate_metrics(y_true, pred)
                model_metrics[model_name] = metrics
            
            # Store metrics for this series
            horizon_results[name] = model_metrics
            
            # Plot forecasts
            plt.figure(figsize=(12, 6))
            plt.plot(y_true, label='Actual', linewidth=2)
            
            for model_name, pred in models_pred.items():
                plt.plot(pred[:len(y_true)], label=f'{model_name}', linewidth=1.5, alpha=0.8)
            
            plt.title(f'Forecast for {name[0]} - {name[1]} (Horizon: {horizon})')
            plt.xlabel('Time Steps')
            plt.ylabel(target_column)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plot_path = os.path.join(output_dir, f"forecast_{name[0]}_{name[1]}_h{horizon}.png")
            plt.savefig(plot_path)
            plt.close()
        
        # Store results for this horizon
        all_results[f"horizon_{horizon}"] = horizon_results
    
    # Save evaluation results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    logger.info(f"Evaluation results saved to {results_path}")
    
    return all_results

def main():
    """Main function to run the complete workflow"""
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning and train models')
    parser.add_argument('--dataset', type=str, help='Path to the dataset file',
                       default='data/final_combined_dataset_optimized.csv')
    parser.add_argument('--target', type=str, help='Name of the target column',
                       default='value')
    parser.add_argument('--trials', type=int, help='Number of trials for optimization',
                       default=50)
    parser.add_argument('--models-dir', type=str, help='Directory to save the trained models',
                       default='models')
    parser.add_argument('--eval-dir', type=str, help='Directory to save the evaluation results',
                       default='evaluation_results')
    parser.add_argument('--skip-tuning', action='store_true', help='Skip hyperparameter tuning')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    logger.info(f"Starting workflow at {start_time}")
    
    # Step 1: Run hyperparameter tuning
    if not args.skip_tuning:
        logger.info("Step 1: Running hyperparameter tuning")
        best_params = run_hyperparameter_tuning(
            args.dataset, 
            args.target, 
            args.trials, 
            args.models_dir
        )
        
        if best_params is None:
            logger.error("Hyperparameter tuning failed")
            return
    else:
        logger.info("Skipping hyperparameter tuning")
    
    # Step 2: Train models with optimized parameters
    logger.info("Step 2: Training models with optimized parameters")
    params_path = os.path.join(args.models_dir, 'best_params.json')
    
    agent = train_with_optimized_params(
        args.dataset, 
        args.target, 
        params_path, 
        args.models_dir
    )
    
    if agent is None:
        logger.error("Training with optimized parameters failed")
        return
    
    # Step 3: Evaluate models
    logger.info("Step 3: Evaluating models")
    evaluation_results = evaluate_optimized_models(
        agent, 
        args.dataset, 
        args.target, 
        args.eval_dir
    )
    
    end_time = datetime.now()
    total_time = end_time - start_time
    logger.info(f"Workflow completed in {total_time}")
    
    logger.info("Complete workflow finished successfully")

if __name__ == "__main__":
    main()
