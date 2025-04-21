#!/usr/bin/env python3
"""
Script to combine Australian electricity demand data with weather and economic data
for improved time series forecasting.
"""
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import other modules
sys.path.append(str(Path(__file__).parent.parent))
from data.clean_utils import clean_time_series_dataset, generate_data_quality_report
from data.datacombine import prepare_datasets_for_forecasting

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def map_regions_to_stations(electricity_df):
    """
    Map electricity demand regions to weather stations.
    
    Parameters
    ----------
    electricity_df : pd.DataFrame
        Electricity demand dataframe with 'region' column
        
    Returns
    -------
    dict
        Mapping of regions to weather stations
    """
    # Get unique regions
    regions = electricity_df['region'].unique()
    
    # Define mapping of regions to stations
    # This is a simplified mapping and might need adjustment based on actual data
    region_to_station = {
        'NSW': 'Sydney',
        'VIC': 'Melbourne',
        'QLD': 'Brisbane',
        'SA': 'Adelaide',
        'WA': 'Perth',
        'TAS': 'Hobart',
        'NT': 'Darwin',
        'ACT': 'Canberra'
    }
    
    # Filter mapping to only include regions in the data
    filtered_mapping = {region: region_to_station.get(region, None) for region in regions}
    
    # Log the mapping
    logger.info(f"Region to station mapping: {filtered_mapping}")
    
    return filtered_mapping

def combine_electricity_weather_data(electricity_file, weather_file, output_file=None):
    """
    Combine electricity demand data with weather data.
    
    Parameters
    ----------
    electricity_file : str
        Path to the electricity demand data file
    weather_file : str
        Path to the weather data file
    output_file : str or None
        Path to save the combined data. If None, a default path is used.
        
    Returns
    -------
    pd.DataFrame
        Combined dataframe
    """
    logger.info(f"Combining electricity data ({electricity_file}) with weather data ({weather_file})")
    
    # Load the datasets
    try:
        electricity_df = pd.read_csv(electricity_file)
        weather_df = pd.read_csv(weather_file)
        
        # Convert timestamps to datetime
        electricity_df['timestamp'] = pd.to_datetime(electricity_df['timestamp'])
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
        
        logger.info(f"Loaded electricity data: {electricity_df.shape}")
        logger.info(f"Loaded weather data: {weather_df.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None
    
    # Map regions to stations
    region_to_station = map_regions_to_stations(electricity_df)
    
    # Create a new column in electricity_df with the corresponding station
    electricity_df['station_name'] = electricity_df['region'].map(region_to_station)
    
    # Drop rows where station_name is None
    electricity_df = electricity_df.dropna(subset=['station_name'])
    logger.info(f"Electricity data after mapping: {electricity_df.shape}")
    
    # Prepare for merge
    # Round timestamps to day for proper joining with daily weather data
    electricity_df['date'] = electricity_df['timestamp'].dt.date
    weather_df['date'] = weather_df['timestamp'].dt.date
    
    # Merge the datasets
    merged_df = pd.merge(
        electricity_df,
        weather_df,
        on=['date', 'station_name'],
        how='left',
        suffixes=('', '_weather')
    )
    
    logger.info(f"Merged data shape: {merged_df.shape}")
    
    # Drop the temporary date column and weather timestamp
    merged_df = merged_df.drop(columns=['date', 'timestamp_weather'])
    
    # Save the combined data if output file is provided
    if output_file:
        merged_df.to_csv(output_file, index=False)
        logger.info(f"Saved combined electricity-weather data to {output_file}")
    
    return merged_df

def combine_with_economic_data(combined_df, economic_file, output_file=None):
    """
    Add economic indicator data to the already combined electricity-weather data.
    
    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined electricity-weather dataframe
    economic_file : str
        Path to the economic data file
    output_file : str or None
        Path to save the final combined data. If None, a default path is used.
        
    Returns
    -------
    pd.DataFrame
        Final combined dataframe
    """
    logger.info(f"Adding economic data ({economic_file}) to combined data")
    
    # Load the economic data
    try:
        economic_df = pd.read_csv(economic_file)
        economic_df['timestamp'] = pd.to_datetime(economic_df['timestamp'])
        logger.info(f"Loaded economic data: {economic_df.shape}")
    except Exception as e:
        logger.error(f"Error loading economic data: {str(e)}")
        return combined_df
    
    # Extract year and month for joining
    combined_df['year_month'] = combined_df['timestamp'].dt.to_period('M')
    economic_df['year_month'] = economic_df['timestamp'].dt.to_period('M')
    
    # Pivot economic data to have indicators as columns
    economic_pivot = economic_df.pivot_table(
        index='year_month',
        columns='indicator',
        values='value',
        aggfunc='mean'
    ).reset_index()
    
    # Convert year_month back to string for merging
    economic_pivot['year_month'] = economic_pivot['year_month'].astype(str)
    combined_df['year_month'] = combined_df['year_month'].astype(str)
    
    # Merge with the combined data
    final_df = pd.merge(
        combined_df,
        economic_pivot,
        on='year_month',
        how='left'
    )
    
    logger.info(f"Final combined data shape: {final_df.shape}")
    
    # Drop the temporary year_month column
    final_df = final_df.drop(columns=['year_month'])
    
    # Forward fill missing economic values (since they're monthly)
    for col in economic_pivot.columns:
        if col != 'year_month' and col in final_df.columns:
            final_df[col] = final_df[col].ffill()
    
    # Save the final combined data if output file is provided
    if output_file:
        final_df.to_csv(output_file, index=False)
        logger.info(f"Saved final combined data to {output_file}")
    
    return final_df

def create_lagged_features(df, value_col='value', lag_periods=[1, 7, 30], id_cols=None):
    """
    Create lagged features for time series forecasting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    value_col : str
        Column to create lags for
    lag_periods : list
        List of lag periods to create
    id_cols : list or None
        List of columns that identify unique time series
        
    Returns
    -------
    pd.DataFrame
        Dataframe with lagged features
    """
    logger.info(f"Creating lagged features for {value_col} with periods {lag_periods}")
    
    # Make a copy of the dataframe
    df_copy = df.copy()
    
    # Sort by timestamp
    if id_cols:
        df_copy = df_copy.sort_values(id_cols + ['timestamp'])
    else:
        df_copy = df_copy.sort_values('timestamp')
    
    # Create lagged features
    for lag in lag_periods:
        if id_cols:
            # Create lags within each group
            df_copy[f'{value_col}_lag_{lag}'] = df_copy.groupby(id_cols)[value_col].shift(lag)
        else:
            # Create lags for the entire dataset
            df_copy[f'{value_col}_lag_{lag}'] = df_copy[value_col].shift(lag)
    
    # Drop rows with NaN values from lagging
    df_copy = df_copy.dropna(subset=[f'{value_col}_lag_{lag}' for lag in lag_periods])
    logger.info(f"Data shape after creating lags: {df_copy.shape}")
    
    return df_copy

def create_rolling_features(df, value_col='value', windows=[7, 30], id_cols=None):
    """
    Create rolling window features for time series forecasting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    value_col : str
        Column to create rolling features for
    windows : list
        List of window sizes to create
    id_cols : list or None
        List of columns that identify unique time series
        
    Returns
    -------
    pd.DataFrame
        Dataframe with rolling features
    """
    logger.info(f"Creating rolling features for {value_col} with windows {windows}")
    
    # Make a copy of the dataframe
    df_copy = df.copy()
    
    # Sort by timestamp
    if id_cols:
        df_copy = df_copy.sort_values(id_cols + ['timestamp'])
    else:
        df_copy = df_copy.sort_values('timestamp')
    
    # Create rolling features
    for window in windows:
        if id_cols:
            # Create rolling features within each group
            df_copy[f'{value_col}_roll_mean_{window}'] = df_copy.groupby(id_cols)[value_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df_copy[f'{value_col}_roll_std_{window}'] = df_copy.groupby(id_cols)[value_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        else:
            # Create rolling features for the entire dataset
            df_copy[f'{value_col}_roll_mean_{window}'] = df_copy[value_col].rolling(window=window, min_periods=1).mean()
            df_copy[f'{value_col}_roll_std_{window}'] = df_copy[value_col].rolling(window=window, min_periods=1).std()
    
    logger.info(f"Data shape after creating rolling features: {df_copy.shape}")
    
    return df_copy

def prepare_final_dataset(electricity_file, weather_file, economic_file, output_file=None, create_lags=True, create_rolling=True):
    """
    Prepare the final dataset for forecasting by combining all data sources and creating features.
    
    Parameters
    ----------
    electricity_file : str
        Path to the electricity demand data file
    weather_file : str
        Path to the weather data file
    economic_file : str
        Path to the economic data file
    output_file : str or None
        Path to save the final dataset. If None, a default path is used.
    create_lags : bool
        Whether to create lagged features
    create_rolling : bool
        Whether to create rolling window features
        
    Returns
    -------
    pd.DataFrame
        Final prepared dataset
    """
    # Set default output file if not provided
    if output_file is None:
        output_dir = os.path.dirname(electricity_file)
        output_file = os.path.join(output_dir, "final_combined_dataset.csv")
    
    # Step 1: Combine electricity and weather data
    combined_df = combine_electricity_weather_data(
        electricity_file,
        weather_file,
        output_file=None  # Don't save intermediate results
    )
    
    if combined_df is None:
        logger.error("Failed to combine electricity and weather data")
        return None
    
    # Step 2: Add economic data
    final_df = combine_with_economic_data(
        combined_df,
        economic_file,
        output_file=None  # Don't save intermediate results
    )
    
    # Step 3: Create lagged features if requested
    if create_lags:
        final_df = create_lagged_features(
            final_df,
            value_col='value',
            lag_periods=[1, 7, 24, 168],  # 1 hour, 7 hours, 1 day, 1 week
            id_cols=['series_id', 'region']
        )
    
    # Step 4: Create rolling window features if requested
    if create_rolling:
        final_df = create_rolling_features(
            final_df,
            value_col='value',
            windows=[24, 168, 720],  # 1 day, 1 week, 30 days
            id_cols=['series_id', 'region']
        )
    
    # Step 5: Save the final dataset
    # Save with better formatting to avoid excessively long lines
    final_df.to_csv(output_file, index=False, lineterminator='\n')
    logger.info(f"Saved final prepared dataset to {output_file}")
    
    return final_df

def visualize_combined_data(final_df, output_dir=None):
    """
    Create visualizations of the combined dataset.
    
    Parameters
    ----------
    final_df : pd.DataFrame
        Final combined dataframe
    output_dir : str or None
        Directory to save visualizations. If None, visualizations are not saved.
    """
    logger.info("Creating visualizations of combined data")
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set the style
    sns.set(style="whitegrid")
    
    # 1. Electricity demand by region over time
    plt.figure(figsize=(15, 8))
    for region in final_df['region'].unique():
        subset = final_df[final_df['region'] == region].sort_values('timestamp')
        plt.plot(subset['timestamp'], subset['value'], label=region, alpha=0.7)
    
    plt.title('Electricity Demand by Region')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "electricity_demand_by_region.png"))
        plt.close()
    else:
        plt.show()
    
    # 2. Correlation between electricity demand and weather
    weather_cols = [col for col in final_df.columns if 'temperature' in col.lower() or 'rainfall' in col.lower()]
    if weather_cols:
        plt.figure(figsize=(12, 8))
        correlation_data = final_df[['value'] + weather_cols].corr()
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation between Electricity Demand and Weather')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "electricity_weather_correlation.png"))
            plt.close()
        else:
            plt.show()
    
    # 3. Correlation between electricity demand and economic indicators
    economic_cols = [col for col in final_df.columns if col in ['GDP', 'CPI', 'UNEMPLOYMENT', 'INDUSTRIAL_PRODUCTION', 'RETAIL_SALES', 'INTEREST_RATE']]
    if economic_cols:
        plt.figure(figsize=(12, 8))
        correlation_data = final_df[['value'] + economic_cols].corr()
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation between Electricity Demand and Economic Indicators')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "electricity_economic_correlation.png"))
            plt.close()
        else:
            plt.show()
    
    # 4. Seasonal patterns
    if 'hour' in final_df.columns and 'day_of_week' in final_df.columns:
        # Hourly patterns
        plt.figure(figsize=(12, 6))
        hourly_demand = final_df.groupby('hour')['value'].mean()
        sns.lineplot(x=hourly_demand.index, y=hourly_demand.values)
        plt.title('Average Electricity Demand by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Demand')
        plt.xticks(range(0, 24, 2))
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "electricity_hourly_pattern.png"))
            plt.close()
        else:
            plt.show()
        
        # Daily patterns
        plt.figure(figsize=(12, 6))
        daily_demand = final_df.groupby('day_of_week')['value'].mean()
        sns.barplot(x=daily_demand.index, y=daily_demand.values)
        plt.title('Average Electricity Demand by Day of Week')
        plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
        plt.ylabel('Average Demand')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "electricity_daily_pattern.png"))
            plt.close()
        else:
            plt.show()
    
    # 5. Relationship between demand and temperature
    if 'Maximum temperature (°C)' in final_df.columns:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='Maximum temperature (°C)', y='value', hue='region', data=final_df, alpha=0.5)
        plt.title('Electricity Demand vs. Maximum Temperature')
        plt.xlabel('Maximum Temperature (°C)')
        plt.ylabel('Electricity Demand')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "electricity_temperature_relationship.png"))
            plt.close()
        else:
            plt.show()
    
    logger.info("Visualizations created successfully")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Combine multiple datasets for forecasting')
    parser.add_argument('--electricity', type=str, help='Path to the electricity demand data file', 
                       default='data/australian_electricity_demand_dataset_parsed_clean.csv')
    parser.add_argument('--weather', type=str, help='Path to the weather data file',
                       default='data/simulated_weather_data_clean.csv')
    parser.add_argument('--economic', type=str, help='Path to the economic data file',
                       default='data/simulated_economic_data_clean.csv')
    parser.add_argument('--output', type=str, help='Path to save the final dataset', default=None)
    parser.add_argument('--no-lags', action='store_true', help='Do not create lagged features')
    parser.add_argument('--no-rolling', action='store_true', help='Do not create rolling window features')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--viz-dir', type=str, help='Directory to save visualizations', default='data/visualizations')
    
    args = parser.parse_args()
    
    # Prepare the final dataset
    final_df = prepare_final_dataset(
        electricity_file=args.electricity,
        weather_file=args.weather,
        economic_file=args.economic,
        output_file=args.output,
        create_lags=not args.no_lags,
        create_rolling=not args.no_rolling
    )
    
    if final_df is not None and args.visualize:
        visualize_combined_data(final_df, output_dir=args.viz_dir)
        print(f"Visualizations saved to {args.viz_dir}")
    
    if final_df is not None:
        print(f"Final dataset shape: {final_df.shape}")
        print(f"Final dataset columns: {final_df.columns.tolist()}")
        print(f"Final dataset head:\n{final_df.head()}")
