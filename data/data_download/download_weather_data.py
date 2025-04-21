#!/usr/bin/env python3
"""
Script to download and prepare Australian weather data for time series forecasting.
This data can be combined with electricity demand data to improve forecasting accuracy.
"""
import pandas as pd
import numpy as np
import requests
import io
import os
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys

# Add parent directory to path to import other modules
sys.path.append(str(Path(__file__).parent.parent))
from data.clean_utils import clean_time_series_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define Australian cities/stations with their station numbers
# These are major cities that likely correspond to electricity demand regions
STATIONS = {
    'Sydney': '066062',     # Observatory Hill
    'Melbourne': '086071',  # Melbourne Regional Office
    'Brisbane': '040913',   # Brisbane
    'Adelaide': '023034',   # Adelaide (Kent Town)
    'Perth': '009021',      # Perth Airport
    'Hobart': '094029',     # Hobart (Ellerslie Road)
    'Darwin': '014015',     # Darwin Airport
    'Canberra': '070351'    # Canberra Airport
}

def download_weather_data(station_id, start_year, end_year, output_dir='../data'):
    """
    Download weather data for a specific station from the Australian Bureau of Meteorology.
    
    Parameters
    ----------
    station_id : str
        BOM station ID
    start_year : int
        Start year for data download
    end_year : int
        End year for data download
    output_dir : str
        Directory to save the downloaded data
        
    Returns
    -------
    str
        Path to the downloaded data file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path
    output_file = os.path.join(output_dir, f"weather_station_{station_id}_{start_year}_{end_year}.csv")
    
    # If file already exists, return its path
    if os.path.exists(output_file):
        logger.info(f"Weather data file already exists: {output_file}")
        return output_file
    
    # Prepare to collect data for all years
    all_data = []
    
    # Download data for each year
    for year in range(start_year, end_year + 1):
        try:
            # Construct URL for BOM data
            # Note: This is a simplified example. The actual BOM data access might require
            # different URLs or authentication. This is for demonstration purposes.
            url = f"http://www.bom.gov.au/climate/dwo/{year}/data/IDCJDW0000.{station_id}.{year}.csv"
            
            logger.info(f"Downloading weather data for station {station_id}, year {year}")
            
            # Download the data
            response = requests.get(url)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the CSV data
                df = pd.read_csv(io.StringIO(response.text), skiprows=1)
                
                # Add year to the dataframe
                df['Year'] = year
                
                # Add to the list of dataframes
                all_data.append(df)
                
                logger.info(f"Successfully downloaded data for year {year}")
            else:
                logger.warning(f"Failed to download data for year {year}: Status code {response.status_code}")
        
        except Exception as e:
            logger.error(f"Error downloading data for year {year}: {str(e)}")
    
    # If no data was downloaded, return None
    if not all_data:
        logger.error(f"No data downloaded for station {station_id}")
        return None
    
    # Combine all years of data
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # Save the combined data
    df_combined.to_csv(output_file, index=False)
    logger.info(f"Saved combined weather data to {output_file}")
    
    return output_file

def prepare_weather_data(stations=None, start_year=2000, end_year=2023, output_dir='../data'):
    """
    Download and prepare weather data for multiple stations.
    
    Parameters
    ----------
    stations : dict or None
        Dictionary of station names and IDs. If None, uses all predefined stations.
    start_year : int
        Start year for data download
    end_year : int
        End year for data download
    output_dir : str
        Directory to save the downloaded and prepared data
        
    Returns
    -------
    str
        Path to the prepared data file
    """
    # Use all predefined stations if none are specified
    if stations is None:
        stations = STATIONS
    
    # Download data for each station
    station_data_files = {}
    for station_name, station_id in stations.items():
        file_path = download_weather_data(station_id, start_year, end_year, output_dir)
        if file_path:
            station_data_files[station_name] = file_path
    
    # If no data was downloaded, return None
    if not station_data_files:
        logger.error("No weather data downloaded")
        return None
    
    # Combine data from all stations
    all_station_data = []
    
    for station_name, file_path in station_data_files.items():
        try:
            # Load the data
            df = pd.read_csv(file_path)
            
            # Add station name
            df['station_name'] = station_name
            
            # Add to the list of dataframes
            all_station_data.append(df)
            
            logger.info(f"Loaded data for station {station_name}")
        
        except Exception as e:
            logger.error(f"Error loading data for station {station_name}: {str(e)}")
    
    # If no data was loaded, return None
    if not all_station_data:
        logger.error("No weather data loaded")
        return None
    
    # Combine all station data
    df_combined = pd.concat(all_station_data, ignore_index=True)
    
    # Clean and prepare the data
    # This will depend on the specific format of the BOM data
    # Here's a simplified example:
    
    # Create a timestamp column
    if 'Year' in df_combined.columns and 'Month' in df_combined.columns and 'Day' in df_combined.columns:
        df_combined['timestamp'] = pd.to_datetime(
            df_combined['Year'].astype(str) + '-' + 
            df_combined['Month'].astype(str) + '-' + 
            df_combined['Day'].astype(str)
        )
    
    # Select relevant columns
    # This will depend on the specific format of the BOM data
    # Here's a simplified example:
    relevant_columns = [
        'timestamp', 'station_name', 
        'Maximum temperature (°C)', 'Minimum temperature (°C)', 
        'Rainfall (mm)', 'Solar exposure (MJ/m*m)'
    ]
    
    # Filter to only include columns that exist in the dataframe
    available_columns = [col for col in relevant_columns if col in df_combined.columns]
    
    # If some columns are missing, log a warning
    if len(available_columns) < len(relevant_columns):
        missing_columns = set(relevant_columns) - set(available_columns)
        logger.warning(f"Missing columns in weather data: {missing_columns}")
    
    # Select available columns
    if available_columns:
        df_filtered = df_combined[available_columns].copy()
    else:
        # If no relevant columns are available, use all columns
        df_filtered = df_combined.copy()
        logger.warning("No relevant columns found in weather data, using all columns")
    
    # Save the combined and filtered data
    combined_file = os.path.join(output_dir, f"weather_data_combined_{start_year}_{end_year}.csv")
    df_filtered.to_csv(combined_file, index=False)
    logger.info(f"Saved combined weather data to {combined_file}")
    
    # Clean the data using the clean_utils module
    try:
        clean_time_series_dataset(
            file_path=combined_file,
            dataset_name=f"weather_data_{start_year}_{end_year}",
            timestamp_col='timestamp',
            value_col='Maximum temperature (°C)',  # Change this to the column you want to use
            id_cols=['station_name'],
            resample_freq='D',  # Daily data
            handle_outliers='cap',
            output_dir=output_dir
        )
        logger.info(f"Cleaned weather data saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error cleaning weather data: {str(e)}")
    
    return combined_file

def simulate_weather_data(start_date='2000-01-01', end_date='2023-12-31', stations=None, output_dir='../data'):
    """
    Simulate weather data for demonstration purposes.
    This is useful when actual API access is not available.
    
    Parameters
    ----------
    start_date : str
        Start date for simulated data
    end_date : str
        End date for simulated data
    stations : dict or None
        Dictionary of station names and IDs. If None, uses all predefined stations.
    output_dir : str
        Directory to save the simulated data
        
    Returns
    -------
    str
        Path to the simulated data file
    """
    # Use all predefined stations if none are specified
    if stations is None:
        stations = STATIONS
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path
    output_file = os.path.join(output_dir, "simulated_weather_data.csv")
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create dataframe with all dates and stations
    data = []
    
    for station_name in stations.keys():
        for date in date_range:
            # Simulate temperature data with seasonal patterns
            day_of_year = date.dayofyear
            season_factor = np.sin(2 * np.pi * day_of_year / 365)
            
            # Base temperature varies by station (latitude)
            if station_name in ['Darwin']:  # Tropical
                base_temp = 30
                amplitude = 5
            elif station_name in ['Brisbane', 'Perth']:  # Subtropical
                base_temp = 25
                amplitude = 8
            elif station_name in ['Sydney', 'Adelaide']:  # Temperate
                base_temp = 20
                amplitude = 10
            else:  # Cooler regions
                base_temp = 15
                amplitude = 12
            
            # Calculate max and min temperatures with seasonal variation
            max_temp = base_temp + amplitude * season_factor + np.random.normal(0, 2)
            min_temp = max_temp - 8 - np.random.normal(0, 2)
            
            # Simulate rainfall (more in summer, less in winter)
            rainfall_prob = 0.3 + 0.2 * season_factor
            rainfall = np.random.exponential(5) if np.random.random() < rainfall_prob else 0
            
            # Simulate solar exposure (more in summer, less in winter)
            solar_exposure = 15 + 10 * season_factor + np.random.normal(0, 2)
            solar_exposure = max(0, solar_exposure)  # Ensure non-negative
            
            # Add to data list
            data.append({
                'timestamp': date,
                'station_name': station_name,
                'Maximum temperature (°C)': max_temp,
                'Minimum temperature (°C)': min_temp,
                'Rainfall (mm)': rainfall,
                'Solar exposure (MJ/m*m)': solar_exposure
            })
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    # Save the simulated data
    df.to_csv(output_file, index=False)
    logger.info(f"Saved simulated weather data to {output_file}")
    
    # Clean the data using the clean_utils module
    try:
        clean_time_series_dataset(
            file_path=output_file,
            dataset_name="simulated_weather_data",
            timestamp_col='timestamp',
            value_col='Maximum temperature (°C)',
            id_cols=['station_name'],
            resample_freq='D',  # Daily data
            handle_outliers='cap',
            output_dir=output_dir
        )
        logger.info(f"Cleaned simulated weather data saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error cleaning simulated weather data: {str(e)}")
    
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and prepare Australian weather data')
    parser.add_argument('--start_year', type=int, help='Start year for data download', default=2000)
    parser.add_argument('--end_year', type=int, help='End year for data download', default=2023)
    parser.add_argument('--output_dir', type=str, help='Directory to save the data', default='../data')
    parser.add_argument('--stations', type=str, help='Comma-separated list of station names to download', default=None)
    parser.add_argument('--simulate', action='store_true', help='Simulate weather data instead of downloading')
    
    args = parser.parse_args()
    
    # Parse stations if provided
    if args.stations:
        station_names = args.stations.split(',')
        selected_stations = {name: STATIONS[name] for name in station_names if name in STATIONS}
        
        if not selected_stations:
            logger.error(f"No valid stations found in: {args.stations}")
            logger.info(f"Available stations: {list(STATIONS.keys())}")
            sys.exit(1)
    else:
        selected_stations = STATIONS
    
    # Download or simulate weather data
    if args.simulate:
        output_file = simulate_weather_data(
            start_date=f"{args.start_year}-01-01",
            end_date=f"{args.end_year}-12-31",
            stations=selected_stations,
            output_dir=args.output_dir
        )
    else:
        output_file = prepare_weather_data(
            stations=selected_stations,
            start_year=args.start_year,
            end_year=args.end_year,
            output_dir=args.output_dir
        )
    
    if output_file:
        print(f"Weather data prepared and saved to: {output_file}")
    else:
        print("Failed to prepare weather data")
