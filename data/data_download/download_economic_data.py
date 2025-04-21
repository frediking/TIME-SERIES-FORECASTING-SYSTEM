#!/usr/bin/env python3
"""
Script to download and prepare Australian economic indicator data for time series forecasting.
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

# Define economic indicators of interest
INDICATORS = {
    'GDP': 'Gross Domestic Product',
    'CPI': 'Consumer Price Index',
    'UNEMPLOYMENT': 'Unemployment Rate',
    'INDUSTRIAL_PRODUCTION': 'Industrial Production Index',
    'RETAIL_SALES': 'Retail Sales',
    'INTEREST_RATE': 'Interest Rate'
}

def download_economic_data(indicator, start_date, end_date, output_dir='../data'):
    """
    Download economic indicator data from the Australian Bureau of Statistics.
    
    Parameters
    ----------
    indicator : str
        Economic indicator to download
    start_date : str
        Start date for data download (YYYY-MM-DD)
    end_date : str
        End date for data download (YYYY-MM-DD)
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
    output_file = os.path.join(output_dir, f"economic_{indicator.lower()}_{start_date}_{end_date}.csv")
    
    # If file already exists, return its path
    if os.path.exists(output_file):
        logger.info(f"Economic data file already exists: {output_file}")
        return output_file
    
    try:
        # Construct URL for ABS data
        # Note: This is a simplified example. The actual ABS data access might require
        # different URLs or authentication. This is for demonstration purposes.
        url = f"https://api.data.abs.gov.au/data/{indicator.lower()}/all?startPeriod={start_date}&endPeriod={end_date}"
        
        logger.info(f"Downloading economic data for indicator {indicator}")
        
        # Download the data
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON data
            data = response.json()
            
            # Convert to dataframe (this will depend on the actual API response format)
            # This is a placeholder - actual implementation will depend on the API
            df = pd.DataFrame(data['observations'])
            
            # Save the data
            df.to_csv(output_file, index=False)
            logger.info(f"Saved economic data to {output_file}")
            
            return output_file
        else:
            logger.warning(f"Failed to download data for indicator {indicator}: Status code {response.status_code}")
            return None
    
    except Exception as e:
        logger.error(f"Error downloading data for indicator {indicator}: {str(e)}")
        return None

def simulate_economic_data(start_date='2000-01-01', end_date='2023-12-31', indicators=None, output_dir='../data'):
    """
    Simulate economic indicator data for demonstration purposes.
    This is useful when actual API access is not available.
    
    Parameters
    ----------
    start_date : str
        Start date for simulated data
    end_date : str
        End date for simulated data
    indicators : list or None
        List of indicators to simulate. If None, uses all predefined indicators.
    output_dir : str
        Directory to save the simulated data
        
    Returns
    -------
    str
        Path to the simulated data file
    """
    # Use all predefined indicators if none are specified
    if indicators is None:
        indicators = list(INDICATORS.keys())
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path
    output_file = os.path.join(output_dir, "simulated_economic_data.csv")
    
    # Create date range (monthly for economic data)
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    
    # Create dataframe with all dates and indicators
    data = []
    
    # Base values for each indicator
    base_values = {
        'GDP': 100,
        'CPI': 100,
        'UNEMPLOYMENT': 5,
        'INDUSTRIAL_PRODUCTION': 100,
        'RETAIL_SALES': 100,
        'INTEREST_RATE': 3
    }
    
    # Trend factors (annual growth/change)
    trend_factors = {
        'GDP': 0.025,  # 2.5% annual growth
        'CPI': 0.02,   # 2% annual inflation
        'UNEMPLOYMENT': -0.001,  # Slight decrease over time
        'INDUSTRIAL_PRODUCTION': 0.015,  # 1.5% annual growth
        'RETAIL_SALES': 0.02,  # 2% annual growth
        'INTEREST_RATE': 0.001  # Slight increase over time
    }
    
    # Seasonal factors (quarterly patterns)
    def seasonal_factor(indicator, month):
        if indicator == 'RETAIL_SALES':
            # Higher in December (Christmas), lower in February
            return 0.15 * np.sin(2 * np.pi * (month - 2) / 12)
        elif indicator == 'GDP':
            # Quarterly pattern
            return 0.01 * np.sin(2 * np.pi * month / 3)
        elif indicator == 'UNEMPLOYMENT':
            # Higher in winter, lower in summer
            return 0.005 * np.sin(2 * np.pi * (month - 6) / 12)
        else:
            # Minimal seasonality for other indicators
            return 0.005 * np.sin(2 * np.pi * month / 12)
    
    # Simulate data for each indicator
    for indicator in indicators:
        # Skip if indicator not in base values
        if indicator not in base_values:
            continue
            
        # Get base value and trend factor
        base = base_values[indicator]
        trend = trend_factors.get(indicator, 0)
        
        # Current value starts at base
        current_value = base
        
        for date in date_range:
            # Calculate time factor (years since start)
            years_since_start = (date - pd.Timestamp(start_date)).days / 365.25
            
            # Apply trend
            trend_value = base * (1 + trend * years_since_start)
            
            # Apply seasonality
            season_value = trend_value * (1 + seasonal_factor(indicator, date.month))
            
            # Add random noise
            noise_factor = 0.01  # 1% random variation
            noise = np.random.normal(0, noise_factor * season_value)
            
            # Calculate final value
            value = season_value + noise
            
            # Ensure non-negative values
            value = max(0, value)
            
            # Add to data list
            data.append({
                'timestamp': date,
                'indicator': indicator,
                'value': value,
                'indicator_name': INDICATORS.get(indicator, indicator)
            })
            
            # Update current value for next iteration
            current_value = value
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    # Save the simulated data
    df.to_csv(output_file, index=False)
    logger.info(f"Saved simulated economic data to {output_file}")
    
    # Clean the data using the clean_utils module
    try:
        clean_time_series_dataset(
            file_path=output_file,
            dataset_name="simulated_economic_data",
            timestamp_col='timestamp',
            value_col='value',
            id_cols=['indicator'],
            resample_freq='M',  # Monthly data
            handle_outliers='cap',
            output_dir=output_dir
        )
        logger.info(f"Cleaned simulated economic data saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error cleaning simulated economic data: {str(e)}")
    
    return output_file

def prepare_economic_data(start_date='2000-01-01', end_date='2023-12-31', indicators=None, output_dir='../data', simulate=True):
    """
    Download or simulate economic indicator data.
    
    Parameters
    ----------
    start_date : str
        Start date for data
    end_date : str
        End date for data
    indicators : list or None
        List of indicators to include. If None, uses all predefined indicators.
    output_dir : str
        Directory to save the data
    simulate : bool
        Whether to simulate data instead of downloading
        
    Returns
    -------
    str
        Path to the prepared data file
    """
    # Use all predefined indicators if none are specified
    if indicators is None:
        indicators = list(INDICATORS.keys())
    
    if simulate:
        return simulate_economic_data(start_date, end_date, indicators, output_dir)
    else:
        # Download each indicator
        indicator_files = {}
        for indicator in indicators:
            file_path = download_economic_data(indicator, start_date, end_date, output_dir)
            if file_path:
                indicator_files[indicator] = file_path
        
        # If no data was downloaded, return None
        if not indicator_files:
            logger.error("No economic data downloaded")
            return None
        
        # Combine data from all indicators
        all_indicator_data = []
        
        for indicator, file_path in indicator_files.items():
            try:
                # Load the data
                df = pd.read_csv(file_path)
                
                # Add indicator name
                df['indicator'] = indicator
                df['indicator_name'] = INDICATORS.get(indicator, indicator)
                
                # Add to the list of dataframes
                all_indicator_data.append(df)
                
                logger.info(f"Loaded data for indicator {indicator}")
            
            except Exception as e:
                logger.error(f"Error loading data for indicator {indicator}: {str(e)}")
        
        # If no data was loaded, return None
        if not all_indicator_data:
            logger.error("No economic data loaded")
            return None
        
        # Combine all indicator data
        df_combined = pd.concat(all_indicator_data, ignore_index=True)
        
        # Save the combined data
        combined_file = os.path.join(output_dir, f"economic_data_combined_{start_date}_{end_date}.csv")
        df_combined.to_csv(combined_file, index=False)
        logger.info(f"Saved combined economic data to {combined_file}")
        
        # Clean the data using the clean_utils module
        try:
            clean_time_series_dataset(
                file_path=combined_file,
                dataset_name=f"economic_data_{start_date}_{end_date}",
                timestamp_col='timestamp',
                value_col='value',
                id_cols=['indicator'],
                resample_freq='M',  # Monthly data
                handle_outliers='cap',
                output_dir=output_dir
            )
            logger.info(f"Cleaned economic data saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error cleaning economic data: {str(e)}")
        
        return combined_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and prepare Australian economic indicator data')
    parser.add_argument('--start_date', type=str, help='Start date for data (YYYY-MM-DD)', default='2000-01-01')
    parser.add_argument('--end_date', type=str, help='End date for data (YYYY-MM-DD)', default='2023-12-31')
    parser.add_argument('--output_dir', type=str, help='Directory to save the data', default='../data')
    parser.add_argument('--indicators', type=str, help='Comma-separated list of indicators to include', default=None)
    parser.add_argument('--simulate', action='store_true', help='Simulate economic data instead of downloading')
    
    args = parser.parse_args()
    
    # Parse indicators if provided
    if args.indicators:
        indicator_list = args.indicators.split(',')
        selected_indicators = [ind for ind in indicator_list if ind in INDICATORS]
        
        if not selected_indicators:
            logger.error(f"No valid indicators found in: {args.indicators}")
            logger.info(f"Available indicators: {list(INDICATORS.keys())}")
            sys.exit(1)
    else:
        selected_indicators = None
    
    # Prepare economic data
    output_file = prepare_economic_data(
        start_date=args.start_date,
        end_date=args.end_date,
        indicators=selected_indicators,
        output_dir=args.output_dir,
        simulate=args.simulate
    )
    
    if output_file:
        print(f"Economic data prepared and saved to: {output_file}")
    else:
        print("Failed to prepare economic data")
