#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

def parse_australian_electricity_data(input_file, output_file):
    """
    Parse the Australian electricity demand dataset from TSF format to a proper CSV.
    
    Parameters
    ----------
    input_file : str
        Path to the input TSF/CSV file
    output_file : str
        Path to save the parsed CSV file
    """
    print(f"Reading data from {input_file}...")
    
    # Read the raw content
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Split by lines if any, otherwise treat as a single line
    lines = content.splitlines() if '\n' in content else [content]
    
    # Process each line
    all_data = []
    
    for line in lines:
        if not line.strip():
            continue
            
        # Split by colon to get parts
        parts = line.split(':')
        
        if len(parts) >= 4:
            series_id = parts[0]
            region = parts[1]
            start_time_str = parts[2]
            values_str = parts[3]
            
            # Parse values
            try:
                values = [float(v) for v in values_str.split(',') if v.strip()]
            except ValueError:
                print(f"Warning: Could not parse values for series {series_id}:{region}")
                continue
            
            # Parse start time
            try:
                # Try different date formats
                try:
                    start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H-%M-%S')
                except ValueError:
                    try:
                        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        start_time = datetime.strptime(start_time_str, '%Y-%m-%d')
            except ValueError:
                print(f"Warning: Could not parse timestamp {start_time_str} for series {series_id}:{region}")
                # Use a default timestamp
                start_time = datetime(2000, 1, 1)
            
            # Generate timestamps for each value (assuming hourly data)
            timestamps = [start_time + timedelta(hours=i) for i in range(len(values))]
            
            # Create data entries
            for i, (timestamp, value) in enumerate(zip(timestamps, values)):
                all_data.append({
                    'series_id': series_id,
                    'region': region,
                    'timestamp': timestamp,
                    'value': value,
                    'index': i  # Keep track of original position
                })
    
    # Convert to DataFrame
    if not all_data:
        print("No data could be parsed from the file.")
        return False
    
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    print(f"Saving parsed data to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print(f"Successfully parsed {len(df)} data points from {len(set(df['series_id'] + ':' + df['region']))} series.")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse Australian electricity demand data')
    parser.add_argument('input_file', type=str, help='Path to the input TSF/CSV file')
    parser.add_argument('--output', type=str, help='Path to save the parsed CSV file', default=None)
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    output_file = args.output if args.output else args.input_file.rsplit('.', 1)[0] + '_parsed.csv'
    
    # Parse the data
    success = parse_australian_electricity_data(args.input_file, output_file)
    
    if success:
        print(f"Data successfully parsed and saved to {output_file}")
    else:
        print("Failed to parse the data.")
