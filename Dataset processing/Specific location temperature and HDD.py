# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 17:07:56 2025

@author: zz1405
"""

import pandas as pd
import requests
from tqdm import tqdm  # Progress bar tool

def get_daily_temperature(lat, lon, year):
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M&community=RE&longitude={lon}&latitude={lat}&start={year}0101&end={year}1231&format=JSON"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        temps = data['properties']['parameter']['T2M']
        return pd.DataFrame.from_dict(temps, orient='index', columns=['temp_avg'])
    except Exception as e:
        print(f"Failed to retrieve data for ({lat}, {lon}): {str(e)}")
        return None

def calculate_hdd(temp_df, base_temp=35):  # Heating degree days temperature set as 35 degrees C
    if temp_df is not None:
        temp_df['hdd'] = (base_temp - temp_df['temp_avg']).clip(lower=0)
        return temp_df['hdd'].sum()
    return None

def process_locations(input_csv, output_csv, year=2024, base_temp=35):
    df = pd.read_csv(input_csv)
    
    # Check for required columns
    required_cols = ['Latitude', 'Longitude']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("The CSV file must contain Latitude and Longitude columns.")
    
    # Add new columns for results
    df['annual_hdd'] = None
    df['avg_temp'] = None
    
    # Loop through each location to get data
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        temp_data = get_daily_temperature(row['Latitude'], row['Longitude'], year)
        if temp_data is not None:
            df.at[idx, 'annual_hdd'] = calculate_hdd(temp_data, base_temp)
            df.at[idx, 'avg_temp'] = temp_data['temp_avg'].mean()
    
    # Save results
    df.to_csv(output_csv, index=False)
    print(f"Processing complete! Results saved to: {output_csv}")

process_locations(
    input_csv='Final dataset.csv',
    output_csv='Final dataset with HDD.csv',
    year=2024,
    base_temp=35
)