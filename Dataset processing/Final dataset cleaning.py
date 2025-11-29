# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 11:58:49 2025

@author: zz1405
"""

import pandas as pd
import numpy as np
import os

# 1. Read CSV File
file_path = r"C:\Users\zz1405\OneDrive - Princeton University\Documents\Work 2_CN energy\Submission to One Earth\Github\Dataset processing\Final dataset with HDD.csv"
df = pd.read_csv(file_path)


# 2. Data Cleaning
print(f"Original data total rows: {len(df)}")

# Remove rows with zeros
df = df.replace(0, np.nan).dropna()
print(f"Total rows after removing zeros: {len(df)}")

# Ensure COD is greater than BOD and TN is greater than NH3-N
df = df[df['CODinf'] >= df['BODinf']]
print(f"Total rows after ensuring COD >= BOD: {len(df)}")
df = df[df['NH3-Ninf'] <= df['TNinf']]
print(f"Total rows after ensuring NH3-N <= TN: {len(df)}")

all_typical_data = pd.DataFrame()

# 3. Group by Treatment Process and Process Data
for process, group in df.groupby('Treatment process'):
    eui_mean = group['EUI (kWh/m3)'].mean()
    eui_std = group['EUI (kWh/m3)'].std()
    lower_bound = eui_mean - 2 * eui_std
    upper_bound = eui_mean + 2 * eui_std
    
    # Filter typical and non-typical data
    typical_data = group[(group['EUI (kWh/m3)'] >= lower_bound) & (group['EUI (kWh/m3)'] <= upper_bound)]
    non_typical_data = group[(group['EUI (kWh/m3)'] < lower_bound) | (group['EUI (kWh/m3)'] > upper_bound)]
    
    # Create a safe filename
    safe_name = "".join([c if c.isalnum() else "_" for c in str(process)]).rstrip("_")
    
    # Append typical data to the total typical data
    all_typical_data = pd.concat([all_typical_data, typical_data], ignore_index=True)
    
    # Print processing information
    print(f"\nProcessing: {process}")
    print(f"EUI Range: {lower_bound:.2f} to {upper_bound:.2f} kWh/m3")
    print(f"Exclusion Rate: {len(non_typical_data) / len(group) * 100:.1f}%")
all_typical_data.to_csv('Cleaned final dataset.csv')