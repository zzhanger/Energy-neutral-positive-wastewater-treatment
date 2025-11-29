# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 19:03:27 2025

@author: zz1405
"""

import pandas as pd
import numpy as np

# Read input data
data = pd.read_csv('Energy_results.csv')

# Part 1: Identify facilities suitable for CEPT or HRAS (C/N ratio <= 2)
cept_filtered = data.loc[data['C/N ratio_CEPT'] <= 2].copy()
cept_filtered['If CEPT'] = 'CEPT'

hras_filtered = data.loc[data['C/N_ratio_HRAS'] <= 2].copy()
hras_filtered['If HRAS'] = 'HRAS'

# Merge CEPT and HRAS flags with original data
data = pd.merge(
    data,
    cept_filtered[['Plant ID', 'If CEPT']],
    on='Plant ID',
    how='left'
)

data = pd.merge(
    data,
    hras_filtered[['Plant ID', 'If HRAS']],
    on='Plant ID',
    how='left'
)

# Part 2: Calculate energy consumption for CEPT-PDA and HRAS-PNA processes

# HRAS-PNA energy calculations
hras_mask = data['If HRAS'] == 'HRAS'
data.loc[hras_mask, 'Electricity consumption HRAS-PNA'] = (
    data.loc[hras_mask, 'Average flow (MGD)'] * 1381675 * (0.122 + 0.03 + 0.01) + 
    data.loc[hras_mask, 'Average flow (MGD)'] * 1381675 * data.loc[hras_mask, 'EUI (kWh/m3)'] * 0.4 +
    data.loc[hras_mask, 'Average flow (MGD)'] * 1381675 * 0.5 * data.loc[hras_mask, 'EUI (kWh/m3)'] * 0.6 * 0.4 +
    (data.loc[hras_mask, 'Sludge_TSS (g/a)'] / 1e6) * 20
)

data.loc[hras_mask, 'Heat consumption HRAS-PNA'] = (
    (data.loc[hras_mask, 'Sludge_TSS (g/a)'] /0.2 / 1e3) / 365 * 4.18 * 
    data.loc[hras_mask, 'annual_hdd'] / 3600
)

data.loc[hras_mask, 'HRAS-PNA energy consumption'] = (
    data.loc[hras_mask, 'Heat consumption HRAS-PNA'] + 
    data.loc[hras_mask, 'Electricity consumption HRAS-PNA']
)

# CEPT-PDA energy calculations
cept_mask = data['If CEPT'] == 'CEPT'
data.loc[cept_mask, 'Electricity consumption CEPT-PDA'] = (
    data.loc[cept_mask, 'Average flow (MGD)'] * 1381675 * 0.03 + 
    data.loc[cept_mask, 'Average flow (MGD)'] * 1381675 * data.loc[cept_mask, 'EUI (kWh/m3)'] * 0.4 +
    data.loc[cept_mask, 'Average flow (MGD)'] * 1381675 * 0.5 * data.loc[cept_mask, 'EUI (kWh/m3)'] * 0.6 +
    data.loc[cept_mask, 'Sludge yield by CEPT (tonne/a)'] * 20
)

data.loc[cept_mask, 'Heat consumption CEPT-PDA'] = (
    data.loc[cept_mask, 'Sludge yield by CEPT (tonne/a)'] / 0.2 * 1000 / 365 * 
    4.18 * data.loc[cept_mask, 'annual_hdd'] / 3600
)

data.loc[cept_mask, 'CEPT-PDA energy consumption'] = (
    data.loc[cept_mask, 'Heat consumption CEPT-PDA'] + 
    data.loc[cept_mask, 'Electricity consumption CEPT-PDA']
)

# Initialize selection columns
data['Selected_Process'] = None
data['Selected_Electricity'] = np.nan
data['Selected_Heat'] = np.nan

# Process selection logic
for idx, row in data.iterrows():
    current_energy = row['Current electricity recovery potential (kWh/a)']
    cept_energy = row['CEPT electricity recovery potential (kWh/a)'] if pd.notna(row['CEPT electricity recovery potential (kWh/a)']) else 0
    hras_energy = row['Electricity recovery potential (kWh/a)_HRAS'] if pd.notna(row['Electricity recovery potential (kWh/a)_HRAS']) else 0
    
    cept_cn_ok = pd.notna(row['C/N ratio_CEPT']) and row['C/N ratio_CEPT'] <= 2
    hras_cn_ok = pd.notna(row['C/N_ratio_HRAS']) and row['C/N_ratio_HRAS'] <= 2
    
    if cept_energy == 0 and hras_energy == 0:
        data.at[idx, 'Selected_Process'] = 'Current'
        data.at[idx, 'Selected_Electricity'] = row['Current electricity recovery potential (kWh/a)']
        data.at[idx, 'Selected_Heat'] = row['Current effective heat recovery potential (kWh/a)']
    else:
        if cept_cn_ok and hras_cn_ok:
            if cept_energy >= hras_energy and cept_energy > current_energy:
                selected = 'CEPT'
            elif hras_energy > cept_energy and hras_energy > current_energy:
                selected = 'HRAS'
            else:
                selected = 'Current'
        elif cept_cn_ok:
            selected = 'CEPT' if cept_energy > current_energy else 'Current'
        elif hras_cn_ok:
            selected = 'HRAS' if hras_energy > current_energy else 'Current'
        else:
            selected = 'Current'
        
        data.at[idx, 'Selected_Process'] = selected
        if selected == 'CEPT':
            data.at[idx, 'Selected_Electricity'] = row['CEPT electricity recovery potential (kWh/a)']
            data.at[idx, 'Selected_Heat'] = row['CEPT heat recovery potential (kWh/a)']
        elif selected == 'HRAS':
            data.at[idx, 'Selected_Electricity'] = row['Electricity recovery potential (kWh/a)_HRAS']
            data.at[idx, 'Selected_Heat'] = row['Heat recovery potential (kWh/a)_HRAS']
        else:
            data.at[idx, 'Selected_Electricity'] = row['Current electricity recovery potential (kWh/a)']
            data.at[idx, 'Selected_Heat'] = row['Current heat recovery potential (kWh/a)']

# Set final energy consumption values based on selected process
current_mask = data['Selected_Process'] == 'Current'
cept_mask = data['Selected_Process'] == 'CEPT'
hras_mask = data['Selected_Process'] == 'HRAS'

data.loc[current_mask, 'Electricity consumption after upgrade'] = data['Current electricity consumption (kWh/a)']
data.loc[current_mask, 'Heat consumption after upgrade'] = data['Current heat consumption (kWh/a)']

data.loc[cept_mask, 'Electricity consumption after upgrade'] = data['Electricity consumption CEPT-PDA']
data.loc[cept_mask, 'Heat consumption after upgrade'] = data['Heat consumption CEPT-PDA']

data.loc[hras_mask, 'Electricity consumption after upgrade'] = data['Electricity consumption HRAS-PNA']
data.loc[hras_mask, 'Heat consumption after upgrade'] = data['Heat consumption HRAS-PNA']


# Save results
data.to_csv('New process selection.csv', index=False)