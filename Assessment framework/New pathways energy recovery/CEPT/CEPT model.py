# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 15:57:29 2025

@author: zz1405
"""

import pandas as pd
import warnings
from typing import Optional

def calculate_cept_performance(input_file: str, output_file: str) -> Optional[pd.DataFrame]:
    try:
        # Read and validate input data
        df = pd.read_csv(input_file)
        required_cols = ['Average flow (MGD)', 'CODinf', 'BODeff', 'TNinf', 'SSinf']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Input file missing required columns: {', '.join(missing)}")
            
    except Exception as e:
        print(f"File read failed: {str(e)}")
        return None

    # Define CEPT removal rates
    REMOVAL_RATES = {
        'COD': 0.75,  # 75% COD removal
        'BOD': 0.5,   # 50% BOD removal  
        'TN': 0.15,   # 15% TN removal
        'SS': 0.8    # 80% TSS removal
    }
    dosage = 50
    cept_pro =5
    # Calculate effluent concentrations
    df['COD_out_CEPT'] = df['CODinf'] * (1 - REMOVAL_RATES['COD'])
    df['TN_out_CEPT'] = df['TNinf'] * (1 - REMOVAL_RATES['TN'])
    
    # Calculate C/N ratio (handle division by zero)
    df['C/N ratio_CEPT'] = df.apply(
        lambda x: x['COD_out_CEPT'] / x['TN_out_CEPT'] if x['TN_out_CEPT'] > 0 else None,
        axis=1
    )
    
    # Calculate mass removal rates (kg/year)
    df['BOD removed (kg/a)'] = (df['BODinf'] * REMOVAL_RATES['BOD'] * 
                               df['Average flow (MGD)'] * 1381675 / 1000)
    
    df['SS removed (kg/a)'] = (df['SSinf'] * REMOVAL_RATES['SS'] * 
                               df['Average flow (MGD)'] * 1381675 / 1000)
    
    # Calculate sludge production (tonnes/year)
    df['Sludge yield by CEPT (tonne/a)'] = (
        df['SS removed (kg/a)'] / 1000 + 
        df['Average flow (MGD)'] * 1378675 * dosage * cept_pro / 1e6
    )
    
    return df

def calculate_energy_consumption(df: pd.DataFrame) -> pd.DataFrame:

    # Electricity consumption (kWh/year)
    cept_EUI = 0.03
    df['CEPT electricity consumption (kWh/a)'] = (
        cept_EUI * df['Average flow (MGD)'] * 1378675 + 
        df['Flow based EUI (kWh/m^3)'] * df['Average flow (MGD)'] * 1378675  +
        (df['Sludge yield by CEPT (tonne/a)']) * 20
    )
    
    # Heat consumption (kWh/year)
    df['CEPT heat consumption (kWh/a)'] = (
        df['Sludge yield by CEPT (tonne/a)'] / 365 * 1000 * 4.18 * 
        df['annual_hdd'] / 3600
    )
    
    return df

def calculate_biogas_production(Q: float, BOD_removed: float) -> dict:

    # Constants
    Y = 0.05      # Yield coefficient (gVSS/gBOD)
    E = 0.8       # Digestion efficiency
    k_d = 0.03    # Decay coefficient (d⁻¹)
    Y_c = 20      # SRT (days)
    methane_frac = 0.65  # Methane fraction in biogas
    
    # Unit conversion
    Q_m3a = Q * 1381674  # Convert MGD to m³/year
    
    # BOD calculations
    BOD5_kg_d = BOD_removed
    BODL_d = BOD5_kg_d / 0.68  # Convert to BODL
    
    # 1. Calculate biomass production (kg VSS/d)
    Px_d = (Y * E * BODL_d) / (1 + k_d * Y_c)
    
    # 2. Calculate methane production (m³/year)
    methane_m3_a = 0.35 * (E * BODL_d - 1.42 * Px_d)
    
    # 3. Energy calculations
    energy_kwh_a = methane_m3_a * 9.94  # Methane energy content
    
    return {
        'CEPT electricity recovery potential (kWh/a)': round(energy_kwh_a * 0.35, 2),
        'CEPT heat recovery potential (kWh/a)': round(energy_kwh_a * 0.4, 2)
    }

if __name__ == "__main__":
    input_csv = r"C:\Users\zz1405\OneDrive - Princeton University\Documents\Work 2_CN energy\Submission to One Earth\Github\Dataset processing\Cleaned final dataset.csv"
    output_csv = "CEPT_output.csv"
    
    # Run calculations
    result_df = calculate_cept_performance(input_csv, output_csv)
    
    if result_df is not None:
        result_df = calculate_energy_consumption(result_df)
        
        # Calculate biogas for each row
        energy_recovery = []
        for _, row in result_df.iterrows():
            recovery = calculate_biogas_production(
                Q=row['Average flow (MGD)'],
                BOD_removed=row['BOD removed (kg/a)']
            )
            energy_recovery.append(recovery)
            
        # Add energy recovery columns
        energy_df = pd.DataFrame(energy_recovery)
        result_df = pd.concat([result_df, energy_df], axis=1)
        
        # Save results
        result_df.to_csv(output_csv, index=False)
        print(f"Calculation completed. Results saved to {output_csv}")