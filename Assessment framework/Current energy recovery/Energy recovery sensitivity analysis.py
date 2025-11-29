import pandas as pd
import numpy as np

# Read data
data_path = r"C:\Users\zz1405\OneDrive - Princeton University\Documents\Work 2_CN energy\Submission to One Earth\Github\Dataset processing\Cleaned final dataset.csv"
data = pd.read_csv(data_path)

def biogas_production(Q, BOD_in, BOD_out, 
                     Y_c=20, Y=0.05, E=0.8, k_d=0.03, theta=1/3, R=0.68,
                     u=0.35, F=1.42):
    """Calculate total biogas energy production (kWh/year)"""
    BOD5_removed = (BOD_in - BOD_out) * theta
    BOD5_kg_d = BOD5_removed * Q * 1381674 * 1e-3
    BODL_d = BOD5_kg_d / R
    
    Px_d = (Y * E * BODL_d) / (1 + k_d * Y_c)
    methane_m3_a = u * (E * BODL_d - F * Px_d)
    
    return methane_m3_a * 9.94  # Convert to kWh/year

# Parameter variation ranges
param_ranges = {
    'Y': (0.04, 0.1),      # min, max
    'E': (0.6, 0.9),
    'k_d': (0.02, 0.4)
}

# Sensitivity analysis for each wastewater treatment plant
sensitivity_results = []

for idx, row in data.iterrows():
    try:
        plant_results = {
            'Plant_ID': row['Plant ID'] if 'Plant ID' in row else idx,
            'Flow_MGD': float(row['Average flow (MGD)']),
            'BOD_in': float(row['BODinf']),
            'BOD_out': float(row['BODeff'])
        }
        
        # Sensitivity analysis for each parameter
        for param in param_ranges:
            # Fix other parameters at default values
            params = {
                'Y': 0.05,
                'E': 0.8,
                'k_d': 0.03,
                'Y_c': 20,
                'theta': 1/3,
                'R': 0.68,
                'u': 0.35,
                'F': 1.42
            }
            
            # Calculate output at minimum value
            params[param] = param_ranges[param][0]
            energy_min = biogas_production(
                Q=plant_results['Flow_MGD'],
                BOD_in=plant_results['BOD_in'],
                BOD_out=plant_results['BOD_out'],
                **params
            )
            
            # Calculate output at maximum value
            params[param] = param_ranges[param][1]
            energy_max = biogas_production(
                Q=plant_results['Flow_MGD'],
                BOD_in=plant_results['BOD_in'],
                BOD_out=plant_results['BOD_out'],
                **params
            )
            
            # Calculate absolute and relative differences
            abs_diff = energy_max - energy_min
            rel_diff = (energy_max - energy_min) / ((energy_max + energy_min)/2) * 100  # Using average as denominator
            
            plant_results.update({
                f'{param}_min_energy': energy_min,
                f'{param}_max_energy': energy_max,
                f'{param}_abs_diff': abs_diff,
                f'{param}_rel_diff(%)': rel_diff
            })
        
        sensitivity_results.append(plant_results)
        
    except Exception as e:
        print(f"Error processing plant {idx}: {str(e)}")

# Convert to DataFrame and save results
if sensitivity_results:
    sensitivity_df = pd.DataFrame(sensitivity_results)
    
    # Calculate average differences
    summary_data = []
    for param in param_ranges:
        avg_min = sensitivity_df[f'{param}_min_energy'].mean()
        avg_max = sensitivity_df[f'{param}_max_energy'].mean()
        avg_abs_diff = sensitivity_df[f'{param}_abs_diff'].mean()
        avg_rel_diff = sensitivity_df[f'{param}_rel_diff(%)'].mean()
        
        summary_data.append({
            'Parameter': param,
            'Avg_Energy_at_Min': avg_min,
            'Avg_Energy_at_Max': avg_max,
            'Avg_Absolute_Difference': avg_abs_diff,
            'Avg_Relative_Difference(%)': avg_rel_diff,
            'Impact_Range': avg_max - avg_min
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by impact range
    summary_df = summary_df.sort_values('Impact_Range', ascending=False)
    
    print("\nSensitivity Analysis Summary:")
    print(summary_df)
