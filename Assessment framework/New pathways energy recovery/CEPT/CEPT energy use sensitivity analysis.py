# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 18:37:48 2025

@author: zz1405
"""
import pandas as pd
import numpy as np

def perform_sensitivity_analysis(input_file: str):
    """Perform sensitivity analysis on CEPT performance parameters"""
    
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded data with {len(df)} rows")
        
    except Exception as e:
        print(f"File read failed: {str(e)}")
        return None

    # Baseline parameters
    baseline_removal = {'COD': 0.75, 'BOD': 0.5, 'TN': 0.15, 'SS': 0.8}
    baseline_dosage = 50
    baseline_cept_pro = 5
    baseline_cept_EUI = 0.03
    
    # Store sensitivity results
    sensitivity_results = []
    
    def calculate_cept_performance(row, removal_rates, dosage, cept_pro, cept_EUI):
        """Calculate CEPT performance with given parameters"""
        
        # Sludge production
        ss_removed_kg_a = (row.get('SSinf', 200) * removal_rates['SS'] * 
                          row['Average flow (MGD)'] * 1381675 / 1000)
        
        sludge_yield = (
            ss_removed_kg_a / 1000 + 
            row['Average flow (MGD)'] * 1378675 * dosage * cept_pro / 1e6
        )
        
        # Energy consumption
        cept_electricity = (
            cept_EUI * row['Average flow (MGD)'] * 1378675 + 
            row.get('Current electricity consumption (kWh/a)', 0) +
            sludge_yield * 20
        )
        
        cept_heat = (
            sludge_yield / 365 * 1000 * 4.18 * 
            row['annual_hdd'] / 3600
        )
        
        total_energy = cept_electricity + cept_heat
        
        # C/N ratio
        cod_out = row.get('CODinf', 300) * (1 - removal_rates['COD'])
        tn_out = row.get('TNinf', 30) * (1 - removal_rates['TN'])
        c_n_ratio = cod_out / tn_out if tn_out > 0 else None
        
        # Biogas production
        bod_removed_kg_a = (row.get('CODinf', 300) * removal_rates['BOD'] * 
                           row['Average flow (MGD)'] * 1381675 / 1000)
        
        # Biogas calculation
        BOD5_kg_d = bod_removed_kg_a / 365
        BODL_d = BOD5_kg_d / 0.68
        Px_d = (0.05 * 0.8 * BODL_d) / (1 + 0.03 * 20)
        methane_m3_a = 0.35 * (0.8 * BODL_d - 1.42 * Px_d) * 365
        energy_kwh_a = methane_m3_a * 9.94
        
        return {
            'C/N ratio': c_n_ratio,
            'Total energy': total_energy,
            'Energy recovery': energy_kwh_a * 0.75
        }

    # Use first row for analysis
    row = df.iloc[0]

    # 1. Removal rates sensitivity (±5%)
    for param in ['COD', 'BOD', 'TN', 'SS']:
        for delta in [-0.05, 0, 0.05]:
            modified_rates = baseline_removal.copy()
            modified_rates[param] = max(0, min(1, modified_rates[param] + delta))
            
            results = calculate_cept_performance(
                row, modified_rates, baseline_dosage, baseline_cept_pro, baseline_cept_EUI
            )
            
            sensitivity_results.append({
                'Parameter': f'{param} removal',
                'Value': modified_rates[param],
                'C/N ratio': results['C/N ratio'],
                'Energy consumption': results['Total energy'],
                'Energy recovery': results['Energy recovery']
            })

    # 2. Dosage sensitivity (±10%)
    for delta_pct in [-10, 0, 10]:
        modified_dosage = baseline_dosage * (1 + delta_pct/100)
        
        results = calculate_cept_performance(
            row, baseline_removal, modified_dosage, baseline_cept_pro, baseline_cept_EUI
        )
        
        sensitivity_results.append({
            'Parameter': 'Dosage',
            'Value': modified_dosage,
            'C/N ratio': results['C/N ratio'],
            'Energy consumption': results['Total energy'],
            'Energy recovery': results['Energy recovery']
        })

    # 3. CEPT parameters sensitivity (±10%)
    for param_name, base_value in [('CEPT pro', baseline_cept_pro), ('CEPT EUI', baseline_cept_EUI)]:
        for delta_pct in [-10, 0, 10]:
            modified_value = base_value * (1 + delta_pct/100)
            
            if param_name == 'CEPT pro':
                results = calculate_cept_performance(
                    row, baseline_removal, baseline_dosage, modified_value, baseline_cept_EUI
                )
            else:
                results = calculate_cept_performance(
                    row, baseline_removal, baseline_dosage, baseline_cept_pro, modified_value
                )
            
            sensitivity_results.append({
                'Parameter': param_name,
                'Value': modified_value,
                'C/N ratio': results['C/N ratio'],
                'Energy consumption': results['Total energy'],
                'Energy recovery': results['Energy recovery']
            })

    return pd.DataFrame(sensitivity_results)

def analyze_sensitivity(sensitivity_df):
    """Analyze and print sensitivity results"""
    base_case = sensitivity_df[
        (sensitivity_df['Value'].isin([0.75, 0.5, 0.15, 0.8, 50, 5, 0.03])) &
        (sensitivity_df.groupby('Parameter')['Value'].transform(lambda x: x == x.iloc[1]))
    ]
    
    print("CEPT Sensitivity Analysis Results")
    print("=" * 50)
    
    for param in sensitivity_df['Parameter'].unique():
        param_data = sensitivity_df[sensitivity_df['Parameter'] == param]
        base_energy = base_case[base_case['Parameter'] == param]['Energy consumption'].values[0]
        
        print(f"\n{param}:")
        for _, row in param_data.iterrows():
            if row['Value'] != base_energy:
                change_pct = (row['Energy consumption'] - base_energy) / base_energy * 100
                direction = "increase" if change_pct > 0 else "decrease"
                print(f"  {row['Value']}: {abs(change_pct):.1f}% {direction}")

if __name__ == "__main__":
    input_csv = r"C:\Users\zz1405\OneDrive - Princeton University\Documents\Work 2_CN energy\Submission to One Earth\Github\Assessment framework\New pathways energy recovery\CEPT\CEPT_output.csv"
    
    sensitivity_df = perform_sensitivity_analysis(input_csv)
    
    if sensitivity_df is not None:
        sensitivity_df.to_csv('CEPT_sensitivity_results.csv', index=False)
        analyze_sensitivity(sensitivity_df)
        print(f"\nResults saved to CEPT_sensitivity_results.csv")
    else:
        print("Sensitivity analysis failed")
