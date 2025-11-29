# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 15:50:36 2025

@author: zz1405
"""

# ANOVA-based variance decomposition for EUI metrics
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from tabulate import tabulate

def anova_variance_decomposition(data, group_col, value_col):

    groups = data.groupby(group_col)[value_col]
    
    # Overall statistics
    overall_mean = data[value_col].mean()
    n_total = len(data)
    n_groups = len(groups)
    
    # Between-group variance (SSB)
    ssb = 0
    for name, group in groups:
        n_group = len(group)
        group_mean = group.mean()
        ssb += n_group * (group_mean - overall_mean) ** 2
    
    # Within-group variance (SSW)
    ssw = 0
    for name, group in groups:
        ssw += ((group - group.mean()) ** 2).sum()
    
    # Total variance (SST)
    sst = ssb + ssw
    
    # Variance components
    between_group_variance = ssb / (n_groups - 1) if n_groups > 1 else 0
    within_group_variance = ssw / (n_total - n_groups)
    total_variance = sst / (n_total - 1)
    
    # Proportion of variance explained
    prop_between = ssb / sst if sst > 0 else 0
    prop_within = ssw / sst if sst > 0 else 0
    
    return {
        'total_variance': total_variance,
        'between_group_variance': between_group_variance,
        'within_group_variance': within_group_variance,
        'ssb': ssb,
        'ssw': ssw,
        'sst': sst,
        'prop_between': prop_between,
        'prop_within': prop_within,
        'n_groups': n_groups,
        'n_total': n_total
    }

def load_and_preprocess_data():
    """Load and preprocess the data"""
    # Province name to abbreviation mapping dictionary
    province_abbr = {
        'Beijing': 'BJ', 'Tianjin': 'TJ', 'Hebei': 'HEB', 'Shanxi': 'SX', 
        'Inner Mongolia': 'IM', 'Liaoning': 'LN', 'Jilin': 'JL', 
        'Heilongjiang': 'HLJ', 'Shanghai': 'SH', 'Jiangsu': 'JS', 
        'Zhejiang': 'ZJ', 'Anhui': 'AH', 'Fujian': 'FJ', 'Jiangxi': 'JX', 
        'Shandong': 'SD', 'Henan': 'HEN', 'Hubei': 'HUB', 'Hunan': 'HUN', 
        'Guangdong': 'GD', 'Guangxi': 'GX', 'Hainan': 'HAN', 'Chongqing': 'CQ', 
        'Sichuan': 'SC', 'Guizhou': 'GZ', 'Yunnan': 'YN', 'Xizang': 'XZ', 
        'Shaanxi': 'SNX', 'Gansu': 'GS', 'Qinghai': 'QH', 'Ningxia': 'NX', 
        'Xinjiang': 'XJ'
    }
    
    df = pd.read_csv('Data for figure 3b.csv')
    
    # Modify column names to use m³ superscript
    df.columns = [col.replace('m^3', 'm³') for col in df.columns]
    
    # Define numerical columns
    numeric_cols = ['CODinf', 'BODinf', 'TNinf', 'NH3_Ninf', 'TPinf', 'SSinf', 'Average_flow__MGD_',
                   'Flow based EUI (kWh/m³)', 'COD removal based EUI (kWh/kg COD removed)', 'TPeff', 'CODeff']
    
    # Add province abbreviation column
    df['Province_abbr'] = df['Province'].map(province_abbr)
    
    # Convert to numeric and drop missing values
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=numeric_cols + ['Province_abbr'])
    
    return df, numeric_cols

def calculate_province_averages(df, numeric_cols):
    """Calculate province averages"""
    return df.groupby('Province_abbr')[numeric_cols].mean().reset_index()

def perform_clustering(province_avg):
    """Perform clustering based on influent characteristics"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    # Use influent characteristics for clustering
    clustering_features = ['CODinf', 'BODinf', 'TNinf', 'NH3_Ninf', 'TPinf', 'SSinf', 'Average_flow__MGD_']
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(province_avg[clustering_features])
    
    n_clusters = min(6, len(province_avg)-1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    province_avg['Cluster'] = kmeans.fit_predict(scaled_features)
    
    return province_avg

def main():
    print("=" * 80)
    print("ANOVA-BASED VARIANCE DECOMPOSITION ANALYSIS")
    print("=" * 80)
    

    print("Loading and preprocessing data...")
    df, numeric_cols = load_and_preprocess_data()
    
    # Calculate province averages
    province_avg = calculate_province_averages(df, numeric_cols)
    
    # Perform clustering
    print("Performing clustering analysis...")
    province_avg = perform_clustering(province_avg)
    
    # EUI metrics to analyze
    eui_metrics = ['Flow based EUI (kWh/m³)', 'COD removal based EUI (kWh/kg COD removed)']
    
    # Store results
    variance_results = []
    anova_results = []
    
    print("\n" + "=" * 80)
    print("VARIANCE DECOMPOSITION RESULTS")
    print("=" * 80)
    
    for metric in eui_metrics:
        print(f"\n--- Analysis for {metric} ---")
        
        # Variance decomposition
        var_decomp = anova_variance_decomposition(province_avg, 'Cluster', metric)
        variance_results.append({
            'Metric': metric,
            'Total Variance': var_decomp['total_variance'],
            'Between-Group Variance': var_decomp['between_group_variance'],
            'Within-Group Variance': var_decomp['within_group_variance'],
            'Proportion Between': var_decomp['prop_between'],
            'Proportion Within': var_decomp['prop_within']
        })
        
        # Print detailed results
        print(f"Total Variance: {var_decomp['total_variance']:.6f}")
        print(f"Between-Group Variance: {var_decomp['between_group_variance']:.6f}")
        print(f"Within-Group Variance: {var_decomp['within_group_variance']:.6f}")
        print(f"Proportion of variance between groups: {var_decomp['prop_between']:.4f} ({var_decomp['prop_between']*100:.2f}%)")
        print(f"Proportion of variance within groups: {var_decomp['prop_within']:.4f} ({var_decomp['prop_within']*100:.2f}%)")
        
        # ANOVA test
        groups = []
        for cluster in sorted(province_avg['Cluster'].unique()):
            cluster_data = province_avg[province_avg['Cluster'] == cluster][metric]
            groups.append(cluster_data)
        
        f_stat, p_value = f_oneway(*groups)
        anova_results.append({
            'f_statistic': f_stat,
            'p_value': p_value
        })
        
        print(f"ANOVA F-statistic: {f_stat:.4f}")
        print(f"ANOVA p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("*** Significant differences between clusters (p < 0.05) ***")
        else:
            print("No significant differences between clusters")
        

    
    summary_table = []
    for i, metric in enumerate(eui_metrics):
        var_result = variance_results[i]
        anova_result = anova_results[i]
        
        summary_table.append({
            'Metric': metric,
            'Total Variance': f"{var_result['Total Variance']:.6f}",
            'Between (%)': f"{var_result['Proportion Between']*100:.2f}%",
            'Within (%)': f"{var_result['Proportion Within']*100:.2f}%",
            'F-statistic': f"{anova_result['f_statistic']:.4f}",
            'p-value': f"{anova_result['p_value']:.4f}",
            'Significant': 'Yes' if anova_result['p_value'] < 0.05 else 'No',
            'Effect Size': f"{var_result['Proportion Between']:.4f}"
        })
    
    print(tabulate(summary_table, headers='keys', tablefmt='grid'))
    


if __name__ == "__main__":
    main()