# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 10:19:09 2025

@author: zz1405
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib as mpl

# Set font and style
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Read CSV file
df = pd.read_csv('merged_output.csv')  # Please replace with actual file path

# Calculate COD/TN ratio
df['COD_TN_ratio'] = df['CODinf'] / df['TNinf']

# Filter plants with Energy gap <= 0 (achievable Energy neutral)
energy_neutral_df = df[df['Energy gap'] <= 0].copy()

print(f"Total number of plants: {len(df)}")
print(f"Number of Energy neutral plants: {len(energy_neutral_df)}")
print(f"Percentage: {len(energy_neutral_df)/len(df)*100:.2f}%")

# Key parameters for analysis with new column names
target_columns = [
    'Average_flow__MGD_', 
    'CODinf', 
    'BODinf', 
    'NH3_Ninf', 
    'TNinf', 
    'COD_TN_ratio'
]

# New column names for display
display_names = {
    'Average_flow__MGD_': 'Treatment scale (MGD)',
    'CODinf': 'Influent COD (mg/L)',
    'BODinf': 'Influent BOD (mg/L)',
    'NH3_Ninf': 'Influent NH$_3$-N (mg/L)',
    'TNinf': 'Influent TN (mg/L)',
    'COD_TN_ratio': 'Influent C/N ratio'
}

# Data cleaning - remove outliers for each variable using 95% percentile range (only for all plants)
def filter_95percentile_data(df, columns):
    """Filter data to keep only values within 2.5% to 97.5% percentile range for each column"""
    filtered_df = df.copy()
    for col in columns:
        if col in filtered_df.columns:
            lower_bound = filtered_df[col].quantile(0.025)
            upper_bound = filtered_df[col].quantile(0.975)
            filtered_df = filtered_df[(filtered_df[col] >= lower_bound) & (filtered_df[col] <= upper_bound)]
    return filtered_df

# Apply 95% percentile filtering only to all plants data
all_data_95pct = filter_95percentile_data(df[target_columns], target_columns)
# Keep Energy neutral data without filtering
energy_neutral_data = energy_neutral_df[target_columns].dropna()

print(f"\nAll plants data after 95% percentile filtering: {len(all_data_95pct)}")
print(f"Energy neutral plants data (no filtering): {len(energy_neutral_data)}")

# Calculate 95% CI for Energy neutral plants (using original data, no filtering)
en_ci_ranges = {}
en_means = {}

for col in target_columns:
    en_col_data = energy_neutral_data[col].dropna()
    if len(en_col_data) > 1:
        mean = en_col_data.mean()
        ci_low, ci_high = stats.t.interval(0.95, len(en_col_data)-1, 
                                         loc=mean, scale=stats.sem(en_col_data))
        en_ci_ranges[col] = (ci_low, ci_high)
        en_means[col] = mean

# 1. Distribution curves for all plants (95% percentile filtered) with Energy neutral 95% CI
print("\n" + "="*80)
print("Distribution Curves for All Plants (95% Percentile Filtered) with Energy Neutral 95% CI")
print("="*80)

# 增大图形尺寸
fig, axes = plt.subplots(3, 2, figsize=(20, 22))
axes = axes.flatten()

for i, col in enumerate(target_columns):
    data = all_data_95pct[col].dropna()
    
    if len(data) > 0:
        # Use kernel density estimation for smooth distribution curves
        kde = stats.gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 1000)
        y_vals = kde(x_vals)
        
        # 只在第一个子图添加label参数
        if i == 0:
            axes[i].plot(x_vals, y_vals, '#344966', linewidth=3, alpha=0.8, label='Distribution of all facilities')
        else:
            axes[i].plot(x_vals, y_vals, '#344966', linewidth=3, alpha=0.8)
            
        axes[i].fill_between(x_vals, y_vals, alpha=0.6, color='#b4cded')
        
        # Calculate and mark 95% CI range for Energy neutral plants (using original data)
        en_data = energy_neutral_data[col].dropna()
        if len(en_data) > 1 and col in en_ci_ranges:
            mean = en_means[col]
            ci_low, ci_high = en_ci_ranges[col]
            
            # Mark 95% CI range on the curve
            y_ci = kde(mean)  # Density value at mean
            axes[i].axvline(ci_low, color='#0D1821', linestyle='--', alpha=0.8, linewidth=2)
            axes[i].axvline(ci_high, color='#0D1821', linestyle='--', alpha=0.8, linewidth=2)
            axes[i].axvline(mean, color='#89937C', linestyle='-', alpha=0.8, linewidth=2.5)
            
            # Fill 95% CI area
            x_fill = np.linspace(ci_low, ci_high, 100)
            y_fill = kde(x_fill)
            
            # 只在第一个子图添加label参数
            if i == 0:
                axes[i].fill_between(x_fill, y_fill, alpha=0.3, color='#89937C', 
                                   label='Energy neutral/positive facilities')
            else:
                axes[i].fill_between(x_fill, y_fill, alpha=0.3, color='#89937C')
            
            # Add annotation - 增大字体
            axes[i].text(mean, y_ci*1.1, f'Mean\n{mean:.2f}', 
                       ha='center', va='bottom', fontsize=26,
                       bbox=dict(boxstyle='round', facecolor='#69385C', alpha=0.3))
        
        # Use display names for labels and titles - 增大字体
        display_name = display_names[col]
        axes[i].set_xlabel(display_name, fontsize=28)
        axes[i].set_ylabel('Probability density', fontsize=28)
        
        # 增大坐标轴刻度字体
        axes[i].tick_params(axis='both', which='major', labelsize=26)
        axes[i].tick_params(axis='both', which='minor', labelsize=24)
        
        # 创建不带单位的显示名称映射
        display_names_no_units = {
            'Average_flow__MGD_': 'Treatment scale',
            'CODinf': 'Influent COD',
            'BODinf': 'Influent BOD', 
            'NH3_Ninf': 'Influent NH$_3$-N',
            'TNinf': 'Influent TN',
            'COD_TN_ratio': 'Influent C/N ratio'
        }
        
        # 在标题中使用不带单位的名称 - 增大字体
        axes[i].set_title(f'{display_names_no_units[col]} distribution', fontsize=26, fontweight='bold', pad=20)
        
        # 只在第一个子图显示图例 - 增大字体
        if i == 0:
            axes[i].legend(fontsize=24)
            
        axes[i].grid(True, alpha=0.3)

# Remove extra subplots
for i in range(len(target_columns), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()