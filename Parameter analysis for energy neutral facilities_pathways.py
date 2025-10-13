# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 14:11:45 2025

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
# Filter plants with Energy gap after selection <= 0
energy_neutral_after_selection_df = df[df['Energy gap after selection'] <= 0].copy()

print(f"Total number of plants: {len(df)}")
print(f"Number of Energy neutral plants: {len(energy_neutral_df)}")
print(f"Number of Energy neutral plants after selection: {len(energy_neutral_after_selection_df)}")
print(f"Percentage of Energy neutral plants: {len(energy_neutral_df)/len(df)*100:.2f}%")
print(f"Percentage of Energy neutral plants after selection: {len(energy_neutral_after_selection_df)/len(df)*100:.2f}%")

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

# Display names without units for titles
display_names_no_units = {
    'Average_flow__MGD_': 'Treatment scale',
    'CODinf': 'Influent COD',
    'BODinf': 'Influent BOD', 
    'NH3_Ninf': 'Influent NH$_3$-N',
    'TNinf': 'Influent TN',
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
# Keep Energy neutral after selection data without filtering
energy_neutral_after_selection_data = energy_neutral_after_selection_df[target_columns].dropna()

print(f"\nAll plants data after 95% percentile filtering: {len(all_data_95pct)}")
print(f"Energy neutral plants data (no filtering): {len(energy_neutral_data)}")
print(f"Energy neutral plants after selection data (no filtering): {len(energy_neutral_after_selection_data)}")

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

# Calculate 95% CI for Energy neutral plants after selection (using original data, no filtering)
en_after_selection_ci_ranges = {}
en_after_selection_means = {}

for col in target_columns:
    en_col_data = energy_neutral_after_selection_data[col].dropna()
    if len(en_col_data) > 1:
        mean = en_col_data.mean()
        ci_low, ci_high = stats.t.interval(0.95, len(en_col_data)-1, 
                                         loc=mean, scale=stats.sem(en_col_data))
        en_after_selection_ci_ranges[col] = (ci_low, ci_high)
        en_after_selection_means[col] = mean

# Distribution curves for all plants (95% percentile filtered) with both Energy neutral 95% CI
print("\n" + "="*80)
print("Distribution Curves for All Plants (95% Percentile Filtered) with Energy Neutral 95% CI")
print("="*80)

fig, axes = plt.subplots(3, 2, figsize=(12, 16))
axes = axes.flatten()

for i, col in enumerate(target_columns):
    data = all_data_95pct[col].dropna()
    
    if len(data) > 0:
        # Use kernel density estimation for smooth distribution curves
        kde = stats.gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 1000)
        y_vals = kde(x_vals)
        
        axes[i].plot(x_vals, y_vals, '#344966', linewidth=2, alpha=0.8, label='Distribution of all facilities')
        axes[i].fill_between(x_vals, y_vals, alpha=0.6, color='#b4cded')
        
        # Calculate and mark 95% CI range for Energy neutral plants (using original data)
        en_data = energy_neutral_data[col].dropna()
        en_after_selection_data = energy_neutral_after_selection_data[col].dropna()
        
        # Plot Energy neutral 95% CI
        if len(en_data) > 1 and col in en_ci_ranges:
            mean = en_means[col]
            ci_low, ci_high = en_ci_ranges[col]
            
            # Mark 95% CI range on the curve
            y_ci = kde(mean)  # Density value at mean
            axes[i].axvline(ci_low, color='#0D1821', linestyle='--', alpha=0.8, linewidth=1.5)
            axes[i].axvline(ci_high, color='#0D1821', linestyle='--', alpha=0.8, linewidth=1.5)
            axes[i].axvline(mean, color='#89937C', linestyle='-', alpha=0.8, linewidth=2)
            
            # Fill 95% CI area
            x_fill = np.linspace(ci_low, ci_high, 100)
            y_fill = kde(x_fill)
            axes[i].fill_between(x_fill, y_fill, alpha=0.3, color='#89937C', 
                               label='Energy neutral facilities')
        
        # Plot Energy neutral after selection 95% CI
        if len(en_after_selection_data) > 1 and col in en_after_selection_ci_ranges:
            mean_after = en_after_selection_means[col]
            ci_low_after, ci_high_after = en_after_selection_ci_ranges[col]
            
            # Mark 95% CI range on the curve
            y_ci_after = kde(mean_after)  # Density value at mean
            axes[i].axvline(ci_low_after, color='#A26769', linestyle='--', alpha=0.8, linewidth=1.5)
            axes[i].axvline(ci_high_after, color='#A26769', linestyle='--', alpha=0.8, linewidth=1.5)
            axes[i].axvline(mean_after, color='#CE7B91', linestyle='-', alpha=0.8, linewidth=2)
            
            # Fill 95% CI area
            x_fill_after = np.linspace(ci_low_after, ci_high_after, 100)
            y_fill_after = kde(x_fill_after)
            axes[i].fill_between(x_fill_after, y_fill_after, alpha=0.3, color='#CE7B91', 
                               label='Energy neutral facilities under\n sustainable pathways')
            
            # Add annotations for both means
            axes[i].text(mean, y_ci*1.1, f'Mean\n{mean:.2f}', 
                       ha='center', va='bottom', fontsize=14,
                       bbox=dict(boxstyle='round', facecolor='#89937C', alpha=0.3))
            axes[i].text(mean_after, y_ci_after*1.1, f'Mean\n{mean_after:.2f}', 
                       ha='right', va='top', fontsize=14,
                       bbox=dict(boxstyle='round', facecolor='#CE7B91', alpha=0.5))
        
        # Use display names for labels and titles
        display_name = display_names[col]
        axes[i].set_xlabel(display_name, fontsize=16)
        axes[i].set_ylabel('Probability density', fontsize=16)
        axes[i].set_title(f'{display_names_no_units[col]} distribution', fontsize=16, fontweight='bold')
        axes[i].legend(fontsize=12)
        axes[i].grid(True, alpha=0.3)

# Remove extra subplots
for i in range(len(target_columns), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()









import matplotlib.gridspec as gridspec
from scipy import stats
import seaborn as sns

# Extract data
x_data = df['Optimal EUI (kWh/m3)']
y_data = df['Recoverable energy density (kWh/m3)']

# Filter out NaN values
valid_data = df[['Optimal EUI (kWh/m3)', 'Recoverable energy density (kWh/m3)']].dropna()
x_data = valid_data['Optimal EUI (kWh/m3)']
y_data = valid_data['Recoverable energy density (kWh/m3)']

# Create joint plot using seaborn
g = sns.jointplot(
    x='Optimal EUI (kWh/m3)', 
    y='Recoverable energy density (kWh/m3)', 
    data=valid_data,
    kind='scatter',
    alpha=0.6,
    color='#476EAE',
    height=10,
    joint_kws={'s': 50},
    marginal_kws={'color': '#476EAE', 'fill': True}
)

# Add y=x line (Energy neutral line)
max_val = max(valid_data['Optimal EUI (kWh/m3)'].max(), valid_data['Recoverable energy density (kWh/m3)'].max())
x_line = np.linspace(0, max_val, 100)
y_line = x_line
g.ax_joint.plot(x_line, y_line, '#48B3AF', linewidth=2, label='Energy production from wastewater treatment (Y=X)')

# Add y=0.5x line (50% energy coverage line)
y_half_line = 0.5 * x_line
g.ax_joint.plot(x_line, y_half_line, '#A7E399', linewidth=2, label='50% energy self sufficient (Y=0.5X)')

# Count points in each region
total_points = len(valid_data)
above_y_equal_x = len(valid_data[valid_data['Recoverable energy density (kWh/m3)'] > valid_data['Optimal EUI (kWh/m3)']])
between_lines = len(valid_data[(valid_data['Recoverable energy density (kWh/m3)'] <= valid_data['Optimal EUI (kWh/m3)']) & 
                              (valid_data['Recoverable energy density (kWh/m3)'] > 0.5 * valid_data['Optimal EUI (kWh/m3)'])])
below_y_equal_half_x = len(valid_data[valid_data['Recoverable energy density (kWh/m3)'] <= 0.5 * valid_data['Optimal EUI (kWh/m3)']])

# Add text annotations with counts
g.ax_joint.text(0.7 * max_val, 0.8 * max_val, f'{above_y_equal_x} facilities', 
         fontsize=12, bbox=dict(boxstyle='round', facecolor='#48B3AF', alpha=0.3),
         ha='center', va='center')
g.ax_joint.text(0.7 * max_val, 0.4 * max_val, f'{between_lines} facilities', 
         fontsize=12, bbox=dict(boxstyle='round', facecolor='#A7E399', alpha=0.3),
         ha='center', va='center')
g.ax_joint.text(0.7 * max_val, 0.1 * max_val, f'{below_y_equal_half_x} facilities', 
         fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
         ha='center', va='center')

# Add shaded regions
g.ax_joint.fill_between(x_line, y_line, max_val, alpha=0.1, color='#48B3AF', label='Energy surplus region')
g.ax_joint.fill_between(x_line, y_half_line, y_line, alpha=0.1, color='#A7E399', label='50-100% energy self sufficient')
g.ax_joint.fill_between(x_line, 0, y_half_line, alpha=0.1, color='yellow', label='<50% energy self sufficient')

# Set labels for main plot
g.ax_joint.set_xlabel('Adjusted energy use intensity after removing technical inefficiency (kWh/m$^3$)', fontsize=14)
g.ax_joint.set_ylabel('Recoverable energy density (kWh/m³)', fontsize=14)
g.ax_joint.grid(True, alpha=0.3)
g.ax_joint.legend(fontsize=12)

# Set equal aspect ratio and limits
g.ax_joint.set_xlim(0, max_val)
g.ax_joint.set_ylim(0, max_val)
g.ax_joint.set_aspect('equal', adjustable='box')

# Add mean to marginal plots (without std)
x_mean = valid_data['Optimal EUI (kWh/m3)'].mean()
y_mean = valid_data['Recoverable energy density (kWh/m3)'].mean()

# Add mean lines to marginal plots (using dark gray)
g.ax_marg_x.axvline(x_mean, color='#333333', linestyle='--', linewidth=2, label=f'Mean: {x_mean:.2f}')
g.ax_marg_y.axhline(y_mean, color='#333333', linestyle='--', linewidth=2, label=f'Mean: {y_mean:.2f}')


plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*80)
print("Energy Coverage Analysis Summary (Recoverable Energy vs Optimal EUI)")
print("="*80)
print(f"Total facilities with valid data: {total_points}")
print(f"Facilities with energy surplus (above Y=X): {above_y_equal_x} ({above_y_equal_x/total_points*100:.1f}%)")
print(f"Facilities with 50-100% energy coverage: {between_lines} ({between_lines/total_points*100:.1f}%)")
print(f"Facilities with <50% energy coverage: {below_y_equal_half_x} ({below_y_equal_half_x/total_points*100:.1f}%)")

# Additional analysis: Calculate energy coverage ratio
energy_coverage_ratio = valid_data['Recoverable energy density (kWh/m3)'] / valid_data['Optimal EUI (kWh/m3)']
print(f"\nEnergy Coverage Statistics:")
print(f"Average energy coverage ratio: {energy_coverage_ratio.mean():.3f}")
print(f"Median energy coverage ratio: {energy_coverage_ratio.median():.3f}")
print(f"Minimum energy coverage ratio: {energy_coverage_ratio.min():.3f}")
print(f"Maximum energy coverage ratio: {energy_coverage_ratio.max():.3f}")

print(f"\nDistribution Statistics:")
print(f"Optimal EUI - Mean: {x_mean:.3f}")
print(f"Recoverable Energy - Mean: {y_mean:.3f}")










# Filter data where Selected_Process is not 'Current'
filtered_df = df[(df['Selected_Process'] != 'Current') & (df['Energy gap'] > 0)]

# Scatter plot for Recoverable energy density vs Optimal EUI (only non-Current processes)
plt.figure(figsize=(12, 10))

# Extract data from filtered dataframe
x_data = filtered_df['Optimal EUI after selection (kWh/m3)']
y_data = filtered_df['Recoverable energy density after selection (kWh/m3)']

# Create scatter plot with process type coloring
process_types = ['CEPT', 'HRAS']
process_names = {'CEPT': 'Pathway I', 'HRAS': 'Pathway II'}
colors = ['#476EAE', '#48B3AF']
markers = ['o', 's']

for i, process in enumerate(process_types):
    process_data = filtered_df[filtered_df['Selected_Process'] == process]
    if len(process_data) > 0:
        x_process = process_data['Optimal EUI after selection (kWh/m3)']
        y_process = process_data['Recoverable energy density after selection (kWh/m3)']
        plt.scatter(x_process, y_process, alpha=0.7, color=colors[i], marker=markers[i], 
                   s=60, label=f'{process_names[process]}', edgecolors='white', linewidth=0.5)

# Add y=x line (Energy production from wastewater treatment)
max_val = max(x_data.max(), y_data.max())
x_line = np.linspace(0, max_val, 100)
y_line = x_line
plt.plot(x_line, y_line, 'k--', linewidth=2, label='Energy production from wastewater treatment (Y=X)')

# Add y=0.5x line (50% energy self-sufficient)
y_half_line = 0.5 * x_line
plt.plot(x_line, y_half_line, 'k:', linewidth=2, label='50% energy self-sufficient (Y=0.5X)')

# Count points in each region for each process type
region_counts = {}

for process in process_types:
    process_data = filtered_df[filtered_df['Selected_Process'] == process]
    if len(process_data) > 0:
        above_y_equal_x = len(process_data[process_data['Recoverable energy density after selection (kWh/m3)'] > 
                                       process_data['Optimal EUI after selection (kWh/m3)']])
        between_lines = len(process_data[(process_data['Recoverable energy density after selection (kWh/m3)'] <= 
                                     process_data['Optimal EUI after selection (kWh/m3)']) & 
                                    (process_data['Recoverable energy density after selection (kWh/m3)'] > 
                                     0.5 * process_data['Optimal EUI after selection (kWh/m3)'])])
        below_y_equal_half_x = len(process_data[process_data['Recoverable energy density after selection (kWh/m3)'] <= 
                                         0.5 * process_data['Optimal EUI after selection (kWh/m3)']])
        
        region_counts[process] = {
            'above': above_y_equal_x,
            'between': between_lines,
            'below': below_y_equal_half_x
        }

# Calculate total counts
total_points = len(filtered_df)
total_above = sum(region_counts[process]['above'] for process in process_types if process in region_counts)
total_between = sum(region_counts[process]['between'] for process in process_types if process in region_counts)
total_below = sum(region_counts[process]['below'] for process in process_types if process in region_counts)

# Add text annotations with counts for each region and process type
# Above Y=X region
above_text = f"Total: {total_above}\n"
for process in process_types:
    if process in region_counts:
        count = region_counts[process]['above']
        above_text += f"{process_names[process]}: {count}\n"
plt.text(0.7 * max_val, 0.85 * max_val, above_text, 
         fontsize=14, bbox=dict(boxstyle='round', facecolor='#476EAE', alpha=0.2),
         ha='center', va='center')

# Between Y=X and Y=0.5X region
between_text = f"Total: {total_between}\n"
for process in process_types:
    if process in region_counts:
        count = region_counts[process]['between']
        between_text += f"{process_names[process]}: {count}\n"
plt.text(0.7 * max_val, 0.5 * max_val, between_text, 
         fontsize=14, bbox=dict(boxstyle='round', facecolor='#48B3AF', alpha=0.2),
         ha='center', va='center')

# Below Y=0.5X region
below_text = f"Total: {total_below}\n"
for process in process_types:
    if process in region_counts:
        count = region_counts[process]['below']
        below_text += f"{process_names[process]}: {count}\n"
plt.text(0.7 * max_val, 0.15 * max_val, below_text, 
         fontsize=14, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.2),
         ha='center', va='center')

# Add total sustainable scenarios count on the plot
plt.text(0.05 * max_val, 0.95 * max_val, f'Total facilities capable\nof sustainable pathways: {total_points}\n(excluded facilities achieved energy positive under baseline pathway', 
         fontsize=14, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         ha='left', va='top', transform=plt.gca().transAxes)

# Add shaded regions
plt.fill_between(x_line, y_line, max_val, alpha=0.1, color='#476EAE', label='Energy surplus region')
plt.fill_between(x_line, y_half_line, y_line, alpha=0.1, color='#48B3AF', label='50-100% self-sufficient')
plt.fill_between(x_line, 0, y_half_line, alpha=0.1, color='yellow', label='<50% self-sufficient')

# Set labels and title
plt.xlabel('Energy use intensity of sustainable pathways (kWh/m³)', fontsize=16)
plt.ylabel('Recoverable energy density of sustainable pathways (kWh/m³)', fontsize=16)

# Add grid and legend
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Set equal aspect ratio and limits
plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.gca().set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*80)
print("Energy Self-Sufficiency Analysis Summary (Sustainable Scenarios Only)")
print("="*80)
print(f"Total facilities with sustainable scenarios: {total_points}")

for process in process_types:
    process_data = filtered_df[filtered_df['Selected_Process'] == process]
    if len(process_data) > 0:
        process_total = len(process_data)
        above = region_counts[process]['above']
        between = region_counts[process]['between']
        below = region_counts[process]['below']
        
        print(f"\n{process_names[process]}:")
        print(f"  Total facilities: {process_total} ({process_total/total_points*100:.1f}%)")
        print(f"  Energy surplus (above Y=X): {above} ({above/process_total*100:.1f}%)")
        print(f"  50-100% self-sufficient: {between} ({between/process_total*100:.1f}%)")
        print(f"  <50% self-sufficient: {below} ({below/process_total*100:.1f}%)")

print(f"\nOverall Summary:")
print(f"Facilities with energy surplus (above Y=X): {total_above} ({total_above/total_points*100:.1f}%)")
print(f"Facilities 50-100% self-sufficient: {total_between} ({total_between/total_points*100:.1f}%)")
print(f"Facilities <50% self-sufficient: {total_below} ({total_below/total_points*100:.1f}%)")

# Additional info about the filtering
original_total = len(df)
current_count = len(df[df['Selected_Process'] == 'Current'])
print(f"\nFiltering Information:")
print(f"Original total facilities: {original_total}")
print(f"Facilities with 'Current' process: {current_count}")
print(f"Facilities with sustainable scenarios: {total_points}")






