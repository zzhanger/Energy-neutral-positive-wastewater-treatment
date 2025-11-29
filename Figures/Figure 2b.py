# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 10:34:18 2025

@author: zz1405
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams['font.family'] = 'Arial'
sns.set_style("whitegrid")
plt.rcParams['grid.color'] = 'lightgray'
plt.rcParams['grid.alpha'] = 0.3


path = r"C:\Users\zz1405\OneDrive - Princeton University\Documents\Work 2_CN energy\Submission to One Earth\Github\Dataset processing\Cleaned final dataset.csv"
df = pd.read_csv(path)

# Check data
print("Column names:", df.columns.tolist())
print("\nFirst 5 rows of data:")
print(df[['Flow based EUI (kWh/m^3)', 'Pollutant removal based EUI (kWh/kg COD removed)']].head())

# Data preprocessing
data_clean = df[['Flow based EUI (kWh/m^3)', 'Pollutant removal based EUI (kWh/kg COD removed)']].dropna()
print(f"\nOriginal data points: {len(df)}")
print(f"Cleaned data points: {len(data_clean)}")

# Create joint distribution plot
joint_plot = sns.jointplot(
    x='Flow based EUI (kWh/m^3)', 
    y='Pollutant removal based EUI (kWh/kg COD removed)',
    data=data_clean,
    kind='kde',
    height=4,          
    ratio=5,            
    cmap='viridis',     
    fill=True,
    alpha=0.7,
    marginal_kws=dict(color='#440154', fill=True, alpha=0.7)
)


joint_plot.fig.set_size_inches(12, 4)


scatter = joint_plot.ax_joint.scatter(
    x='Flow based EUI (kWh/m^3)', 
    y='Pollutant removal based EUI (kWh/kg COD removed)',
    data=data_clean,
    facecolor='none',   
    alpha=0.7,
    s=8,
    edgecolor='#69626d',
    linewidth=0.5,
    label='Facility level energy use intensity'
)


joint_plot.ax_joint.set_xscale('log')
joint_plot.ax_joint.set_yscale('log')


joint_plot.ax_joint.set_xlabel('Flow based EUI (kWh/m$^3$)', fontsize=14, fontweight='bold')
joint_plot.ax_joint.set_ylabel('COD removal based EUI\n(kWh/kg COD removed)', fontsize=14, fontweight='bold')


from matplotlib.ticker import ScalarFormatter
for axis in [joint_plot.ax_joint.xaxis, joint_plot.ax_joint.yaxis]:
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_formatter(ScalarFormatter())


joint_plot.ax_joint.set_xlim(0.01, 1) 


joint_plot.ax_joint.set_ylim(0.01, 100) 


legend = joint_plot.ax_joint.legend(
    frameon=False,     
    loc='best',        
    fontsize=14,        
    handletextpad=0.5, 
    scatterpoints=1     
)


joint_plot.figure.subplots_adjust(wspace=0.05, hspace=0.05)


joint_plot.ax_marg_x.set_xscale('log')
joint_plot.ax_marg_y.set_yscale('log')

plt.show()

# Print statistical summary (including statistics after logarithmic transformation)
print("\nStatistical Summary:")
print("=" * 50)
print("EUI (kWh/m³):")
print(f"  Mean: {data_clean['Flow based EUI (kWh/m^3)'].mean():.3f}")
print(f"  Standard Deviation: {data_clean['Flow based EUI (kWh/m^3)'].std():.3f}")
print(f"  Minimum: {data_clean['Flow based EUI (kWh/m^3)'].min():.3f}")
print(f"  Maximum: {data_clean['Flow based EUI (kWh/m^3)'].max():.3f}")

print("\nCOD removal based EUI (kWh/kg COD removed):")
print(f"  Mean: {data_clean['Pollutant removal based EUI (kWh/kg COD removed)'].mean():.3f}")
print(f"  Standard Deviation: {data_clean['Pollutant removal based EUI (kWh/kg COD removed)'].std():.3f}")
print(f"  Minimum: {data_clean['Pollutant removal based EUI (kWh/kg COD removed)'].min():.3f}")
print(f"  Maximum: {data_clean['Pollutant removal based EUI (kWh/kg COD removed)'].max():.3f}")

# Statistics after logarithmic transformation
print("\nStatistics After Logarithmic Transformation:")
print("=" * 50)
print("log(EUI) (log(kWh/m³)):")
log_eui = np.log(data_clean['Flow based EUI (kWh/m^3)'])
print(f"  Mean: {log_eui.mean():.3f}")
print(f"  Standard Deviation: {log_eui.std():.3f}")
print(f"  Minimum: {log_eui.min():.3f}")
print(f"  Maximum: {log_eui.max():.3f}")

print("\nlog(Pollutant removal based EUI) (log(kWh/kg COD removed)):")
log_cod = np.log(data_clean['Pollutant removal based EUI (kWh/kg COD removed)'])
print(f"  Mean: {log_cod.mean():.3f}")
print(f"  Standard Deviation: {log_cod.std():.3f}")
print(f"  Minimum: {log_cod.min():.3f}")
print(f"  Maximum: {log_cod.max():.3f}")