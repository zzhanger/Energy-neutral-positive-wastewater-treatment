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

# 设置字体和样式
plt.rcParams['font.family'] = 'Arial'
sns.set_style("whitegrid")
plt.rcParams['grid.color'] = 'lightgray'
plt.rcParams['grid.alpha'] = 0.3

# 读取数据
df = pd.read_csv('merged_output.csv')

# 检查数据
print("数据列名:", df.columns.tolist())
print("\n数据前5行:")
print(df[['Flow based EUI (kWh/m^3)', 'COD removal based EUI (kWh/kg COD removed)']].head())

# 数据预处理
data_clean = df[['Flow based EUI (kWh/m^3)', 'COD removal based EUI (kWh/kg COD removed)']].dropna()
print(f"\n原始数据点数: {len(df)}")
print(f"清理后数据点数: {len(data_clean)}")

# 创建联合分布图
joint_plot = sns.jointplot(
    x='Flow based EUI (kWh/m^3)', 
    y='COD removal based EUI (kWh/kg COD removed)',
    data=data_clean,
    kind='kde',
    height=4,           # 较低的高度
    ratio=5,            # 很大的比例使图形变长
    cmap='viridis',     # Viridis渐变色
    fill=True,
    alpha=1,
    marginal_kws=dict(color='#440154', fill=True, alpha=0.7)
)

# 手动调整图形大小为长方形
joint_plot.fig.set_size_inches(12, 4)

# 叠加散点图
scatter = joint_plot.ax_joint.scatter(
    x='Flow based EUI (kWh/m^3)', 
    y='COD removal based EUI (kWh/kg COD removed)',
    data=data_clean,
    facecolor='none',   # 填充颜色设为无
    alpha=0.7,
    s=5,
    edgecolor='grey',
    linewidth=0.5,
    label='Facility level energy use intensity'
)
# 设置对数坐标轴
joint_plot.ax_joint.set_xscale('log')
joint_plot.ax_joint.set_yscale('log')

# 设置坐标轴范围和标签
joint_plot.ax_joint.set_xlabel('Flow based EUI (kWh/m$^3$)', fontsize=14, fontweight='bold')
joint_plot.ax_joint.set_ylabel('COD removal based EUI\n(kWh/kg COD removed)', fontsize=14, fontweight='bold')

# 设置对数坐标轴的刻度格式
from matplotlib.ticker import ScalarFormatter
for axis in [joint_plot.ax_joint.xaxis, joint_plot.ax_joint.yaxis]:
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_formatter(ScalarFormatter())

# 设置x轴范围
joint_plot.ax_joint.set_xlim(0.01, 1)  # 最小值和最大值

# 设置y轴范围  
joint_plot.ax_joint.set_ylim(0.01, 100)  # 最小值和最大值

# 添加图例
joint_plot.ax_joint.legend(loc='upper right', framealpha=0.9)

# 移除间隙
joint_plot.figure.subplots_adjust(wspace=0.05, hspace=0.05)

# 设置边距图的坐标轴也为对数坐标
joint_plot.ax_marg_x.set_xscale('log')
joint_plot.ax_marg_y.set_yscale('log')

plt.show()

# 打印统计摘要（包括对数变换后的统计）
print("\n统计摘要:")
print("=" * 50)
print("EUI (kWh/m3):")
print(f"  均值: {data_clean['Flow based EUI (kWh/m^3)'].mean():.3f}")
print(f"  标准差: {data_clean['Flow based EUI (kWh/m^3)'].std():.3f}")
print(f"  最小值: {data_clean['Flow based EUI (kWh/m^3)'].min():.3f}")
print(f"  最大值: {data_clean['Flow based EUI (kWh/m^3)'].max():.3f}")

print("\nCOD removal based EUI (kWh/kg COD removed):")
print(f"  均值: {data_clean['COD removal based EUI (kWh/kg COD removed)'].mean():.3f}")
print(f"  标准差: {data_clean['COD removal based EUI (kWh/kg COD removed)'].std():.3f}")
print(f"  最小值: {data_clean['COD removal based EUI (kWh/kg COD removed)'].min():.3f}")
print(f"  最大值: {data_clean['COD removal based EUI (kWh/kg COD removed)'].max():.3f}")

# 对数变换后的统计
print("\n对数变换后统计:")
print("=" * 50)
print("log(EUI) (log(kWh/m3)):")
log_eui = np.log(data_clean['Flow based EUI (kWh/m^3)'])
print(f"  均值: {log_eui.mean():.3f}")
print(f"  标准差: {log_eui.std():.3f}")
print(f"  最小值: {log_eui.min():.3f}")
print(f"  最大值: {log_eui.max():.3f}")

print("\nlog(COD removal based EUI) (log(kWh/kg COD removed)):")
log_cod = np.log(data_clean['COD removal based EUI (kWh/kg COD removed)'])
print(f"  均值: {log_cod.mean():.3f}")
print(f"  标准差: {log_cod.std():.3f}")
print(f"  最小值: {log_cod.min():.3f}")
print(f"  最大值: {log_cod.max():.3f}")