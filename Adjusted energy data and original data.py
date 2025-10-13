# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:53:45 2025

@author: zz1405
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.family'] = 'Arial'
sns.set_style("whitegrid")

# 读取数据
df = pd.read_csv('merged_output.csv')

# 检查数据列
print("数据列名:", df.columns.tolist())
print("\n数据前5行:")
print(df[['Flow based EUI (kWh/m^3)', 'COD removal based EUI (kWh/kg COD removed)', 
          'Optimal EUI (kWh/m3)', 'Optimal EUI (kWh/kg COD removed)']].head())

# 数据预处理 - 处理缺失值
data_clean = df[['Flow based EUI (kWh/m^3)', 'COD removal based EUI (kWh/kg COD removed)', 
                'Optimal EUI (kWh/m3)', 'Optimal EUI (kWh/kg COD removed)']].dropna()

print(f"\n原始数据点数: {len(df)}")
print(f"清理后数据点数: {len(data_clean)}")

# 创建双面板图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# =========================================
# 左图：Flow based EUI 对比
# =========================================
# 绘制原始数据核密度
sns.kdeplot(data=data_clean, x='Flow based EUI (kWh/m^3)', 
            ax=ax1, label='Observed EUI\n(with technical inefficiency)', 
            fill=True, alpha=0.6, color='#b4cded', linewidth=2)

# 绘制调整后数据核密度
sns.kdeplot(data=data_clean, x='Optimal EUI (kWh/m3)', 
            ax=ax1, label='Optimal EUI\n(technical inefficiency removed)', 
            fill=True, alpha=0.6, color='#bfcc94', linewidth=2)

# 添加均值线
orig_mean_flow = data_clean['Flow based EUI (kWh/m^3)'].mean()
opt_mean_flow = data_clean['Optimal EUI (kWh/m3)'].mean()
improvement_flow = ((orig_mean_flow - opt_mean_flow) / orig_mean_flow) * 100

ax1.axvline(orig_mean_flow, color='#b4cded', linestyle='--', alpha=0.8, 
           label=f'Observed mean: {orig_mean_flow:.2f}')
ax1.axvline(opt_mean_flow, color='#bfcc94', linestyle='--', alpha=0.8, 
           label=f'Optimal mean: {opt_mean_flow:.2f}')

# 设置左图属性
ax1.set_xlabel('Flow based EUI (kWh/m³)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Density', fontsize=14, fontweight='bold')
ax1.set_title('Flow based Energy Use Intensity\n'
             f'Improvement: {improvement_flow:.1f}%', 
             fontsize=16, fontweight='bold', pad=15)
ax1.legend(fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.3)

# =========================================
# 右图：COD removal based EUI 对比
# =========================================
# 绘制原始数据核密度
sns.kdeplot(data=data_clean, x='COD removal based EUI (kWh/kg COD removed)', 
            ax=ax2, label='Observed EUI\n(with technical inefficiency)', 
            fill=True, alpha=0.6, color='#b4cded', linewidth=2)

# 绘制调整后数据核密度
sns.kdeplot(data=data_clean, x='Optimal EUI (kWh/kg COD removed)', 
            ax=ax2, label='Optimal EUI\n(technical inefficiency removed)', 
            fill=True, alpha=0.6, color='#bfcc94', linewidth=2)

# 添加均值线
orig_mean_cod = data_clean['COD removal based EUI (kWh/kg COD removed)'].mean()
opt_mean_cod = data_clean['Optimal EUI (kWh/kg COD removed)'].mean()
improvement_cod = ((orig_mean_cod - opt_mean_cod) / orig_mean_cod) * 100

ax2.axvline(orig_mean_cod, color='#b4cded', linestyle='--', alpha=1, 
           label=f'Observed mean: {orig_mean_cod:.2f}')
ax2.axvline(opt_mean_cod, color='#bfcc94', linestyle='--', alpha=1, 
           label=f'Optimal mean: {opt_mean_cod:.2f}')

# 设置右图属性
ax2.set_xlabel('COD removal based EUI (kWh/kg COD removed)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Density', fontsize=14, fontweight='bold')
ax2.set_title('COD Removal based Energy Use Intensity\n'
             f'Improvement: {improvement_cod:.1f}%', 
             fontsize=16, fontweight='bold', pad=15)
ax2.legend(fontsize=11, framealpha=0.9)
ax2.grid(True, alpha=0.3)

# 调整布局
plt.tight_layout()
plt.show()

# =========================================
# 打印统计摘要
# =========================================
print("\n" + "="*70)
print("技术无效性剔除效果统计摘要")
print("="*70)

print("\nFlow based EUI:")
print(f"  观测均值: {orig_mean_flow:.3f} kWh/m³")
print(f"  最优均值: {opt_mean_flow:.3f} kWh/m³")
print(f"  改进幅度: {improvement_flow:.1f}%")
print(f"  绝对减少: {orig_mean_flow - opt_mean_flow:.3f} kWh/m³")

print("\nCOD removal based EUI:")
print(f"  观测均值: {orig_mean_cod:.3f} kWh/kg COD")
print(f"  最优均值: {opt_mean_cod:.3f} kWh/kg COD")
print(f"  改进幅度: {improvement_cod:.1f}%")
print(f"  绝对减少: {orig_mean_cod - opt_mean_cod:.3f} kWh/kg COD")

# 计算标准差变化
print("\n分布离散度变化 (标准差):")
orig_std_flow = data_clean['Flow based EUI (kWh/m^3)'].std()
opt_std_flow = data_clean['Optimal EUI (kWh/m3)'].std()
orig_std_cod = data_clean['COD removal based EUI (kWh/kg COD removed)'].std()
opt_std_cod = data_clean['Optimal EUI (kWh/kg COD removed)'].std()

print(f"  Flow EUI 观测标准差: {orig_std_flow:.3f}")
print(f"  Flow EUI 最优标准差: {opt_std_flow:.3f}")
print(f"  COD EUI 观测标准差: {orig_std_cod:.3f}")
print(f"  COD EUI 最优标准差: {opt_std_cod:.3f}")

# 计算变异系数变化
print("\n变异系数变化 (CV%):")
cv_orig_flow = (orig_std_flow / orig_mean_flow) * 100
cv_opt_flow = (opt_std_flow / opt_mean_flow) * 100
cv_orig_cod = (orig_std_cod / orig_mean_cod) * 100
cv_opt_cod = (opt_std_cod / opt_mean_cod) * 100

print(f"  Flow EUI 观测CV: {cv_orig_flow:.1f}%")
print(f"  Flow EUI 最优CV: {cv_opt_flow:.1f}%")
print(f"  COD EUI 观测CV: {cv_orig_cod:.1f}%")
print(f"  COD EUI 最优CV: {cv_opt_cod:.1f}%")