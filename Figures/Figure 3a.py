# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 15:22:00 2025

@author: zz1405
"""

import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
import numpy as np
from scipy.spatial.distance import pdist, squareform


province_abbr = {
    'Beijing': 'BJ',
    'Tianjin': 'TJ',
    'Hebei': 'HEB',
    'Shanxi': 'SX',
    'Inner': 'IM',
    'Liaoning': 'LN',
    'Jilin': 'JL',
    'Heilongjiang': 'HLJ',
    'Shanghai': 'SH',
    'Jiangsu': 'JS',
    'Zhejiang': 'ZJ',
    'Anhui': 'AH',
    'Fujian': 'FJ',
    'Jiangxi': 'JX',
    'Shandong': 'SD',
    'Henan': 'HEN',
    'Hubei': 'HUB',
    'Hunan': 'HUN',
    'Guangdong': 'GD',
    'Guangxi': 'GX',
    'Hainan': 'HAN',
    'Chongqing': 'CQ',
    'Sichuan': 'SC',
    'Guizhou': 'GZ',
    'Yunnan': 'YN',
    'Xizang': 'XZ',
    'Shaanxi': 'SNX',
    'Gansu': 'GS',
    'Qinghai': 'QH',
    'Ningxia': 'NX',
    'Xinjiang': 'XJ'
}


df = pd.read_csv('Data for figure 3a.csv')


df['Province_abbr'] = df['Province'].map(province_abbr)


def standardize(series):
    return (series - series.mean()) / series.std()

df['mean_m3_z'] = standardize(df['mean_m3'])
df['mean_kg_z'] = standardize(df['mean_kg'])
df['ci_95_low_m3_z'] = standardize(df['ci_95_low_m3'])
df['ci_95_high_m3_z'] = standardize(df['ci_95_high_m3'])
df['ci_95_low_kg_z'] = standardize(df['ci_95_low_kg'])
df['ci_95_high_kg_z'] = standardize(df['ci_95_high_kg'])


plt.figure(figsize=(20, 10))


texts = []


for _, row in df.iterrows():
    x_err = [[abs(row['mean_m3_z'] - row['ci_95_low_m3_z'])], 
             [abs(row['ci_95_high_m3_z'] - row['mean_m3_z'])]]
    y_err = [[abs(row['mean_kg_z'] - row['ci_95_low_kg_z'])], 
             [abs(row['ci_95_high_kg_z'] - row['mean_kg_z'])]]
    
    plt.errorbar(
        x=row['mean_m3_z'],
        y=row['mean_kg_z'],
        xerr=x_err,
        yerr=y_err,
        fmt='none', 
        ecolor='#3182bd', 
        elinewidth=1.0,
        capsize=4,
        alpha=0.6
    )

# 创建颜色映射（表示COD浓度）
norm = plt.Normalize(vmin=df['mean_COD'].min(), vmax=df['mean_COD'].max())
cmap = plt.get_cmap('viridis')

# 绘制均值点（颜色=COD浓度）
scatter = plt.scatter(
    x=df['mean_m3_z'],
    y=df['mean_kg_z'],
    c=df['mean_COD'],
    cmap=cmap,
    norm=norm,
    edgecolors='w', 
    s=100,
    linewidths=0.8,
    alpha=0.8,
    zorder=10
)

# 添加颜色条（COD浓度）
cbar = plt.colorbar(scatter, shrink=0.8)
cbar.set_label('Mean influent COD concentration (mg/L)', fontsize=22)

# 添加省份缩写标签
for _, row in df.iterrows():
    texts.append(plt.text(
        x=row['mean_m3_z'],
        y=row['mean_kg_z'],
        s=row['Province_abbr'],
        fontsize=16,
        ha='center',
        va='center',
        fontweight='bold'
    ))


adjust_text(texts, arrowprops=dict(
    arrowstyle='-',
    color='gray',
    lw=0.5,
    alpha=0.5
))

# 计算平均成对空间距离
points = df[['mean_m3_z', 'mean_kg_z']].values
pairwise_distances = pdist(points)
mean_distance = np.mean(pairwise_distances)
std_distance = np.std(pairwise_distances)

# 添加距离信息到图中
distance_text = f'Mean pairwise distance: {mean_distance:.2f} ± {std_distance:.2f}'
plt.text(0.02, 0.98, distance_text, 
         transform=plt.gca().transAxes,
         fontsize=22, 
         bbox=dict(facecolor='#f0f0f0', alpha=0.7),
         verticalalignment='top')

# 添加参考线
plt.axvline(0, color='#636363', linestyle='--', linewidth=1.5, alpha=0.7)
plt.axhline(0, color='#636363', linestyle='--', linewidth=1.5, alpha=0.7)

# 添加象限标注
plt.text(1, 2, 'Q1: High flow based EUI\nHigh COD removal based EUI', 
         fontsize=22, bbox=dict(facecolor='white', alpha=0.7))
plt.text(-2.8, 2, 'Q2: Low flow based EUI\nHigh COD removal based EUI', 
         fontsize=22, bbox=dict(facecolor='white', alpha=0.7))
plt.text(-2.8, -4, 'Q3: Low flow based EUI\nLow COD removal based EUI', 
         fontsize=22, bbox=dict(facecolor='white', alpha=0.7))
plt.text(1, -4, 'Q4: High flow based EUI\nLow COD removaL based EUI', 
         fontsize=22, bbox=dict(facecolor='white', alpha=0.7))

# 坐标轴标签
plt.xlabel('Standardized flow based EUI', fontsize=24)
plt.ylabel('Standardized COD removal based EUI', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()