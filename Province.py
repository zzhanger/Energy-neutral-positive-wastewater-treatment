# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 09:42:37 2025

@author: zz1405
"""
import pandas as pd
import numpy as np

# 英文省份名称到拼音缩写的映射
province_mapping = {
    'Beijing': 'BJ',
    'Hebei': 'HEB', 
    'Tianjin': 'TJ',
    'Shandong': 'SD',
    'Henan': 'HEN',
    'Jilin': 'JL',
    'Shanxi': 'SX',
    'Guangdong': 'GD',
    'Inner Mongolia': 'IM',
    'Liaoning': 'LN',
    'Shaanxi': 'SNX',
    'Heilongjiang': 'HLJ',
    'Jiangsu': 'JS',
    'Shanghai': 'SH',
    'Zhejiang': 'ZJ',
    'Anhui': 'AH',
    'Hubei': 'HUB',
    'Jiangxi': 'JX',
    'Hunan': 'HUN',
    'Guangxi': 'GX',
    'Hainan': 'HAN',
    'Guizhou': 'GZ',
    'Chongqing': 'CQ',
    'Sichuan': 'SC',
    'Yunnan': 'YN',
    'Xizang': 'XZ',
    'Qinghai': 'QH',
    'Xinjiang': 'XJ',
    'Gansu': 'GS',
    'Fujian': 'FJ'
}

# 读取CSV文件
df = pd.read_csv('merged_output.csv')  # 请替换为您的实际文件名

print("=== 数据检查 ===")
print(f"数据总行数: {len(df)}")
print(f"数据列名: {list(df.columns)}")

# 检查Province列的数据类型和缺失值
print(f"\n=== Province列信息 ===")
print(f"数据类型: {df['Province'].dtype}")
print(f"缺失值数量: {df['Province'].isnull().sum()}")

# 处理缺失值并检查唯一值
df_clean = df.dropna(subset=['Province']).copy()
unique_provinces = df_clean['Province'].unique()

print(f"\n=== Province列中的省份名称 ({len(unique_provinces)}个) ===")
# 安全排序：先转换为字符串再排序
for province in sorted([str(p) for p in unique_provinces]):
    print(f"  {province}")

# 执行替换
print(f"\n=== 执行省份名称到缩写的替换 ===")
# 创建副本用于处理
df_processed = df.copy()

# 将Province列转换为字符串类型并去除可能的空格
df_processed['Province'] = df_processed['Province'].astype(str).str.strip()

# 执行映射
df_processed['Province_Abbr'] = df_processed['Province'].map(province_mapping)

# 对于无法映射的，保留原值
df_processed['Province_Abbr'] = df_processed['Province_Abbr'].fillna(df_processed['Province'])

print(f"替换后的省份缩写分布:")
print(df_processed['Province_Abbr'].value_counts(dropna=False))

# 统计各省效率比例
if 'Efficiency_Group' in df_processed.columns:
    print(f"\n=== 各省效率统计 ===")
    
    # 确保Efficiency_Group没有缺失值
    df_for_stats = df_processed.dropna(subset=['Efficiency_Group']).copy()
    
    # 计算统计信息
    efficiency_stats = df_for_stats.groupby('Province_Abbr').agg(
        Efficient_Rate=('Efficiency_Group', lambda x: round((x == 'Efficient').sum() / len(x) * 100, 2)),
        Inefficient_Rate=('Efficiency_Group', lambda x: round((x == 'Inefficient').sum() / len(x) * 100, 2)),
        Total_Count=('Efficiency_Group', 'count')
    ).reset_index()
    
    # 重命名列
    efficiency_stats.columns = ['Province', 'Efficient_Rate', 'Inefficient_Rate', 'Total_Count']
    
    # 按省份字母顺序排序
    efficiency_stats = efficiency_stats.sort_values('Province')
    
    # 输出结果到CSV文件
    output_df = df_processed.copy()
    output_df['Province'] = output_df['Province_Abbr']  # 用缩写替换原Province列
    output_df = output_df.drop('Province_Abbr', axis=1)  # 删除临时列
    
    output_df.to_csv('provinces_abbreviated.csv', index=False, encoding='utf-8-sig')
    efficiency_stats.to_csv('efficiency_statistics.csv', index=False, encoding='utf-8-sig')
    
    print("处理完成！")
    print(f"\n各省效率比例 (按省份排序):")
    print(efficiency_stats.to_string(index=False))
    
    print(f"\n已生成文件:")
    print("1. provinces_abbreviated.csv - 省份名称替换为缩写后的完整数据")
    print("2. efficiency_statistics.csv - 各省效率统计")
    
    # 显示格式说明
    print(f"\n输出格式:")
    print("第一列: Province (省份缩写)")
    print("第二列: Efficient_Rate (Efficient比例%)") 
    print("第三列: Inefficient_Rate (Inefficient比例%)")
    print("第四列: Total_Count (总样本数)")
    
else:
    print("无法进行效率统计，因为缺少 Efficiency_Group 列")