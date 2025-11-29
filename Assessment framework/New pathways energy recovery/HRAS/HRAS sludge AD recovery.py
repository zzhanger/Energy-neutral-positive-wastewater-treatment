# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 17:48:22 2025

@author: zz1405
"""

import pandas as pd

def biogas_production(Sludge_COD, Y_c=20, Y=0.05, E=0.8, k_d=0.03):

    # 单位转换 (g/a → kg/a)
    BOD5 = Sludge_COD * 0.4 / 1000  # 假设COD中40%为可生物降解部分
    BODL = BOD5 / 0.68  # 转换为BODL
    
    # 1. 计算细胞净产量Px (kg VSS/d)
    Px = (Y * E * BODL) / (1 + k_d * Y_c)
    
    # 2. 计算甲烷产量 (m³/d)
    methane_a = 0.35 * (E * BODL - 1.42 * Px)
    
    # 3. 年化计算
    biogas_a = methane_a / 0.6  # 甲烷占比60%
    energy_kwh_a = methane_a * 9.94  # 甲烷热值9.94 kWh/m³
    
    return {
        "Electricity recovery potential (kWh/a)_HRAS": round(energy_kwh_a * 0.35, 2),
        "Heat recovery potential (kWh/a)_HRAS": round(energy_kwh_a * 0.4, 2),
    }

# 读取原始数据
try:
    data = pd.read_csv('HRAS_output.csv')
    if 'Sludge_COD (g/a)' not in data.columns:
        raise ValueError("输入文件缺少必需列 'Sludge_COD (g/a)'")
except Exception as e:
    print(f"文件读取失败: {str(e)}")
    exit()

# 为每行计算沼气产量并直接添加到原DataFrame
for idx, row in data.iterrows():
    try:
        # 计算沼气产量
        result = biogas_production(Sludge_COD=row['Sludge_COD (g/a)'])
        
        # 将结果添加到当前行
        for col, value in result.items():
            data.at[idx, col] = value
            
    except Exception as e:
        print(f"水厂 {idx} 计算失败: {str(e)}")
        # 失败行填充NA
        for col in ['Electricity recovery potential (kWh/a)_HRAS', 'Heat recovery potential (kWh/a)_HRAS']:
            data.at[idx, col] = None


output_file = 'HRAS_output.csv'
try:
    data.to_csv(output_file, index=False)
    print(f"计算完成，结果已保存至 {output_file}")
    
    
except Exception as e:
    print(f"结果保存失败: {str(e)}")