# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 18:19:23 2025

@author: zz1405
"""

import pandas as pd
import numpy as np

# 读取数据
path = r"C:\Users\zz1405\OneDrive - Princeton University\Documents\Work 2_CN energy\Submission to One Earth\Github\Dataset processing\Cleaned final dataset.csv"
data = pd.read_csv(path)

def calculate_energy_consumption(data, sludge_energy_factor=20, solid_content=0.2):
    """
    计算总能耗（电力+热能）
    参数:
        sludge_energy_factor: 污泥处理能耗系数 (kWh/ton DS)
        solid_content: 污泥固体含量比例
    """
    # 电力消耗 = 水处理能耗 + 污泥处理能耗
    elec_con = (data['Average flow (MGD)'] * 1381674 * data['EUI (kWh/m3)'] + 
               data['Sludge yield (tone)'] * solid_content * sludge_energy_factor)
    
    # 热能消耗
    heat_con = data['Sludge yield (tone)'] * 1000 / 365 * 4.18 * data['Annual HDD'] / 3600
    
    # 总能耗
    total_energy = elec_con + heat_con
    return elec_con, heat_con, total_energy

# 基准情景计算
data['Base_Electricity'], data['Base_Heat'], data['Base_Total'] = calculate_energy_consumption(data)

# 情景1：污泥处理能耗系数从20降到5 kWh/ton DS
data['Scenario1_Electricity'], data['Scenario1_Heat'], data['Scenario1_Total'] = calculate_energy_consumption(data, sludge_energy_factor=5)

# 情景2：固体含量+10% (0.22)
data['Scenario2a_Electricity'], data['Scenario2a_Heat'], data['Scenario2a_Total'] = calculate_energy_consumption(data, solid_content=0.22)

# 情景3：固体含量-10% (0.18)
data['Scenario2b_Electricity'], data['Scenario2b_Heat'], data['Scenario2b_Total'] = calculate_energy_consumption(data, solid_content=0.18)

# 计算差异
data['Diff_S1_Total(%)'] = (data['Scenario1_Total'] - data['Base_Total']) / data['Base_Total'] * 100
data['Diff_S2a_Total(%)'] = (data['Scenario2a_Total'] - data['Base_Total']) / data['Base_Total'] * 100
data['Diff_S2b_Total(%)'] = (data['Scenario2b_Total'] - data['Base_Total']) / data['Base_Total'] * 100

# 汇总统计
summary = pd.DataFrame({
    'Scenario': ['Base', 'Sludge Energy=5', 'Solid+10%', 'Solid-10%'],
    'Avg_Electricity(kWh/a)': [
        data['Base_Electricity'].mean(),
        data['Scenario1_Electricity'].mean(),
        data['Scenario2a_Electricity'].mean(),
        data['Scenario2b_Electricity'].mean()
    ],
    'Avg_Heat(kWh/a)': [
        data['Base_Heat'].mean(),
        data['Scenario1_Heat'].mean(),
        data['Scenario2a_Heat'].mean(),
        data['Scenario2b_Heat'].mean()
    ],
    'Avg_Total(kWh/a)': [
        data['Base_Total'].mean(),
        data['Scenario1_Total'].mean(),
        data['Scenario2a_Total'].mean(),
        data['Scenario2b_Total'].mean()
    ],
    'Total_Change(%)': [
        0,
        (data['Scenario1_Total'].mean() - data['Base_Total'].mean()) / data['Base_Total'].mean() * 100,
        (data['Scenario2a_Total'].mean() - data['Base_Total'].mean()) / data['Base_Total'].mean() * 100,
        (data['Scenario2b_Total'].mean() - data['Base_Total'].mean()) / data['Base_Total'].mean() * 100
    ]
})


print("\n汇总统计:")
print(summary)