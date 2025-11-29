# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 11:42:29 2025

@author: zz1405
"""

import pandas as pd
import numpy as np
from itertools import product

def perform_sensitivity_analysis(data_path):
    """
    对污水处理厂进行敏感性分析
    参数变化范围：
    - hras_EUI(0.162) ±10%
    - r_aeration(0.6) ±10%
    - c_solid(0.2) ±10%
    - sludge_factor(1.0) ±50%
    """
    # 读取数据
    data = pd.read_csv(data_path)
    
    # 基础参数值
    base_params = {
        'hras_EUI': 0.162,
        'r_aeration': 0.6,
        'c_solid': 0.2,
        'sludge_factor': 1.0
    }
    
    # 参数变化范围
    param_variations = {
        'hras_EUI': [base_params['hras_EUI'] * 0.9, base_params['hras_EUI'], base_params['hras_EUI'] * 1.1],
        'r_aeration': [0.54, 0.6, 0.66],
        'c_solid': [0.18, 0.2, 0.22],
        'sludge_factor': [0.5, 1.0, 1.5]
    }
    
    # 生成所有参数组合
    param_combinations = list(product(*param_variations.values()))
    param_names = list(param_variations.keys())
    
    # 对每个参数组合进行计算
    results = []
    for combo in param_combinations:
        params = dict(zip(param_names, combo))
        
        # 计算能耗（简化公式）
        electricity = (
            data['Average flow (MGD)'] * 1381675 * params['hras_EUI'] +
            (data['Sludge_TSS (g/a)'] * params['sludge_factor'] / 1e6) * 20
        )
        
        heat = (
            (data['Sludge_TSS (g/a)'] * params['sludge_factor'] / params['c_solid'] / 1e3) / 365 * 
            4.18 * data['annual_hdd'] / 3600
        )
        
        total_energy = electricity + heat
        
        # 存储汇总结果
        results.append({
            'hras_EUI': params['hras_EUI'],
            'r_aeration': params['r_aeration'],
            'c_solid': params['c_solid'],
            'sludge_factor': params['sludge_factor'],
            'Avg_Electricity_consumption': electricity.mean(),
            'Avg_Heat_consumption': heat.mean(),
            'Avg_Total_energy_consumption': total_energy.mean(),
            'Plant_count': len(data)
        })
    
    return pd.DataFrame(results)

def analyze_sensitivity(sensitivity_results):
    """分析敏感性并返回汇总结果"""
    # 基础情况
    base_case = sensitivity_results[
        (sensitivity_results['hras_EUI'] == 0.162) &
        (sensitivity_results['r_aeration'] == 0.6) &
        (sensitivity_results['c_solid'] == 0.2) &
        (sensitivity_results['sludge_factor'] == 1.0)
    ]
    
    base_energy = base_case['Avg_Total_energy_consumption'].iloc[0]
    
    # 分析每个参数的敏感性
    sensitivity_summary = []
    
    for param in ['hras_EUI', 'r_aeration', 'c_solid', 'sludge_factor']:
        for value in sensitivity_results[param].unique():
            if value == base_case[param].iloc[0]:
                continue
                
            varied_case = sensitivity_results[
                (sensitivity_results[param] == value) &
                (sensitivity_results[[p for p in ['hras_EUI', 'r_aeration', 'c_solid', 'sludge_factor'] 
                                  if p != param]] == base_case[[p for p in ['hras_EUI', 'r_aeration', 'c_solid', 'sludge_factor'] 
                                                              if p != param]].iloc[0]).all(axis=1)
            ]
            
            if len(varied_case) > 0:
                varied_energy = varied_case['Avg_Total_energy_consumption'].iloc[0]
                change_pct = (varied_energy - base_energy) / base_energy * 100
                
                sensitivity_summary.append({
                    'Parameter': param,
                    'Value': value,
                    'Energy_change_percent': change_pct,
                    'Sensitivity_level': '高' if abs(change_pct) > 10 else ('中' if abs(change_pct) > 5 else '低')
                })
    
    return pd.DataFrame(sensitivity_summary)

# 主程序
if __name__ == "__main__":
    data_path = r"C:\Users\zz1405\OneDrive - Princeton University\Documents\Work 2_CN energy\Submission to One Earth\Github\Assessment framework\New pathways energy recovery\HRAS\HRAS_output.csv"
    
    # 进行敏感性分析
    sensitivity_results = perform_sensitivity_analysis(data_path)
    sensitivity_summary = analyze_sensitivity(sensitivity_results)
    
    # 保存结果
    sensitivity_results.to_csv('sensitivity_results.csv', index=False)
    sensitivity_summary.to_csv('sensitivity_summary.csv', index=False)
    
    # 打印结果
    print("敏感性分析结果:")
    print("=" * 40)
    for param in ['hras_EUI', 'r_aeration', 'c_solid', 'sludge_factor']:
        param_data = sensitivity_summary[sensitivity_summary['Parameter'] == param]
        print(f"\n{param}:")
        for _, row in param_data.iterrows():
            direction = "增加" if row['Energy_change_percent'] > 0 else "减少"
            print(f"  {row['Value']}: {direction} {abs(row['Energy_change_percent']):.1f}%")