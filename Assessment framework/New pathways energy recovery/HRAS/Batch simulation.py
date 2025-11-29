# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 16:05:58 2025

@author: zz1405
"""

from HRAS_model import HRASModel
import pandas as pd
import logging
from tqdm import tqdm  # 用于显示进度条

class BatchHRASProcessor:
    def __init__(self, model_type='diauxic'):
        """初始化批量处理器"""
        self.model = HRASModel(model_type)
        self.batch_results = []
        logging.basicConfig(filename='batch_HRAS.log', level=logging.INFO)
        
    def process_csv(self, input_file, output_file=None):
        """
        处理包含多个水厂数据的CSV文件，将计算结果直接添加到原文件中
        
        参数:
            input_file: 输入CSV文件路径，应包含各水厂的进水参数和经纬度
            output_file: 可选，输出文件路径（默认覆盖原文件）
        """
        try:
            # 读取输入数据
            plant_data = pd.read_csv(input_file)
            
            # 检查必要的列是否存在
            required_columns = ['CODinf', 'BODinf', 'Average flow (MGD)', 'NH3-Ninf']
            for col in required_columns:
                if col not in plant_data.columns:
                    raise ValueError(f"输入文件缺少必要列: {col}")
            
            # 处理每个水厂
            for idx, row in tqdm(plant_data.iterrows(), total=len(plant_data), desc="处理水厂数据"):
                try:
                    # 运行模拟
                    success = self.model.run_single_plant(
                        COD_in=row['CODinf'] ,
                        BOD_in=row['BODinf'],
                        Q_in=row['Average flow (MGD)'] * 3785,  # m3/day
                        S_NHx_in=row['NH3-Ninf'],
                        S_O2=row.get('S_O2', 0.5),  # 默认值
                        S_NOx=row.get('S_NOx', 1.0)  # 默认值
                    )
                    
                    if success:
                        # 获取完整时间序列结果
                        full_results = self.model.get_single_plant_results()
                        
                        # 只取最后一行（最终状态）
                        final_state = full_results.iloc[-1:].copy()
                        
                        # 将结果添加到原始数据行
                        for col in final_state.columns:
                            plant_data.at[idx, f'Final_{col}'] = final_state[col].values[0]
                        
                except Exception as e:
                    logging.error(f"水厂 {idx} 处理失败: {str(e)}")
                    continue
            
            # 重命名结果列以更清晰
            column_mapping = {
                'Final_time': 'Time (d)',
                'Final_COD_eff': 'COD_eff (mg/L)',
                'Final_COD_removal': 'COD_removal (%)',
                'Final_NHx_eff': 'NHx_eff (mg/L)',
                'Final_NHx_removal': 'NHx_removal (%)',
                'Final_CN_ratio': 'C/N_ratio_HRAS',
                'Final_Sludge_VSS': 'Sludge_VSS (g/a)',
                'Final_Sludge_TSS': 'Sludge_TSS (g/a)',
                'Final_Sludge_COD': 'Sludge_COD (g/a)',
                'Final_Q_Waste_flow': 'Waste_flow (m3/a)'
            }
            plant_data = plant_data.rename(columns=column_mapping)
            
            return plant_data
                
        except Exception as e:
            logging.error(f"批量处理失败: {str(e)}")
            raise

# 示例用法
if __name__ == "__main__":
    # 初始化批量处理器
    processor = BatchHRASProcessor(model_type='diauxic')
    
    # 处理输入文件
    input_csv = r"C:\Users\zz1405\OneDrive - Princeton University\Documents\Work 2_CN energy\Submission to One Earth\Github\Dataset processing\Cleaned final dataset.csv"  # 替换为您的输入文件
    results = processor.process_csv(input_csv)
    results.to_csv("HRAS_output.csv", index=False)
    
    # 统计CN比小于2的数量
    if results is not None:
        count_less_than_2 = (results['C/N_ratio_HRAS'] < 2).sum()
        print(f"CN比小于2的水厂数量: {count_less_than_2}")