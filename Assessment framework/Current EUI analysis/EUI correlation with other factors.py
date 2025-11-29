# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 17:25:29 2025

@author: zz1405
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, f_oneway, variation, shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from matplotlib.patches import Patch

# 设置全局字体大小
plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 16, 'axes.labelsize': 16,
    'xtick.labelsize': 16, 'ytick.labelsize': 16, 
    'legend.fontsize': 16, 'figure.titlesize': 16
})

# 加载数据
df = pd.read_csv('WaterPlant_Efficiency_Results.csv')

# 重命名列
df = df.rename(columns={
    'Flow based EUI (kWh/m^3)': 'Flow_EUI',
    'COD removal based EUI (kWh/kg COD removed)': 'COD_EUI',
    'Average_flow__MGD_': 'Treatment_scale',
    'CODinf': 'Influent COD', 'BODinf': 'Influent BOD', 'SSinf': 'Influent SS',
    'NH3_Ninf': 'Influent NH3_N', 'TNinf': 'Influent TN', 'TPinf': 'Influent TP',
    'CODeff': 'Discharge COD', 'BODeff': 'Discharge BOD', 'SSeff': 'Discharge SS',
    'NH3_Neff': 'Discharge NH3_N', 'TNeff': 'Discharge TN', 'TPeff': 'Discharge TP',
    'Treatment_process': 'Treatment_process', 'Annual_HDD': 'Climate'
})

# 定义预测变量
all_predictors = ['Treatment_scale', 'Climate', 
                 'Influent COD', 'Influent BOD', 'Influent SS', 'Influent NH3_N', 'Influent TN', 'Influent TP',
                 'Discharge COD', 'Discharge BOD', 'Discharge SS', 'Discharge NH3_N', 'Discharge TN', 'Discharge TP']

# 变量分类
climate_var = ['Climate']
scale_var = ['Treatment_scale']
influent_vars = [var for var in all_predictors if var.startswith('Influent')]
discharge_vars = [var for var in all_predictors if var.startswith('Discharge')]

# =========================================
# 正态性检验函数
# =========================================
def check_normality(data, alpha=0.05):
    """检验数据是否服从正态分布"""
    stat, p = shapiro(data)
    print(f"Shapiro-Wilk检验: p-value = {p:.4f}", end=" - ")
    if p > alpha:
        print("数据服从正态分布 (可使用Pearson)")
    else:
        print("数据不服从正态分布 (建议使用Spearman)")
    return p > alpha

# =========================================
# 1. 相关性分析 - 使用Spearman
# =========================================
def plot_correlation_barchart(eui_type):
    """绘制变量与EUI相关性的横向柱状图"""
    # 先对目标变量进行正态性检验
    print(f"\n=== {eui_type} 正态性检验 ===")
    is_normal = check_normality(df[eui_type].dropna())
    
    # 计算Spearman相关系数
    correlation_results = []
    for var in all_predictors:
        valid_data = df[[eui_type, var]].dropna()
        if len(valid_data) > 2:  # 至少需要3个点
            corr, p_value = spearmanr(valid_data[eui_type], valid_data[var])
            # 对预测变量进行正态性检验
            var_normal = check_normality(valid_data[var])
        else:
            corr, p_value = np.nan, np.nan
            var_normal = False
        
        # 确定变量类别
        category = ('Climate' if var in climate_var else 
                   'Treatment scale' if var in scale_var else
                   'Influent Characteristics' if var in influent_vars else 
                   'Discharge')
        
        correlation_results.append({
            'Variable': var,
            'Correlation': corr,
            'p-value': p_value,
            'Significant': p_value < 0.05 if not np.isnan(p_value) else False,
            'Category': category,
            'Normal': var_normal
        })
    
    corr_df = pd.DataFrame(correlation_results).dropna().sort_values('Correlation', key=abs)
    
    # 使用viridis调色板
    palette = sns.color_palette("viridis", 4)
    category_colors = {
        'Climate': palette[0], 'Treatment scale': palette[1],
        'Influent Characteristics': palette[2], 'Discharge': palette[3]
    }
    
    # 绘制横向柱状图
    plt.figure(figsize=(14, 8))  # 增大图形尺寸
    colors = [category_colors[cat] if sig else 'lightgray' 
              for cat, sig in zip(corr_df['Category'], corr_df['Significant'])]
    
    bars = plt.barh(corr_df['Variable'], corr_df['Correlation'], 
                   color=colors, edgecolor='black')
    
    # 添加相关系数值标签和显著性标记
    for i, bar in enumerate(bars):
        width = bar.get_width()
        # 相关系数值
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', va='center', 
                ha='left' if width >0 else 'right', fontsize=12)
        
        # 显著性标记（在柱条右侧）
        if corr_df.iloc[i]['Significant']:
            significance_marker = '*' if corr_df.iloc[i]['p-value'] < 0.05 else ''
            significance_marker = '**' if corr_df.iloc[i]['p-value'] < 0.01 else significance_marker
            significance_marker = '***' if corr_df.iloc[i]['p-value'] < 0.001 else significance_marker
            
            # 在柱条右侧添加显著性标记
            x_pos = width + (0.02 if width >= 0 else -0.02)
            plt.text(x_pos, bar.get_y() + bar.get_height()/2, 
                    significance_marker, va='center', ha='left' if width >=0 else 'right', 
                    fontsize=14, fontweight='bold', color='red')
    
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(f'Spearman Correlation with {eui_type}\n(Normality: {"Passed" if is_normal else "Failed"})',
              fontsize=16)
    plt.xlabel('Correlation Coefficient', fontsize=14)
    plt.ylabel('Predictor Variables', fontsize=14)
    
    # 添加图例
    legend_elements = [Patch(facecolor=color, label=cat) 
                      for cat, color in category_colors.items()]
    # 添加显著性图例
    significance_legend = Patch(facecolor='white', label='* p<0.05, ** p<0.01, *** p<0.001')
    legend_elements.append(significance_legend)
    
    # 根据图形类型选择不同的图例位置
    if eui_type == 'Flow_EUI':
        legend_loc = 'lower right'  # 为Flow_EUI设置右上角
    else:
        legend_loc = 'lower left'   # 为COD_EUI设置左下角
    
    plt.legend(handles=legend_elements, loc=legend_loc, 
               title='Variable Categories', framealpha=0.9)
    
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 打印显著性统计
    significant_count = corr_df['Significant'].sum()
    total_count = len(corr_df)
    print(f"显著相关的变量数量: {significant_count}/{total_count} ({significant_count/total_count*100:.1f}%)")
    print("显著性水平: * p<0.05, ** p<0.01, *** p<0.001")
    
    # 打印详细的相关系数表
    print(f"\n详细的相关系数表 ({eui_type}):")
    print("=" * 60)
    for _, row in corr_df.iterrows():
        sig_marker = '*' if row['p-value'] < 0.05 else ''
        sig_marker = '**' if row['p-value'] < 0.01 else sig_marker
        sig_marker = '***' if row['p-value'] < 0.001 else sig_marker
        print(f"{row['Variable']:25} r = {row['Correlation']:6.3f} {sig_marker:3} (p = {row['p-value']:.4f})")

# =========================================
# 2. ANOVA分析
# =========================================
def plot_anova_results(eui_type):
    """绘制处理工艺的ANOVA分析结果"""
    # 执行ANOVA
    groups = [group[eui_type].dropna().values 
              for name, group in df.groupby('Treatment_process')]
    f_stat, p_value = f_oneway(*groups)
    
    # 创建带有两个子图的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))  # 增大图形宽度
    
    # 子图1: 箱线图
    process_order = df.groupby('Treatment_process')[eui_type].median().sort_values().index
    sns.boxplot(data=df, x='Treatment_process', y=eui_type, 
                order=process_order,
                ax=ax1, palette='viridis')
    ax1.set_title(f'{eui_type} by Treatment Process\n(ANOVA p-value = {p_value:.3f})')
    ax1.set_xlabel('Treatment Process')
    ax1.set_ylabel(eui_type + (' (kWh/m³)' if eui_type == 'Flow_EUI' else ' (kWh/kg COD)'))
    ax1.tick_params(axis='x', rotation=45)
    
    # 子图2: CV柱状图
    cv_results = []
    for name, group in df.groupby('Treatment_process'):
        if len(group[eui_type].dropna()) > 1:
            cv = variation(group[eui_type].dropna()) * 100
            cv_results.append({'Treatment_process': name, 'CV': cv})
    
    cv_df = pd.DataFrame(cv_results).sort_values('CV', ascending=False)
    colors = sns.color_palette("viridis", len(cv_df))
    bars = ax2.bar(cv_df['Treatment_process'], cv_df['CV'], color=colors)
    ax2.set_title('Coefficient of Variation (CV) by Treatment Process')
    ax2.set_xlabel('Treatment Process')
    ax2.set_ylabel('CV (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加CV值标签
    for i, v in enumerate(cv_df['CV']):
        ax2.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # 打印ANOVA结果
    print(f"\n=== {eui_type} ANOVA分析结果 ===")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    # 如果ANOVA显著，添加Tukey检验结果
    if p_value < 0.05:
        print("\nTukey HSD Test Results (组间多重比较):")
        tukey = pairwise_tukeyhsd(endog=df[eui_type].dropna(),
                                 groups=df['Treatment_process'].dropna(),
                                 alpha=0.05)
        print(tukey)
        
        # 打印显著的组间差异
        print("\n显著的组间差异 (p < 0.05):")
        significant_pairs = []
        for i in range(len(tukey.reject)):
            if tukey.reject[i]:
                group1 = tukey.groups[tukey.group1[i]]
                group2 = tukey.groups[tukey.group2[i]]
                mean_diff = tukey.meandiff[i]
                p_val = tukey.pvalues[i]
                significant_pairs.append(f"{group1} vs {group2}: mean difference = {mean_diff:.3f}, p = {p_val:.4f}")
        
        if significant_pairs:
            for pair in significant_pairs:
                print(pair)
        else:
            print("没有发现显著的组间差异")
    else:
        print("ANOVA结果不显著，无需进行多重比较")

# =========================================
# 执行分析
# =========================================
if __name__ == "__main__":
    print("开始分析污水处理厂能效数据...")
    print(f"数据集中共有 {len(df)} 个样本")
    print(f"预测变量数量: {len(all_predictors)}")
    
    # 执行相关性分析
    print("\n" + "="*80)
    print("相关性分析结果")
    print("="*80)
    plot_correlation_barchart('Flow_EUI')
    plot_correlation_barchart('COD_EUI')
    
    # 执行ANOVA分析
    print("\n" + "="*80)
    print("ANOVA分析结果")
    print("="*80)
    plot_anova_results('Flow_EUI')
    plot_anova_results('COD_EUI')
    
    # 数据概览
    print("\n" + "="*80)
    print("数据概览")
    print("="*80)
    print(f"Flow_EUI 统计:")
    print(f"  均值: {df['Flow_EUI'].mean():.3f} kWh/m³")
    print(f"  标准差: {df['Flow_EUI'].std():.3f} kWh/m³")
    print(f"  范围: {df['Flow_EUI'].min():.3f} - {df['Flow_EUI'].max():.3f} kWh/m³")
    
    print(f"\nCOD_EUI 统计:")
    print(f"  均值: {df['COD_EUI'].mean():.3f} kWh/kg COD")
    print(f"  标准差: {df['COD_EUI'].std():.3f} kWh/kg COD")
    print(f"  范围: {df['COD_EUI'].min():.3f} - {df['COD_EUI'].max():.3f} kWh/kg COD")
    
    print(f"\n处理工艺种类: {df['Treatment_process'].nunique()}")
    print("处理工艺分布:")
    process_counts = df['Treatment_process'].value_counts()
    for process, count in process_counts.items():
        percentage = count / len(df) * 100
        print(f"  {process}: {count} 个样本 ({percentage:.1f}%)")