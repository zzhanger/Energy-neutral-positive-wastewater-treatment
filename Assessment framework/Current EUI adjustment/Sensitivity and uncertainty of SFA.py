# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 19:45:53 2025

@author: zz1405
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy import stats
import statsmodels.api as sm
from tqdm import tqdm
import warnings
import sys
import os
from SFA import SFAEnergyModel


warnings.filterwarnings('ignore')

class SFASensitivityAnalyzer:
    def __init__(self, sfa_model, data, x_cols, y_col):
        self.sfa_model = sfa_model
        self.data = data
        self.x_cols = x_cols
        self.y_col = y_col
        self.results = sfa_model.results
        
    def _log_likelihood_cost(self, params, X, y):
        beta = params[:-2]
        sigma_v = np.exp(params[-2])
        sigma_u = np.exp(params[-1])
        epsilon = y - X @ beta
        total_sigma = np.sqrt(sigma_u**2 + sigma_v**2)
        lambda_ = sigma_u / sigma_v
        
        term1 = len(y) * np.log(2)
        term2 = -len(y) * np.log(total_sigma)
        term3 = -1/(2 * total_sigma**2) * np.sum(epsilon**2)
        term4 = np.sum(np.log(stats.norm.cdf(epsilon * lambda_ / total_sigma)))
        
        log_lik = term1 + term2 + term3 + term4
        return -log_lik

    def _run_sfa_on_data(self, data):
        try:
            X = sm.add_constant(data[self.x_cols])
            y = data[self.y_col]
            ols_model = sm.OLS(y, X).fit()
            init_params = np.append(ols_model.params, 
                                  [np.log(ols_model.scale), 
                                   np.log(ols_model.scale/2)])
            

            result = minimize(self._log_likelihood_cost, init_params, 
                            args=(X, y), method='L-BFGS-B')
            
            if result.success:
                beta_hat = result.x[:-2]
                sigma_v = np.exp(result.x[-2])
                sigma_u = np.exp(result.x[-1])
                epsilon_hat = y - X @ beta_hat
                mu_star = (epsilon_hat * sigma_u**2) / (sigma_u**2 + sigma_v**2)
                sigma_star = (sigma_u**2 * sigma_v**2) / (sigma_u**2 + sigma_v**2)
                efficiencies = np.exp(-mu_star + 0.5 * sigma_star)
                
                return {
                    'success': True,
                    'efficiencies': efficiencies,
                    'params': result.x
                }
            else:
                return {'success': False, 'error': 'Optimization failed'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def bootstrap_uncertainty(self, n_bootstrap=100, ci_width=0.95):
        
        bootstrap_params = []
        bootstrap_efficiencies = []
        efficiency_stats = []
        failed_samples = 0
        
        for i in tqdm(range(n_bootstrap), desc="Bootstrap Progress"):
            try:

                sample = self.data.sample(frac=1, replace=True)
                result = self._run_sfa_on_data(sample)
                
                if result['success']:
                    efficiencies = result['efficiencies']
                    eff_mean = np.mean(efficiencies)
                    eff_std = np.std(efficiencies)
                    efficiency_stats.append((eff_mean, eff_std))
                    
                    bootstrap_efficiencies.append(efficiencies)
                    bootstrap_params.append(result['params'])
                else:
                    failed_samples += 1
                    
            except Exception as e:
                failed_samples += 1
                continue
        
        if len(bootstrap_params) == 0:
            print("All bootstrap samples failed to converge!")
            return None
        
        bootstrap_params = np.array(bootstrap_params)
        bootstrap_efficiencies = np.array(bootstrap_efficiencies)
        efficiency_stats = np.array(efficiency_stats)
        
        # 计算置信区间
        alpha = (1 - ci_width) / 2
        lower_q = alpha * 100
        upper_q = (1 - alpha) * 100
        
        param_ci = np.percentile(bootstrap_params, [lower_q, upper_q], axis=0)
        eff_ci = np.percentile(bootstrap_efficiencies, [lower_q, upper_q], axis=0)
        avg_eff_ci = np.percentile(np.mean(bootstrap_efficiencies, axis=1), [lower_q, upper_q])
        

        mean_of_means = np.mean(efficiency_stats[:, 0])
        std_of_means = np.std(efficiency_stats[:, 0])
        cv_of_means = std_of_means / mean_of_means if mean_of_means != 0 else np.nan
        
        mean_of_stds = np.mean(efficiency_stats[:, 1])
        std_of_stds = np.std(efficiency_stats[:, 1])
        cv_of_stds = std_of_stds / mean_of_stds if mean_of_stds != 0 else np.nan
        
        results = {
            'param_ci': param_ci,
            'efficiency_ci': eff_ci,
            'avg_efficiency_ci': avg_eff_ci,
            'efficiency_variability': {
                'mean_of_means': mean_of_means,
                'std_of_means': std_of_means,
                'cv_of_means': cv_of_means,
                'mean_of_stds': mean_of_stds,
                'std_of_stds': std_of_stds,
                'cv_of_stds': cv_of_stds
            },
            'bootstrap_samples': len(bootstrap_params),
            'failed_samples': failed_samples,
            'bootstrap_efficiencies': bootstrap_efficiencies,
            'bootstrap_params': bootstrap_params
        }
        
        self._print_bootstrap_summary(results)
        return results


    def outlier_sensitivity(self, n_removals=10):
        X = sm.add_constant(self.data[self.x_cols])
        y = self.data[self.y_col]
        residuals = y - X @ self.results['params']

        outlier_indices = np.argsort(np.abs(residuals))[-n_removals:]
        outlier_facility_ids = self.data.index[outlier_indices].tolist()
        
        rank_results = []
        base_ranks = stats.rankdata(self.results['efficiencies'])
        base_facility_ids = self.data.index.tolist()
        
        for i in tqdm(range(0, n_removals + 1), desc="Outlier Removal"):
            if i == 0:

                current_eff = self.results['efficiencies']
                current_ranks = base_ranks
                removed_facilities = []
                current_data = self.data
            else:

                removed_facilities = outlier_facility_ids[:i]
                current_data = self.data[~self.data.index.isin(removed_facilities)].copy()
                
                # 在修改后的数据上重新运行SFA
                result = self._run_sfa_on_data(current_data)
                
                if result['success']:
                    current_eff = result['efficiencies']
                    current_ranks = stats.rankdata(current_eff)
                else:
                    print(f"⚠️ SFA failed after removing {i} outliers")
                    continue
            

            common_facilities = set(base_facility_ids) - set(removed_facilities)
            common_facilities = list(common_facilities)
            
            if len(common_facilities) < 2:
                print(f"⚠️ Only {len(common_facilities)} facilities remaining after removing {i} outliers")
                continue
            

            base_positions = []
            current_positions = []
            
            for facility in common_facilities:
                base_idx = base_facility_ids.index(facility)
                base_positions.append(base_idx)
                

                if i == 0:
                    current_positions.append(base_idx)
                else:
                    current_idx = current_data.index.tolist().index(facility)
                    current_positions.append(current_idx)
            
            # 计算秩相关系数
            try:
                base_ranks_common = base_ranks[base_positions]
                current_ranks_common = current_ranks[current_positions]
                corr = stats.spearmanr(base_ranks_common, current_ranks_common).correlation
                
                rank_results.append({
                    'n_removed': i,
                    'correlation': corr,
                    'n_facilities': len(common_facilities),
                    'removed_facilities': removed_facilities.copy()
                })
                
            except Exception as e:
                print(f"Correlation calculation failed for {i} removals: {e}")
                continue
        
        if not rank_results:
            print("No valid results from outlier sensitivity analysis")
            return [], outlier_facility_ids
        
        return rank_results, outlier_facility_ids


    def _print_bootstrap_summary(self, results):
        print("\n" + "="*60)
        print("BOOTSTRAP UNCERTAINTY ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nBootstrap Performance:")
        print(f"Successful samples: {results['bootstrap_samples']}")
        print(f"Failed samples: {results['failed_samples']}")
        print(f"Success rate: {results['bootstrap_samples']/(results['bootstrap_samples']+results['failed_samples'])*100:.1f}%")
        
        print(f"\nAverage Efficiency Confidence Interval:")
        print(f"{results['avg_efficiency_ci'][0]:.3f} - {results['avg_efficiency_ci'][1]:.3f}")
        
        print(f"\nEfficiency Variability Across Bootstrap Samples:")
        var_stats = results['efficiency_variability']
        print(f"Mean of means: {var_stats['mean_of_means']:.4f} ± {var_stats['std_of_means']:.4f}")
        print(f"Coefficient of variation: {var_stats['cv_of_means']*100:.1f}%")

    def comprehensive_sensitivity_analysis(self, n_bootstrap=100, variation_range=0.1, n_removals=5):
        
        results = {}
        
        # 1. Bootstrap不确定性分析
        print("\n1. 🎯 BOOTSTRAP UNCERTAINTY ANALYSIS")
        bootstrap_results = self.bootstrap_uncertainty(n_bootstrap=n_bootstrap)
        results['bootstrap'] = bootstrap_results
        
        # 2. 变量敏感性分析
        print("\n2. 📊 VARIABLE SENSITIVITY ANALYSIS")
        variable_results = self.variable_sensitivity(variation_range=variation_range)
        results['variable_sensitivity'] = variable_results
        
        # 3. 异常值敏感性分析
        print("\n3. 🔍 OUTLIER SENSITIVITY ANALYSIS")
        outlier_results, outlier_indices = self.outlier_sensitivity(n_removals=n_removals)
        results['outlier_sensitivity'] = {
            'rank_correlations': outlier_results,
            'outlier_indices': outlier_indices
        }
        
        # 生成综合报告
        self._generate_comprehensive_report(results)
        
        return results
    
    def _generate_comprehensive_report(self, results):
        """生成综合敏感性分析报告"""
        print("\n" + "="*70)
        print("COMPREHENSIVE SENSITIVITY ANALYSIS REPORT")
        print("="*70)
        
        # Bootstrap结果
        if results.get('bootstrap'):
            bootstrap = results['bootstrap']
            print(f"\n📈 BOOTSTRAP UNCERTAINTY:")
            print(f"  Success rate: {bootstrap['bootstrap_samples']/(bootstrap['bootstrap_samples']+bootstrap['failed_samples'])*100:.1f}%")
            print(f"  Average efficiency 95% CI: [{bootstrap['avg_efficiency_ci'][0]:.3f}, {bootstrap['avg_efficiency_ci'][1]:.3f}]")
            print(f"  Efficiency variability (CV): {bootstrap['efficiency_variability']['cv_of_means']*100:.1f}%")
        
        
        # 异常值敏感性
        if results.get('outlier_sensitivity'):
            outlier_sens = results['outlier_sensitivity']
            if outlier_sens['rank_correlations']:
                final_corr = outlier_sens['rank_correlations'][-1]['correlation']
                initial_corr = outlier_sens['rank_correlations'][0]['correlation']
                
                print(f"\n🔍 OUTLIER SENSITIVITY:")
                print(f"  Initial rank correlation: {initial_corr:.3f}")
                print(f"  Final rank correlation: {final_corr:.3f}")
                print(f"  Correlation change: {final_corr - initial_corr:+.3f}")
                robustness = 'High' if (final_corr - initial_corr) > -0.1 else 'Medium' if (final_corr - initial_corr) > -0.2 else 'Low'
                print(f"  Model robustness: {robustness}")
        
        print(f"\n✅ Sensitivity analysis completed successfully!")


def main():
    """主函数：运行完整的SFA和敏感性分析"""
    
    # 配置文件路径
    filepath = r"C:\Users\zz1405\OneDrive - Princeton University\Documents\Work 2_CN energy\Submission to One Earth\Github\Dataset processing\Cleaned final dataset.csv"
    
    print("🚀 Starting Complete SFA and Sensitivity Analysis...")
    
    # 1. 运行主SFA分析
    print("\n" + "="*50)
    print("STEP 1: MAIN SFA ANALYSIS")
    print("="*50)
    
    model = SFAEnergyModel(filepath)
    
    if not model.load_data():
        print("❌ Failed to load data")
        return
    
    main_results = model.run_analysis()
    
    if not main_results:
        print("❌ SFA analysis failed")
        return
    
    model.print_results()

    
    # 2. 运行敏感性分析
    print("\n" + "="*50)
    print("STEP 2: SENSITIVITY ANALYSIS")
    print("="*50)
    
    # 初始化敏感性分析器
    sensitivity_analyzer = SFASensitivityAnalyzer(
        sfa_model=model,
        data=model.data,
        x_cols=model.x_cols,
        y_col=model.y_col
    )
    
    # 执行全面的敏感性分析
    sensitivity_results = sensitivity_analyzer.comprehensive_sensitivity_analysis(
        n_bootstrap=100,        # Bootstrap样本数
        n_removals=5           # 异常值移除数量
    )
    
    print("\n✅ All analyses completed successfully!")

if __name__ == "__main__":
    main()