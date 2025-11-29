# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 18:56:29 2025

@author: zz1405
"""

import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize
from scipy import stats
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

class SFAEnergyModel:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.results = None

    def load_data(self):
        """Load and preprocess data"""
        print("Loading data...")
        try:
            self.data = pd.read_csv(self.filepath)
            predictors = ['Average flow (MGD)', 'CODinf', 'BODinf', 'SSinf', 
                         'TPinf', 'NH3-Ninf', 'TNinf', 'CODeff']
            self.y_col = 'Pollutant removal based EUI (kWh/kg COD removed)'
            required_cols = predictors + [self.y_col]
            
            print(f"Original data: {len(self.data)} records")
            self.data = self.data.dropna(subset=required_cols)
            print(f"After removing missing values: {len(self.data)} records")

            self.x_cols = []
            for col in predictors:
                log_col = f'log_{col}'
                min_val = self.data[col].min()
                
                if min_val <= 0:
                    offset = abs(min_val) + 0.001
                    self.data[col] += offset
                    print(f"Applied offset {offset:.4f} to {col} for log-transform")
                
                self.data[log_col] = np.log(self.data[col])
                self.x_cols.append(log_col)

            return True
            
        except Exception as e:
            print(f"Data loading failed: {str(e)}")
            return False

    def _log_likelihood_cost(self, params, X, y):
        """Log-likelihood cost function (half-normal distribution)"""
        beta = params[:-2]
        sigma_v = np.exp(params[-2])
        sigma_u = np.exp(params[-1])
        
        epsilon = y - X @ beta
        total_sigma = np.sqrt(sigma_u**2 + sigma_v**2)
        lambda_ = sigma_u / sigma_v
        
        term1 = len(y) * np.log(np.sqrt(2/np.pi))
        term2 = -len(y) * np.log(total_sigma)
        term3 = -1/(2 * total_sigma**2) * np.sum(epsilon**2)
        term4 = np.sum(np.log(stats.norm.cdf(epsilon * lambda_ / total_sigma)))
        
        return -(term1 + term2 + term3 + term4)

    def run_analysis(self):
        """Run SFA analysis (cost function)"""
        print("Running SFA analysis (Cost Function)...")
        
        X = sm.add_constant(self.data[self.x_cols])
        y = self.data[self.y_col]

        ols_model = sm.OLS(y, X).fit()
        init_params = np.append(ols_model.params, [np.log(ols_model.scale), np.log(ols_model.scale / 2)])

        # Optimize using cost function likelihood
        result = minimize(self._log_likelihood_cost, init_params, args=(X, y), 
                          method='L-BFGS-B', options={'maxiter': 1000})
        
        if result.success:
            print("Optimization converged successfully")
            self._process_results_cost(result, X, y)
            return self.results
        else:
            print(f"Optimization failed: {result.message}")
            return None

    def _process_results_cost(self, result, X, y):
        """Process results - cost function"""
        beta_hat = result.x[:-2]
        sigma_v = np.exp(result.x[-2])
        sigma_u = np.exp(result.x[-1])
        
        epsilon_hat = y - X @ beta_hat
        
        print(f"\nResidual Analysis (Cost Function):")
        print(f"  ε = y - Xβ, Min: {epsilon_hat.min():.4f}, Max: {epsilon_hat.max():.4f}")

        mu_star = (epsilon_hat * sigma_u**2) / (sigma_u**2 + sigma_v**2)
        sigma_star = (sigma_u**2 * sigma_v**2) / (sigma_u**2 + sigma_v**2)

        efficiencies = np.exp(-mu_star + 0.5 * sigma_star)
        efficiencies = np.clip(efficiencies, 0.001, 1.0)
        
        self.results = {
            'params': beta_hat,
            'sigma_u': sigma_u,
            'sigma_v': sigma_v,
            'lambda': sigma_u / sigma_v,
            'efficiencies': efficiencies,
            'log_likelihood': -result.fun,
            'residuals': epsilon_hat,
            'mu_star': mu_star,
            'sigma_star': sigma_star,
            'converged': result.success
        }

        self._calculate_fit_stats(X, y)

    def _calculate_fit_stats(self, X, y):
        """Calculate model fit statistics"""
        n = len(y)
        k = len(self.results['params']) + 2  # +2 for sigma parameters
        
        self.results.update({
            'aic': 2 * k + 2 * (-self.results['log_likelihood']),
            'bic': np.log(n) * k + 2 * (-self.results['log_likelihood']),
            'n_observations': n,
            'n_parameters': k
        })

    def print_results(self):
        """Print main results"""
        if not self.results:
            print("No results available")
            return
        
        print("\n" + "="*60)
        print("SFA COST FUNCTION ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\nModel Fit Statistics:")
        print(f"Log-Likelihood: {self.results['log_likelihood']:.2f}")
        print(f"AIC: {self.results['aic']:.2f}")
        print(f"BIC: {self.results['bic']:.2f}")
        print(f"Observations: {self.results['n_observations']}")

        print(f"\nVariance Parameters:")
        print(f"Sigma_u (inefficiency): {self.results['sigma_u']:.4f}")
        print(f"Sigma_v (noise): {self.results['sigma_v']:.4f}")
        print(f"Lambda (σ_u/σ_v): {self.results['lambda']:.4f}")

        print(f"\nParameter Estimates:")
        param_names = ['const'] + self.x_cols
        for name, value in zip(param_names, self.results['params']):
            print(f"{name:25}: {value:10.4f}")

    def get_efficiency_data(self):
        """Return efficiency score data"""
        if not self.results:
            return None
        
        return pd.DataFrame({
            'Actual_EUI': self.data[self.y_col],
            'Efficiency_Score': self.results['efficiencies'],
        })

# Example usage
if __name__ == "__main__":
    model = SFAEnergyModel(
        r"C:\Users\zz1405\OneDrive - Princeton University\Documents\Work 2_CN energy\Submission to One Earth\Github\Dataset processing\Cleaned final dataset.csv"
    )
    
    if model.load_data():
        results = model.run_analysis()
        
        if results:
            model.print_results()
            eff_data = model.get_efficiency_data()
            print(f"\n📋 Efficiency data summary:")
            print(eff_data.describe())
            eff_data.to_csv('sfa_efficiency_results.csv', index=False)
            print("Results saved to 'sfa_efficiency_results.csv'")