# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:34:49 2025

@author: zz1405
"""

import numpy as np
from scipy.integrate import odeint
import pandas as pd
import logging
from tqdm import tqdm 

class HRASModel:
    def __init__(self, model_type='diauxic'):
        """Initialize HRAS model parameters"""
        self.model_type = model_type
        self.HRT = 0.0208  # Fixed HRT as 30min (converted to day), reference: https://doi.org/10.1016/j.watres.2022.119044 
        
        # Kinetic parameters, reference doi: 10.2166/wst.2015.051
        self.params = {
            # Growth parameters
            'μ_max': 7.0,        # d⁻¹
            'μ_slow': 3.0,       # d⁻¹
            'K_Bf': 5.0,         # g/m³
            'K_Bs': 40.0,        # g/m³
            'Y_OHO': 0.67,       # g VSS/g COD
            'b_OHO': 0.62,       # d⁻¹
            'K_NHx': 0.5,        # g/m³, Ammonia half-saturation coefficient
            'K_O2_OHO': 0.1,     # g/m³, Heterotrophic oxygen half-saturation coefficient
            
            # Nitrogen related parameters, reference Activated Sludge Models ASM1, ASM2, ASM2d and ASM3
            'i_N_XB': 0.07,      # gN/gCOD (cellular nitrogen content)
            'f_N_decay': 0.15,   # Nitrogen release ratio from decay
            
            # Sludge characteristic parameters, reference Activated Sludge Models ASM1, ASM2, ASM2d and ASM3
            'f_VSS_TSS': 0.85,   # VSS/TSS ratio
            'i_XB': 1.48,        # g COD/g VSS (biomass)
            'i_XU': 1.8,         # g COD/g TSS (inert solids)
            'i_XEPS': 1.6,       # g COD/g VSS (EPS)
            'i_XSTO': 1.42,      # g COD/g VSS (storage substances)
            'f_U': 0.08,         # Non-biodegradable fraction
            
            # Other parameters, reference doi: 10.2166/wst.2015.051
            'K_B_HYD': 0.03,
            'q_ADS': 0.07,       # d⁻¹
            'K_SL': 0.002,       # -
            'k_EPS_MAX': 0.25,   # g COD_EPS/g VSS
            'K_O_EPS': 1.5,      # g/m³
            'q_EPS_HYD': 0.12,   # d⁻¹
            'f_Shunt_max': 0.30, # -
            'K_O_STO': 1.0,      # g/m³
            'q_STO_HYD': 3.0,    # d⁻¹
            'K_STO_HYD': 0.15,   # g/g
            'q_XB_HYD': 3.5,     # d⁻¹
            'K_NOx': 0.1,        # g/m³
            'eta_HYD': 0.6,      # Anoxic hydrolysis efficiency coefficient
            'K_EPS': 50.0        # g/m³, EPS half-saturation coefficient
        }
    
    def estimate_influent_fractions(self, COD_in, BOD_in, S_NHx_in):
        """Estimate influent component concentrations, including NH3-N"""
        # Assume soluble fast biodegradable COD equals BOD5 or 40% of COD, whichever is smaller
        SBf = min(BOD_in, COD_in * 0.4)
        remaining = COD_in - SBf
        # Distribution ratios (sum should be remaining)
        SBs = remaining * 0.25
        CB  = remaining * 0.2
        CU  = remaining * 0.15
        XB  = remaining * 0.3
        XU  = remaining * 0.01
        return {
            'SBf': SBf, 'SBs': SBs, 'CB': CB, 'CU': CU, 
            'XB': XB, 'XU': XU, 'S_NHx': S_NHx_in
        }
    
    def sto_frac(self, S_O2):
        """Calculate storage substance shunt fraction"""
        return (self.params['f_Shunt_max'] * 
               (S_O2/(self.params['K_O_STO'] + S_O2)))
    
    def kinetic_rates(self, y, t, Q_in, S_O2, S_NHx, S_NOx):
        """Calculate all kinetic process rates"""
        SBf, SBs, CB, CU, XB, XU, XOHO, XEPS, XSTO, S_NHx = y  
        
        # 1. Aerobic growth (r1 & r2) - NH3-N consumption for biomass synthesis
        growth_fast = (
            self.params['μ_max'] * 
            (SBf/(self.params['K_Bf'] + SBf)) *
            (S_O2/(self.params['K_O2_OHO'] + S_O2)) *
            (S_NHx/(self.params['K_NHx'] + S_NHx)) * XOHO)
        
        if self.model_type == 'diauxic':
            growth_slow = (self.params['μ_slow'] * (SBs/(self.params['K_Bs'] + SBf)) * (
                          self.params['K_Bf']/(self.params['K_Bf']+SBf)) * (
                          S_O2/(self.params['K_O2_OHO'] + S_O2)) * (
                          S_NHx/(self.params['K_NHx'] + S_NHx)) * XOHO)
        else:  # dual substrate
            growth_slow = (self.params['μ_max'] * (SBs/(self.params['K_Bs'] + SBs)) * (
                          S_O2/(self.params['K_O2_OHO'] + S_O2)) * (
                          S_NHx/(self.params['K_NHx'] + S_NHx)) * XOHO)
        
        # 2. Decay (r3)
        decay = self.params['b_OHO'] * XOHO
        
        # 3. Hydrolysis (r4)
        hydrolysis = (
            self.params['q_XB_HYD'] *                           
            ((XB/XOHO) / (self.params['K_B_HYD'] + (XB/XOHO))) *  
            ((S_O2 / (self.params['K_O2_OHO'] + S_O2)) +     
             self.params['eta_HYD'] *                        
             (self.params['K_O2_OHO'] / (self.params['K_O2_OHO'] + S_O2)) * 
             (S_NOx / (self.params['K_NOx'] + S_NOx))) * XOHO)
        
        # 4. Colloidal adsorption (r5 & r6)
        ads_biodegradable = (self.params['q_ADS'] * CB * XOHO * 
                           (self.params['K_SL']/((CB/XOHO) + self.params['K_SL'])) * 
                           (XEPS/(self.params['K_EPS'] + XEPS)))
        
        ads_nonbiodegradable = (self.params['q_ADS'] * CU * XOHO * 
                              (self.params['K_SL']/((CU/XOHO) + self.params['K_SL'])) * 
                              (XEPS/(self.params['K_EPS'] + XEPS)))
        
        # 5. EPS generation 
        k_EPS_SC = (self.params['k_EPS_MAX'] / self.params['i_XB'] * 
                   (S_O2/(self.params['K_O_EPS'] + S_O2)))
        k_EPS_PC = (k_EPS_SC * self.params['Y_OHO'] * (1 - self.sto_frac(S_O2))) / (
                  1 + (k_EPS_SC * self.params['Y_OHO']))
        
        # 6. Storage substance generation
        sto_frac = self.sto_frac(S_O2)
        
        # 7. Hydrolysis processes (r7 & r8)
        sto_hydrolysis = (self.params['q_STO_HYD'] * (XSTO/XOHO) / (
                         self.params['K_STO_HYD'] + XSTO/XOHO) * 
                         (self.params['K_Bf']/(self.params['K_Bf']+SBf)) *
                         (self.params['K_Bs']/(self.params['K_Bs']+SBs)) *
                         (S_O2/(self.params['K_O_EPS'] + S_O2)) * XOHO)
        eps_hydrolysis = self.params['q_EPS_HYD'] * XEPS
        
        # 8. NH3-N change related rates
        nhx_consumption = self.params['i_N_XB'] * (growth_fast + growth_slow)  # Heterotrophic growth consumption
        nhx_release = self.params['f_N_decay'] * decay  # Decay release
        
        return {
            'growth_fast': growth_fast,
            'growth_slow': growth_slow,
            'decay': decay,
            'hydrolysis': hydrolysis,
            'ads_biodegradable': ads_biodegradable,
            'ads_nonbiodegradable': ads_nonbiodegradable,
            'k_EPS_PC': k_EPS_PC,
            'sto_frac': sto_frac,
            'sto_hydrolysis': sto_hydrolysis,
            'eps_hydrolysis': eps_hydrolysis,
            'nhx_consumption': nhx_consumption,
            'nhx_release': nhx_release
        }
    
    def mass_balance(self, y, t, Q_in, S_O2, V_reactor, influent, S_NHx, S_NOx):
        """Define mass balance differential equations"""
        SBf, SBs, CB, CU, XB, XU, XOHO, XEPS, XSTO, S_NHx = y
        rates = self.kinetic_rates(y, t, Q_in, S_O2, S_NHx, S_NOx)
        
        # Influent loading
        dSBf_in = (influent['SBf'] * Q_in - SBf * Q_in) / V_reactor
        dSBs_in = (influent['SBs'] * Q_in - SBs * Q_in) / V_reactor
        dCB_in  = (influent['CB'] * Q_in - CB * Q_in) / V_reactor
        dCU_in  = (influent['CU'] * Q_in - CU * Q_in) / V_reactor
        dXB_in  = (influent['XB'] * Q_in - XB * Q_in) / V_reactor
        dXU_in  = (influent['XU'] * Q_in - XU * Q_in) / V_reactor
        dNHx_in = (influent['S_NHx'] * Q_in - S_NHx * Q_in) / V_reactor
        
        # Biomass effective yield coefficient
        Y_eff = self.params['Y_OHO'] * (1 - rates['k_EPS_PC'] - rates['sto_frac'])
        
        # Differential equations, substrate concentration changes
        dSBf = dSBf_in - (1/Y_eff) * rates['growth_fast'] 
        dSBs = dSBs_in - (1/Y_eff) * rates['growth_slow'] + rates['hydrolysis'] + rates['sto_hydrolysis']
        dCB  = dCB_in - rates['ads_biodegradable']
        dCU  = dCU_in - rates['ads_nonbiodegradable']
        dXB  = dXB_in + rates['ads_biodegradable'] - rates['hydrolysis']
        dXU  = dXU_in + rates['ads_nonbiodegradable'] + self.params['f_U'] * rates['decay']
        dXOHO = (Y_eff * (rates['growth_fast'] + rates['growth_slow']) - 
                rates['decay'] + rates['eps_hydrolysis'])
        dXEPS = (rates['k_EPS_PC'] * (rates['growth_fast'] + rates['growth_slow']) - 
                rates['eps_hydrolysis'])
        dXSTO = (rates['sto_frac'] * (rates['growth_fast'] + rates['growth_slow']) - 
                rates['sto_hydrolysis'])
        dNHx  = (dNHx_in - rates['nhx_consumption'] + rates['nhx_release'])
        
        return [dSBf, dSBs, dCB, dCU, dXB, dXU, dXOHO, dXEPS, dXSTO, dNHx]
    
    def calculate_sludge(self, y, Q_in, V_reactor):
        """Calculate sludge production and COD"""
        XB, XU, XOHO, XEPS, XSTO = y[4], y[5], y[6], y[7], y[8]
        
        VSS_per_vol = XOHO + ((XB/self.params['i_XB']) + (XU/self.params['i_XU']))* 0.6 + (
                     XEPS/self.params['i_XEPS']) + (XSTO/self.params['i_XSTO'])      # Assume 60% particulate COD in sludge
        TSS_per_vol = VSS_per_vol / self.params['f_VSS_TSS']
        COD_per_vol = (XOHO * self.params['i_XB']) + (XB + XU) * 0.6 + XEPS + XSTO
        SRT = 0.5  # Sludge age (d)
        Q_waste = V_reactor / SRT        
        return {
            'VSS': VSS_per_vol * Q_waste,
            'TSS': TSS_per_vol * Q_waste,
            'COD': COD_per_vol * Q_waste,
            'Q_waste': Q_waste
        }
    
    def calculate_effluent_COD(self, state):
        """Calculate effluent COD"""
        SBf, SBs, CB, CU, *_ = state
        return SBf + SBs + CB + CU + (state[4] + state[5]) * 0.4  # 40% particulate COD remains in water
    
    def run_single_plant(self, COD_in, BOD_in, Q_in, S_NHx_in, S_O2=0.5, S_NOx=1.0, t_span=None):
        """Run model simulation for a single plant"""
        if t_span is None:
            t_span = np.linspace(0, 2, 400)  # recommended SRT=0.5days
        
        influent = self.estimate_influent_fractions(COD_in, BOD_in, S_NHx_in)
        V_reactor = Q_in * self.HRT
        
        # Initial conditions (including NH3-N initial value)
        y0 = [
            influent['SBf'] * 0.1,
            influent['SBs'] * 0.1,
            influent['CB'] * 0.1,
            influent['CU'] * 0.1,
            influent['XB'] * 0.1,
            influent['XU'] * 0.1,
            100.0,          # XOHO
            1.0,            # XEPS
            1.0,            # XSTO
            S_NHx_in * 0.1  # Initial NH3-N
        ]
        
        # Run simulation
        solution = odeint(self.mass_balance, y0, t_span,
                         args=(Q_in, S_O2, V_reactor, influent, S_NHx_in, S_NOx))
        
        # Process results
        results = []
        for i, t in enumerate(t_span):
            state = solution[i]
            effluent_COD = self.calculate_effluent_COD(state)
            cod_removal = (COD_in - effluent_COD) / COD_in * 100
            
            # Dynamically calculate NH3-N removal rate
            current_nhx = state[-1]
            nhx_removal = (S_NHx_in - current_nhx) / S_NHx_in * 100
            cn_ratio = effluent_COD / current_nhx if current_nhx > 0 else float('inf')
            
            # Calculate sludge data for current time point
            sludge_data = self.calculate_sludge(state, Q_in, V_reactor)
            
            results.append({
                'time': t,
                'COD_eff': effluent_COD,
                'COD_removal': cod_removal,
                'NHx_eff': current_nhx,
                'NHx_removal': nhx_removal,
                'CN_ratio': cn_ratio,
                'Sludge_VSS': sludge_data['VSS'] * 365,        # Add VSS data (g/a)
                'Sludge_TSS': sludge_data['TSS'] * 365,        # Add TSS data (g/a)
                'Sludge_COD': sludge_data['COD'] * 365,        # Add sludge COD data (g/a)
                'Q_Waste_flow': sludge_data['Q_waste'] * 365   # Add waste sludge flow (m3/a)
            })
        
        self.single_plant_results = pd.DataFrame(results)
        return True
    
    def get_single_plant_results(self):
        """Get results from the last single plant simulation"""
        return getattr(self, 'single_plant_results', None)