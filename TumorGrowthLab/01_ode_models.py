"""
File: 01_ode_models.py
Purpose: Solves the continuous and discrete dynamical systems of ODEs
for the Logistic and Gompertz Growth equations.
Version: 1.0
Author: Troy Chin
Date: 9 June 2025
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp

class TumorGrowthModel:
    
    def __init__(self, r, K, P0):
        self.r = r
        self.K = K
        self.P0 = P0
        
    def logistic_model(self, t, P):
        return self.r * P * (1 - P/self.K)
    
    def gompertz_model(self, t, P):
        return self.r * P * np.log(self.K/P)
    
    def solve_continuous(self, t_span, t_eval):
        sol_log = solve_ivp(self.logistic_model, t_span, [self.P0], t_eval=t_eval)
        sol_gom = solve_ivp(self.gompertz_model, t_span, [self.P0], t_eval=t_eval)
        
        df = pd.DataFrame({
            'time': t_eval,
            'logistic': sol_log.y[0],
            'gompertz': sol_gom.y[0]
        })
        
        df.to_csv('simulated_growth_model.csv', index=False)
        return df
    
    def simulate_discrete(self, T, dt):
        steps = int(T / dt)
        time = np.linspace(0, T, steps + 1)
        
        logistic_vals = np.zeros(steps + 1)
        gompertz_vals = np.zeros(steps + 1)
        logistic_vals[0] = self.P0
        gompertz_vals[0] = self.P0
        
        for n in range(steps):
            P_log = logistic_vals[n]
            P_gom = gompertz_vals[n]
            logistic_vals[n + 1] = P_log + self.r * P_log * (1 - P_log / self.K) * dt
            gompertz_vals[n + 1] = P_gom + self.r * P_gom * np.log(self.K / P_gom) * dt
            
        df = pd.DataFrame({
            'time': time,
            'logistic_discrete': logistic_vals,
            'gompertz_discrete': gompertz_vals
        })
        
        df.to_csv('simulated_growth_discrete.csv', index=False)
        return df
    
    def plot_continuous(self, df):
        plt.figure(figsize=(10, 6))
        plt.plot(df['time'], df['logistic'], label='Logistic Growth', linewidth=2)
        plt.plot(df['time'], df['gompertz'], label='Gompertz Growth', linewidth=2, linestyle='--')
        plt.title('Tumor Growth Models: Logistic vs Gompertz (Continuous)')
        plt.xlabel('Time (t)')
        plt.ylabel('Tumor Size')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def plot_discrete(self, df):
        plt.figure(figsize=(10, 6))
        plt.plot(df['time'], df['logistic_discrete'], label='Logistic Growth (Discrete)', linestyle='--')
        plt.plot(df['time'], df['gompertz_discrete'], label='Gompertz Growth (Discrete)', linestyle='--')
        plt.title("Tumor Growth Models: Logistic vs Gompertz (Discrete)")
        plt.xlabel("Time")
        plt.ylabel("Tumor Size")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    model = TumorGrowthModel(r=0.2, K=1000, P0=10)
    
    # Continuous simulation
    t_span = (0, 100)
    t_eval = np.linspace(t_span[0], t_span[1], 500)
    df_continuous = model.solve_continuous(t_span, t_eval)
    model.plot_continuous(df_continuous)
    
    # Discrete simulation
    df_discrete = model.simulate_discrete(T=10, dt=0.01)
    model.plot_discrete(df_discrete)
