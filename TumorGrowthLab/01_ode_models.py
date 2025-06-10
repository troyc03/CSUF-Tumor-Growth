import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import matplotlib.pyplot as plt

class TumorGrowthModel:

    def __init__(self, r=0.2, K=1000, P0=10, T=100, dt=0.1):
        self.r = r # Growth rate
        self.K = K # Carrying capacity
        self.P0 = P0 # Initial condition
        self.T = T # Time
        self.dt = dt # Change in time
        self.steps = int(T / dt) # Time step
        self.results_continuous = None
        self.results_discrete = None

    # Logistic model
    def logistic_model(self, t, P):
        return self.r * P * (1 - P / self.K)

    # Gompertz model
    def gompertz_model(self, t, P):
        return self.r * P * np.log(self.K / P)

    # Solve continuous ODE system
    def simulate_continuous(self):
        t_eval = np.linspace(0, self.T, 500)
        sol_logistic = solve_ivp(self.logistic_model, (0, self.T), [self.P0], t_eval=t_eval)
        sol_gompertz = solve_ivp(self.gompertz_model, (0, self.T), [self.P0], t_eval=t_eval)

        self.results_continuous = pd.DataFrame({
            'time': t_eval,
            'logistic': sol_logistic.y[0],
            'gompertz': sol_gompertz.y[0]
        })
        return self.results_continuous

    # Solve discrete recurrence relations
    def simulate_discrete(self):
        time = np.linspace(0, self.T, self.steps + 1)
        logistic_vals = np.zeros(self.steps + 1)
        gompertz_vals = np.zeros(self.steps + 1)
        logistic_vals[0] = self.P0
        gompertz_vals[0] = self.P0

        for n in range(self.steps):
            P_log = logistic_vals[n]
            P_gomp = gompertz_vals[n]
            logistic_vals[n + 1] = P_log + self.r * P_log * (1 - P_log / self.K) * self.dt
            gompertz_vals[n + 1] = P_gomp + self.r * P_gomp * np.log(self.K / P_gomp) * self.dt

        self.results_discrete = pd.DataFrame({
            'time': time,
            'logistic_discrete': logistic_vals,
            'gompertz_discrete': gompertz_vals
        })
        return self.results_discrete

    # Save both datasets if available
    def save_to_csv(self):
        if self.results_continuous is not None:
            self.results_continuous.to_csv('simulated_growth_continuous.csv', index=False)
        if self.results_discrete is not None:
            self.results_discrete.to_csv('simulated_growth_discrete.csv', index=False)

    # Plot both sets of results
    def plot_results(self):
        if self.results_discrete is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(self.results_discrete['time'], self.results_discrete['logistic_discrete'],
                     label='Logistic (Discrete)', linestyle='--', color='black')
            plt.plot(self.results_discrete['time'], self.results_discrete['gompertz_discrete'],
                     label='Gompertz (Discrete)', linestyle='--', color='green')
            plt.xlabel('Time')
            plt.ylabel('Tumor Size')
            plt.title('Discrete-Time Tumor Growth Simulation')
            plt.legend()
            plt.grid(True)
            plt.show()

        if self.results_continuous is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(self.results_continuous['time'], self.results_continuous['logistic'],
                     label='Logistic Growth (Continuous)', linewidth=2, color='red')
            plt.plot(self.results_continuous['time'], self.results_continuous['gompertz'],
                     label='Gompertz Growth (Continuous)', linewidth=2, linestyle='--', color='blue')
            plt.xlabel('Time (days)')
            plt.ylabel('Tumor Size')
            plt.title('Tumor Growth Models: Logistic vs Gompertz')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

# Usage
if __name__ == '__main__':
    model = TumorGrowthModel(r=0.5, K=100, P0=10, T=10, dt=0.1)

    df_cont = model.simulate_continuous()
    df_disc = model.simulate_discrete()

    print("Continuous Model Sample:\n", df_cont.head())
    print("\nDiscrete Model Sample:\n", df_disc.head())

    model.save_to_csv()
    model.plot_results()
