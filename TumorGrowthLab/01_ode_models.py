import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class TumorGrowthModel:
    def __init__(self, r=0.2, K=1000, P0=10, T=100, dt=0.1):
        self.r = r                  # growth rate
        self.K = K                  # carrying capacity
        self.P0 = P0                # initial population
        self.T = T                  # total time
        self.dt = dt                # time step (for discrete simulation)
        self.steps = int(T / dt)
        self.t_eval = np.linspace(0, T, 500)
        self.results_continuous = None
        self.results_discrete = None

    def logistic_ode(self, t, P):
        return self.r * P * (1 - P / self.K)

    def gompertz_ode(self, t, P):
        return self.r * P * np.log(self.K / P)

    def simulate_continuous(self):
        sol_log = solve_ivp(self.logistic_ode, (0, self.T), [self.P0], t_eval=self.t_eval)
        sol_gomp = solve_ivp(self.gompertz_ode, (0, self.T), [self.P0], t_eval=self.t_eval)

        self.results_continuous = pd.DataFrame({
            'time': self.t_eval,
            'logistic': sol_log.y[0],
            'gompertz': sol_gomp.y[0]
        })

        return self.results_continuous

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

    def save_to_csv(self):
        if self.results_continuous is not None:
            self.results_continuous.to_csv('simulated_growth_model.csv', index=False)
        if self.results_discrete is not None:
            self.results_discrete.to_csv('simulated_growth_discrete.csv', index=False)

    def plot_results(self):
        if self.results_discrete is not None:
            plt.figure(figsize=(10, 5))
            plt.plot(self.results_discrete['time'], self.results_discrete['logistic_discrete'],
                     label='Logistic (Discrete)', linestyle='--', color='k')
            plt.plot(self.results_discrete['time'], self.results_discrete['gompertz_discrete'],
                     label='Gompertz (Discrete)', linestyle='--', color='g')
            plt.xlabel('Time')
            plt.ylabel('Tumor Size')
            plt.title('Discrete-Time Tumor Growth Simulation')
            plt.legend()
            plt.grid(True)
            plt.show()

        if self.results_continuous is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(self.results_continuous['time'], self.results_continuous['logistic'],
                     label='Logistic Growth', linewidth=2, color='r')
            plt.plot(self.results_continuous['time'], self.results_continuous['gompertz'],
                     label='Gompertz Growth', linewidth=2, linestyle='--', color='b')
            plt.title("Tumor Growth Models: Logistic vs Gompertz")
            plt.xlabel("Time (days)")
            plt.ylabel("Tumor Size")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


# Usage
if __name__ == "__main__":
    model = TumorGrowthModel(r=0.5, K=100, P0=10, T=10, dt=0.1)

    df_cont = model.simulate_continuous()
    df_disc = model.simulate_discrete()

    print("Continuous Model Sample:\n", df_cont.head())
    print("\nDiscrete Model Sample:\n", df_disc.head())

    model.save_to_csv()
    model.plot_results()
