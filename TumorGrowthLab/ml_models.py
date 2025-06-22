import numpy as np
import matplotlib.pyplot as plt
from ode_models import TumorGrowthModel

class LinearRegressionModel:
    
    @staticmethod
    def basic_linear_regression(x: np.ndarray, y: np.ndarray):
        """
        Constructs a basic linear regression model.
        """
        N = len(x)
        
        # Input validation
        if len(x) != len(y):
            raise ValueError(f"Error: x (size {len(x)}) and y (size {len(y)}) must be the same length.")
        if N <= 2:
            raise ValueError("Error: Need at least 3 points for linear regression.")
        
        # Key calculations
        x_sum, y_sum = np.sum(x), np.sum(y)
        x_avg, y_avg = x_sum / N, y_sum / N
        
        # Covariance and variance calculations
        xy_cov = np.sum((x - x_avg) * (y - y_avg))
        x_var = np.sum((x - x_avg) ** 2)
        
        slope = xy_cov / x_var
        intercept = y_avg - slope * x_avg
        
        # Error calculations
        predictions = intercept + slope * x
        residuals = y - predictions
        rss = np.sum(residuals ** 2)
        std_error = np.sqrt(rss / (N - 2)) if N > 2 else 0
        
        slope_error = std_error / np.sqrt(x_var)
        intercept_error = std_error * np.sqrt(1/N + (x_avg ** 2) / x_var)
        
        # Correlation coefficient
        y_var = np.sum((y - y_avg) ** 2)
        r = xy_cov / np.sqrt(x_var * y_var)
        
        return slope, intercept, r, predictions, slope_error, intercept_error
    
    @staticmethod
    def plot_regression(x: np.ndarray, y: np.ndarray, predictions: np.ndarray,
                        title: str = "Tumor Growth Regression"):
        """
        Visualizes regression results.
        """
        plt.figure(figsize=(10, 8))
        plt.scatter(x, y, label='Actual Data', color='blue')
        plt.plot(x, predictions, label='Regression Line', color='red', linewidth=2)
        plt.xlabel('Time (days)')
        plt.ylabel('Tumor Size')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
        
def main():
    """
    Example linear regression model.
    """
    # Generate growth data
    model = TumorGrowthModel(r=0.5, K=100, P0=10, T=10, dt=0.1)
    cont_data = model.simulate_continuous()
    disc_data = model.simulate_discrete()
    
    # Run regression model for both discrete and continuous data
    for name, data, target in [
            ('Continuous Growth (Logistic)', cont_data, 'logistic'),
            ('Discrete Growth (Logistic)', disc_data, 'logistic_discrete')
            ]:
        x = data['time'].values
        y = data[target].values
        
        slope, intercept, r, preds, slope_err, int_err = LinearRegressionModel.basic_linear_regression(x, y)
        
        # Print results
        print(f'\n{name} Regression results:')
        print(f'- Slope: {slope:.4f} +/- {slope_err:.4f}')
        print(f'- Intercept: {intercept:.4f} +/- {int_err:.4f}')
        print(f'- R: {r:.4f}')
        
        # Plot results
        LinearRegressionModel.plot_regression(x, y, preds, f"{name} Regression")

if __name__ == "__main__":
    main()
