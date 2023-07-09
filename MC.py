import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define parameters
num_simulations = 1000  # number of simulations to run
num_years = 5  # number of years to simulate
initial_portfolio_value = 100000  # initial value of the portfolio
annual_contribution = 10000  # amount of annual contribution
mean_returns = 0.07  # mean annual return
volatility = 0.2  # annual volatility
correlation = 0.5  # correlation between stock returns

# Generate stock price data
num_months = num_years * 12
dates = pd.date_range(start='01/01/2020', periods=num_months, freq='MS')
stock_returns = np.random.multivariate_normal(
    mean=[mean_returns, mean_returns],
    cov=[[volatility ** 2, volatility * volatility * correlation],
         [volatility * volatility * correlation, volatility ** 2]],
    size=num_months)
stock_prices = initial_portfolio_value * np.cumprod(1 + stock_returns, axis=0)

# Run simulations
portfolio_values = np.zeros((num_simulations, num_years + 1))
for i in range(num_simulations):
    portfolio_value = initial_portfolio_value
    for j in range(num_years):
        portfolio_value += annual_contribution
        portfolio_value *= (1 + np.random.normal(mean_returns, volatility))
        portfolio_values[i, j + 1] = portfolio_value
    portfolio_values[i, 0] = initial_portfolio_value

# Plot results
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(num_simulations):
    ax.plot(range(num_years + 1), portfolio_values[i])
ax.set_xlabel('Year')
ax.set_ylabel('Portfolio Value')
ax.set_title('Monte Carlo Simulation of Stock Portfolio')
plt.show()

