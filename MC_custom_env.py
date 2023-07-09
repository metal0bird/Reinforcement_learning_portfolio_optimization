import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class StockPortfolioEnv(gym.Env):
    def __init__(self, prices, max_shares=100, annual_contribution=10000):
        super(StockPortfolioEnv, self).__init__()
        self.prices = prices
        self.max_shares = max_shares
        self.annual_contribution = annual_contribution
        self.current_step = 0
        self.total_steps = len(prices) - 1
        self.action_space = gym.spaces.Discrete(max_shares + 1)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.portfolio_value = 100000
        self.shares_held = 0
        return np.array([self.portfolio_value, self.shares_held])

    def step(self, action):
        current_price = self.prices[self.current_step]
        self.current_step += 1

        # Compute the reward
        shares_bought = min(action, self.max_shares - self.shares_held)
        self.shares_held += shares_bought
        self.portfolio_value += shares_bought * current_price + self.annual_contribution
        reward = self.portfolio_value

        # Check if we have reached the end of the episode
        done = (self.current_step > self.total_steps)

        # Compute the next observation
        next_observation = np.array([self.portfolio_value, self.shares_held])

        # Return the observation, reward, done flag, and any additional information
        return next_observation, reward, done, {}

# Define parameters
num_simulations = 20  # number of simulations to run
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

# Create the Gym environment
env = StockPortfolioEnv(stock_prices)

# Run simulations
portfolio_values = np.zeros((num_simulations, num_years + 1))
for i in range(num_simulations):
    observation = env.reset()
    for j in range(num_years):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        portfolio_values[i, j + 1] = observation[0][0]
    portfolio_values[i, 0] = initial_portfolio_value
print(portfolio_values)
# Plot results
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(num_simulations):
    ax.plot(range(num_years + 1), portfolio_values[i])
ax.set_xlabel('Year')
ax.set_ylabel('Portfolio Value')
ax.set_title('Monte Carlo Simulation of Stock Portfolio with custom environment')
plt.show()