import numpy as np
import pandas as pd

# Load data from CSV file
data=pd.read_csv("teststock2.csv")
# Define Q-learning parameters
num_states = 100  # number of states
num_actions = 3   # number of actions (buy, sell, hold)
epsilon = 0.1     # exploration rate
alpha = 0.2       # learning rate
gamma = 0.9       # discount factor
max_episodes = 1000  # maximum number of episodes
max_steps = 100     # maximum number of steps per episode

# Define state and action functions
def discretize_state(observation):
    state = int(np.floor((observation - data.min()) / (data.max() - data.min() + 0.0001) * (num_states - 1)))
    return np.clip(state, 0, num_states - 1)

def choose_action(state, Q, epsilon):
    if np.random.uniform() < epsilon:
        # choose a random action
        action = np.random.choice(num_actions)
    else:
        # choose the action with the highest Q-value
        action = np.argmax(Q[state])
    return action

# Initialize Q-table
Q = np.zeros([num_states, num_actions])

# Train agent using Q-learning
for episode in range(max_episodes):
    # Reset environment
    state = 0
    total_reward = 0
    for step in range(max_steps):
        # Choose action
        action = choose_action(state, Q, epsilon)
        # Apply action to environment
        reward = data.iloc[state+1]['Close'] - data.iloc[state]['Close']
        next_state = discretize_state(data.iloc[state+1]['Close'])
        # Update Q-value
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state,action]) - Q[state, action])
        # Update state and total reward
        state = next_state
        total_reward += reward
    # Print episode results
    print("Episode {}: Total reward = {}".format(episode+1, total_reward))