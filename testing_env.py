import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from enum import Enum
import matplotlib.pyplot as plt

#importing env class 
from Trading_Stock_env import StocksEnv

# Stable baselines - rl stuff
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

df=pd.read_csv("teststock2.csv")

#df.head()
#len(df)

df['Date'] = pd.to_datetime(df['Date'])
df['Volume'] = df['Volume'].apply(lambda x: float(x.replace(",", "")))
#df.dtypes

df.set_index('Date', inplace=True)
#df.head()

env = gym.make('stocks-v0', df=df, frame_bound=(5,253), window_size=5)
env=StocksEnv(df,frame_bound=(5,253), window_size=5)

#features of env

#env.observation_space
#env.signal_features
#env.action_space

state = env.reset()
while True: 
    action = env.action_space.sample()
    n_state, reward, done, info = env.step(action)
    if done: 
        print("info", info)
        break
        
plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()


