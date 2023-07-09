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

#env_maker = lambda: gym.make('stocks-v0', df=df, frame_bound=(5,100), window_size=5)
#env = DummyVecEnv([env_maker])

env = gym.make('stocks-v0', df=df, frame_bound=(5,253), window_size=5)
env=StocksEnv(df,frame_bound=(5,253), window_size=5)

#Applying the Trading RL Algorithm
#model_train = A2C('MlpLstmPolicy', env, verbose=1) 
model = A2C('MlpPolicy', env, verbose=1) 

#setting the learning timesteps
model.learn(total_timesteps=10000)

obs = env.reset()
while True:
    action = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    plt.figure(figsize=(15,6))
    env.render_all()



