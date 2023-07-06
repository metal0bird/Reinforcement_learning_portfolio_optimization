from multistockenv import MultiStockTradingEnv, add_features
import numpy as np
import gym
import matplotlib.pyplot as plt
import glob
import pandas as pd

# from multi_stock_trading_env import MultiStockTradingEnv
from stable_baselines3 import SAC
CUDA_LAUNCH_BLOCKING=1 
directory = 'history_data'

indicators = ['open', 'high', 'low', 'close', 'volume', 'ToD', 'DoW',
        'ret1min', 'ret2min', 'ret3min', 'ret4min', 'ret5min', 'ret6min',
        'ret7min', 'ret8min', 'ret9min', 'ret10min', 'sma', '5sma', '20sma',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_sell', 'bb_buy', 'bb_squeeze',
        'mom', 'adx', 'mfi', 'rsi', 'trange', 'bop', 'cci', 'STOCHRSI', 'slowk',
        'slowd', 'macd', 'macdsignal', 'macdhist', 'NATR', 'KAMA', 'MAMA',
        'FAMA', 'MAMA_buy', 'KAMA_buy', 'sma_buy', 'maco', 'rsi_buy',
        'rsi_sell', 'macd_buy_sell']

dfs = pd.DataFrame()
num_assets = 0
names = []
data_files = glob.iglob(f'.\{directory}/*')
'''modifying the columns for training model'''
for filename in data_files:
        df = pd.read_csv(filename)
        df['datetime'] = pd.to_datetime(df['datetime'])
        name = df['name'].iloc[0]
        names.append(name)
        df['ToD'] = df['datetime'].dt.hour + df['datetime'].dt.minute/60
        df['DoW'] = df['datetime'].dt.weekday/6
        df.sort_values(['timestamp'], inplace=True)
        updated_df = add_features(df)
        updated_df['datetime'] = pd.to_datetime(updated_df['datetime'])
        updated_df = df.set_index(pd.DatetimeIndex(updated_df['datetime']))
        updated_df.drop(['timestamp','name','token'], axis=1, inplace=True)
        updated_df.replace([np.inf, -np.inf], 0, inplace=True)
        dfs = pd.concat([dfs,updated_df], axis=1)
        num_assets += 1
dfs.interpolate(method='pad', limit_direction='forward', inplace=True)
#print(dfs.columns)
cols_per_asset = int(len(dfs.columns)/num_assets)
df_list = []
price_df = pd.DataFrame()
for i in range(num_assets):
        df = dfs.iloc[:,i*cols_per_asset:i*cols_per_asset+cols_per_asset]
        #print(df.columns)
        df.drop(['datetime'], axis=1, inplace=True)
        price_df[names[i]] = df['close']
        df_list.append(df)
cols_per_asset -= 1
#print(names)

env = MultiStockTradingEnv(df_list,
        price_df,
        num_stocks=num_assets,
        initial_amount=1000000,
        trade_cost=0,
        num_features=cols_per_asset,
        window_size=12,
        frame_bound = (12,len(price_df)-1500),
        tech_indicator_list=indicators)

prices, features = env.process_data()
model = SAC("MlpPolicy", env, verbose=2,tensorboard_log='tb_logs')
model.learn(total_timesteps=100000)

plt.figure(figsize=(16, 6))
name="sacmodel"
model.save("saved_models/"+name)
