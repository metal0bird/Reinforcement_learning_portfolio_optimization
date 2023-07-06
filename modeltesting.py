CUDA_LAUNCH_BLOCKING=1 
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import glob
from multistockenv import MultiStockTradingEnv, add_features
from stable_baselines3 import SAC,A2C,PPO

directory = 'test_data'

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
for filename in data_files:
        
        if "SENSEXA" in filename:
                print("a",filename)
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
                print(updated_df.head())
                dfs = pd.concat([dfs,updated_df], axis=1)
                num_assets += 1
        
        else:
                print(filename)
                df = pd.read_csv(filename)
                df.rename(columns = {'timestamp':'datetime'}, inplace = True)
                df['datetime'] = pd.to_datetime(df['datetime'])
                name = filename
                names.append(name[12:-5])
                df['ToD'] = df['datetime'].dt.hour + df['datetime'].dt.minute/60
                df['DoW'] = df['datetime'].dt.weekday/6
                updated_df = add_features(df)
                updated_df['datetime'] = pd.to_datetime(updated_df['datetime'])
                updated_df = df.set_index(pd.DatetimeIndex(updated_df['datetime']))
                updated_df.replace([np.inf, -np.inf], 0, inplace=True)
                print(updated_df.head())
                dfs = pd.concat([dfs,updated_df], axis=1)
                num_assets += 1
dfs.interpolate(method='pad', limit_direction='forward', inplace=True)
print(dfs.columns)
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

'''change the suppression rate and transaction cost in the multistockenv python script to analyse how they affect the rl models.'''
models=["ppomodel","a2cmodel","sacmodel"]
for a in models:
        env = MultiStockTradingEnv(df_list,
        price_df,
        num_stocks=num_assets,
        initial_amount=1000000,
        trade_cost=0,
        num_features=cols_per_asset,
        window_size=12,
        scalers=scalers,
        frame_bound = (len(price_df)-1500,len(price_df)),
        tech_indicator_list=indicators)
        if a=="ppomodel":
                model = PPO.load("saved_models/"+a)
        elif a=="a2cmodel":
                model = A2C.load("saved_models/"+a)
        else:
                model = SAC.load("saved_models/"+a) 
        print("\n",a.upper())
        prices, features = env.process_data()
        obs = env.reset()
        count=0
        total_rewards = 0
        infer_rewards = []
        while True: 
        # obs = obs[np.newaxis, ...]
                action, _states = model.predict(obs)
                count+=1
                obs, rewards, done, info = env.step(action)
        # print(action, rewards,sum(action))
                total_rewards += rewards
                infer_rewards.append(rewards)
                if done:
                        print("info", count,info)
                        break
        print("Total profit from "+name+": \n", sum(infer_rewards))
        # print("Sensex profit: \n", env.representative[-1]-env.representative[0])
        infer_steps = price_df.index[len(price_df)-len(infer_rewards):len(price_df)]#np.array(list(range(len(infer_rewards))))
        infer_rewards = np.cumsum(np.array(infer_rewards))
        sensex_values = env.representative[-len(infer_steps):]
        plt.clf()
        plt.title(name)
        plt.plot(infer_steps, infer_rewards, color="red", label='Profit')
        plt.legend(loc="upper left")
        plt.savefig('rewards_'+a+'.jpg')
        print(action)
        print("the recommended actions on the stocks based on ",a," are:")
        for i in range(4):
                print(names[i],"\tbuy" if action[i]>0 else "\tsell")