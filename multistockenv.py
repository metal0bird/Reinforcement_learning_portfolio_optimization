CUDA_LAUNCH_BLOCKING=1 
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from gym import spaces
from gym.utils import seeding
from sklearn.preprocessing import StandardScaler
import talib

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000

matplotlib.use("Agg")

class MultiStockTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    def __init__(
        self,
        dfs,
        price_df,
        initial_amount,
        num_features,
        num_stocks,
        window_size,
        frame_bound,
        scalers=None,
        tech_indicator_list=[],
        trade_cost=0.05,
        reward_scaling=1e-5,
        suppresention_rate=0.755,
        representative=None
    ):
        if len(tech_indicator_list)!=0:
            num_features = len(tech_indicator_list)
        self.dfs = dfs
        self.price_df = price_df
        self.initial_amount = initial_amount
        self.margin = initial_amount
        self.portfolio = [0]*num_stocks
        self.PortfolioValue = 0
        self.reserve = initial_amount
        self.trade_cost = trade_cost
        self.state_space = num_features
        self.assets = num_stocks
        self.reward_scaling=reward_scaling
        self.tech_indicators = tech_indicator_list
        self.window_size = window_size
        self.frame_bound = frame_bound
        # spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_stocks,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_stocks,window_size,num_features), dtype=np.float32)
        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.price_df) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = np.zeros(self.assets)
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None
        self.rewards = []
        self.pvs = []
        if scalers == None:
            self.scalers = [None]*self.assets
        else:
            self.scalers =scalers
        self.representative = representative
        self.suppression_rate = suppresention_rate
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def process_data(self):
        signal_features = []
        for i in range(self.assets):
            df = self.dfs[i]
            start = self.frame_bound[0] - self.window_size
            end = self.frame_bound[1]
            if self.scalers[i]:
                current_scaler = self.scalers[i]
                signal_features_i = current_scaler.transform(df.loc[:, self.tech_indicators])[start:end]
            else:
                current_scaler = StandardScaler()
                signal_features_i = current_scaler.fit_transform(df.loc[:, self.tech_indicators])[start:end]
                self.scalers[i] = current_scaler
            signal_features.append(signal_features_i)

        self.prices = self.price_df.loc[:, :].to_numpy()[start:end]
        if self.representative:
            self.representative = self.price_df.loc[:, self.representative].to_numpy()[start:end]
        else:
            self.representative = self.price_df.loc[:, 'SENSEX'].to_numpy()[start:end]
        self.signal_features = np.array(signal_features)
        self._end_tick = len(self.prices)-1
        return self.prices, self.signal_features
    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._end_tick = len(self.prices)-1
        self._last_trade_tick = self._current_tick - 1
        self._position = np.zeros(self.assets)
        self._position_history = (self.window_size * [None]) + [self._position]
        self.margin = self.initial_amount
        self.portfolio = [0]*self.assets
        self.PortfolioValue = 0
        self.reserve = self.initial_amount
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        return self._get_observation()
    def _update_profit(self, ):
        self._total_profit = (self.PortfolioValue+self.reserve)/self.initial_amount
    def step(self, actions):
        self._done = False
        self._current_tick += 1
        if self._current_tick == self._end_tick:
            self._done = True
        #Get the current prices
        current_prices = self.prices[self._current_tick]
        current_prices[np.isnan(current_prices)] = 0
        current_prices_for_division = current_prices
        current_prices_for_division[current_prices_for_division == 0] = 1e9
        abs_portfolio_dist = abs(actions)
        N = int(np.round(abs_portfolio_dist.size*self.suppression_rate))
        abs_portfolio_dist[np.argpartition(abs_portfolio_dist,kth=N)[:N]] = 0
        self.margin = self.reserve + sum(self.portfolio*current_prices)
        norm_margin_pos = (abs_portfolio_dist/sum(abs_portfolio_dist))*self.margin
        next_positions = np.sign(actions)*norm_margin_pos
        change_in_positions = next_positions - self._position
        actions_in_market = np.divide(change_in_positions,current_prices_for_division).astype(int)
        new_portfolio = actions_in_market + self.portfolio
        new_pv = sum(new_portfolio*current_prices)
        new_reserve = self.margin - new_pv
        profit = (new_pv + new_reserve) - (self.PortfolioValue + self.reserve)
        cost = self.trade_cost*sum(abs(np.sign(actions_in_market)))
        self._position = next_positions
        self.portfolio = new_portfolio
        self.PortfolioValue = new_pv
        self.reserve = new_reserve - cost
        step_reward = profit - cost
        self._total_reward += self.reward_scaling*step_reward
        self.rewards.append(self._total_reward)
        self.pvs.append(new_pv)
        self._update_profit()
        self._position = next_positions
        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
        )
        self._update_history(info)
        if self.margin < 0:
            self._done = True
        return observation, step_reward, self._done, info
    def _get_observation(self):
        return np.nan_to_num(self.signal_features[:,(self._current_tick-self.window_size+1):self._current_tick+1,:])
    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}
        for key, value in info.items():
            self.history[key].append(value)
    def render(self, mode='human'):
        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.pvs)
        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )
        plt.pause(0.01)
        

    def render_all(self, mode='human'):
        plt.plot(self.pvs)
        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )
    def close(self):
        plt.close()
    def save_rendering(self, filepath):
        plt.savefig(filepath)
    def pause_rendering(self):
        plt.show()
    def _process_data(self):
        raise NotImplementedError
    def _calculate_reward(self, action):
        raise NotImplementedError
    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError
def add_features(tic_df):
    # Returns in the last t intervals
        for t in range(1, 11):
                tic_df[f'ret{t}min'] = tic_df['close'].div(tic_df['open'].shift(t-1)).sub(1)
        tic_df['sma'] = talib.SMA(tic_df['close'])
        tic_df['5sma'] = talib.SMA(tic_df['close'], timeperiod=5)
        tic_df['20sma'] = talib.SMA(tic_df['close'], timeperiod=20)
        tic_df['bb_upper'], tic_df['bb_middle'], tic_df['bb_lower'] = talib.BBANDS(tic_df['close'], matype=talib.MA_Type.T3)
        tic_df['bb_sell'] = (tic_df['close'] > tic_df['bb_upper'])*1
        tic_df['bb_buy'] = (tic_df['close'] < tic_df['bb_lower'])*1
        tic_df['bb_squeeze'] = (tic_df['bb_upper'] - tic_df['bb_lower'])/tic_df['bb_middle']
        tic_df['mom'] = talib.MOM(tic_df['close'], timeperiod=10)
        tic_df['adx'] = talib.ADX(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=10)
        tic_df['mfi'] = talib.MFI(tic_df['high'], tic_df['low'], tic_df['close'], tic_df['volume'], timeperiod=10)
        tic_df['rsi'] = talib.RSI(tic_df['close'], timeperiod=10)
        tic_df['trange'] = talib.TRANGE(tic_df['high'], tic_df['low'], tic_df['close'])
        tic_df['bop'] = talib.BOP(tic_df['open'], tic_df['high'], tic_df['low'], tic_df['close'])
        tic_df['cci'] = talib.CCI(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
        tic_df['STOCHRSI'] = talib.STOCHRSI(tic_df['close'],timeperiod=14,fastk_period=14,fastd_period=3,fastd_matype=0)[0]
        slowk, slowd = talib.STOCH(tic_df['high'], tic_df['low'], tic_df['close'], fastk_period=14,slowk_period=3,slowk_matype=0,slowd_period=3,slowd_matype=0)
        macd, macdsignal, macdhist = talib.MACD(tic_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        tic_df['slowk'] = slowk
        tic_df['slowd'] = slowd
        tic_df['macd'] = macd
        tic_df['macdsignal'] = macdsignal
        tic_df['macdhist'] = macdhist
        tic_df['NATR'] = talib.NATR(tic_df['high'].ffill(), tic_df['low'].ffill(), tic_df['close'].ffill())
        tic_df['KAMA'] = talib.KAMA(tic_df['close'], timeperiod=10)
        tic_df['MAMA'], tic_df['FAMA'] = talib.MAMA(tic_df['close'])
        tic_df['MAMA_buy'] = np.where((tic_df['MAMA'] < tic_df['FAMA']), 1, 0)
        tic_df['KAMA_buy'] = np.where((tic_df['close'] < tic_df['KAMA']), 1, 0)
        tic_df['sma_buy'] = np.where((tic_df['close'] < tic_df['5sma']), 1, 0)
        tic_df['maco'] = np.where((tic_df['5sma'] < tic_df['20sma']), 1, 0)
        tic_df['rsi_buy'] = np.where((tic_df['rsi'] < 30), 1, 0)
        tic_df['rsi_sell'] = np.where((tic_df['rsi'] > 70), 1, 0)
        tic_df['macd_buy_sell'] = np.where((tic_df['macd'] < tic_df['macdsignal']), 1, 0)
        return tic_df


